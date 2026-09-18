# Model-agnostic inference primitives: generation and activation reading.
#
# Works for any HuggingFace causal LM whose decoder layers live at
# `model.model.layers[i]` (Gemma/Llama/Qwen-style). The model-specific bits
# (which weights, the chat template) live in models_specific/.
#
# `python -m general.inference` runs the light self-tests (no model).

from __future__ import annotations

import torch

# Gemma-2's creator-recommended sampling defaults; callers may override.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
# Reasoning traces need room for <think>...</think> + a boxed answer.
MAX_NEW_TOKENS = 320


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, seed: int | None = None,
             do_sample: bool = False, temperature: float = DEFAULT_TEMPERATURE,
             stop_strings: list[str] | None = None) -> str:
    """Generate a continuation for a pre-rendered `prompt`. Returns only the new text.

    `stop_strings` ends generation right after a literal string (e.g. "</think>"),
    which is how steering can treat the thinking and answering phases differently.
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id)
    if do_sample:
        kw.update(temperature=temperature, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K)
    if stop_strings:
        kw.update(stop_strings=stop_strings, tokenizer=tok)
    out = model.generate(**ids, **kw)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS,
                   seed: int | None = None, do_sample: bool = False,
                   temperature: float = DEFAULT_TEMPERATURE) -> list[str]:
    """Generate continuations for several prompts in one call.

    Main use: n copies of ONE prompt with do_sample=True — each batch row samples
    independently, which is exactly the "n random seeds" pattern.

    Padding note: left-padded batched generation is BROKEN on Apple MPS (padded
    rows emit empty text / NaN). So when prompts differ in length we pad+batch on
    CUDA/CPU (correct there) but fall back to a sequential loop on MPS. Equal
    length prompts need no padding and batch everywhere.
    """
    if seed is not None:
        torch.manual_seed(seed)
    lengths = {len(tok(p, add_special_tokens=False).input_ids) for p in prompts}
    needs_padding = len(lengths) > 1
    if needs_padding and model.device.type == "mps":
        return [generate(model, tok, p, max_new_tokens=max_new_tokens,
                         do_sample=do_sample, temperature=temperature) for p in prompts]

    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        ids = tok(prompts, return_tensors="pt", padding=needs_padding,
                  add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id)
        if do_sample:
            kw.update(temperature=temperature, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K)
        out = model.generate(**ids, **kw)
        new = out[:, ids["input_ids"].shape[1]:]
        return [tok.decode(row, skip_special_tokens=True) for row in new]
    finally:
        tok.padding_side = old_side


@torch.no_grad()
def next_token_logits(model, tok, text: str) -> torch.Tensor:
    """Logits for the NEXT token after `text` -> [vocab] on CPU, float32.

    The "read the distribution at a decision point" primitive: pass prompt + the
    model's own reasoning up to the answer position.
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    return model(**ids).logits[0, -1].float().cpu()


@torch.no_grad()
def _forward_last_logits(model, ids):
    """model(**ids).logits[:, -1] but only materializing the last position when supported."""
    try:
        return model(**ids, logits_to_keep=1).logits[:, -1]        # newer transformers
    except TypeError:
        pass
    try:
        return model(**ids, num_logits_to_keep=1).logits[:, -1]    # older transformers
    except TypeError:
        return model(**ids).logits[:, -1]                          # full logits fallback


def _length_buckets(tok, texts: list[str], batch_size: int) -> list[list[int]]:
    """Index chunks of `batch_size`, grouped by token length so padding stays ~0.

    Bucketing similar-length rows minimizes intra-batch padding, which keeps
    batched greedy decoding/readout numerically close to the unbatched path.
    """
    order = sorted(range(len(texts)), key=lambda i: len(tok(texts[i], add_special_tokens=False).input_ids))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def next_token_logits_batch(model, tok, texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """Next-token logits for many texts at once -> [N, vocab] on CPU, float32.

    Left-pads each chunk so the final real token sits at position -1 for every
    row (correct next-token readout under a causal mask), and buckets by length
    so padding is minimal. `logits_to_keep=1` avoids materializing the full
    [B, seq, vocab] tensor. Runs under whatever forward hooks are installed.
    """
    old_side = tok.padding_side
    tok.padding_side = "left"
    out: list[torch.Tensor | None] = [None] * len(texts)
    try:
        for idx in _length_buckets(tok, texts, batch_size):
            ids = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            logits = _forward_last_logits(model, ids).float().cpu()
            for row, j in enumerate(idx):
                out[j] = logits[row]
    finally:
        tok.padding_side = old_side
    return torch.stack(out)


@torch.no_grad()
def generate_batch_chunked(model, tok, prompts: list[str], batch_size: int = 64, **kw) -> list[str]:
    """generate_batch over `prompts` in length-bucketed chunks (bounds memory + padding)."""
    out: list[str | None] = [None] * len(prompts)
    for idx in _length_buckets(tok, prompts, batch_size):
        res = generate_batch(model, tok, [prompts[j] for j in idx], **kw)
        for r, j in zip(res, idx):
            out[j] = r
    return out


@torch.no_grad()
def get_activations(model, tok, text: str, layer: int, pos: str = "last") -> torch.Tensor:
    """Residual-stream activation at `layer` for `text` -> [d_model] on CPU.

    pos="last" -> final token; pos="mean" -> mean over all tokens.
    """
    store = {}

    def hook(_m, _inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        handle.remove()
    h = store["h"][0]
    return (h[-1] if pos == "last" else h.mean(0)).float().cpu()


@torch.no_grad()
def get_activations_all_layers(model, tok, text: str, pos: str = "last") -> torch.Tensor:
    """Residual activations at EVERY layer in one forward -> [n_layers, d_model] on CPU."""
    store: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            store[i] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.model.layers)]
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([(store[i][0][-1] if pos == "last" else store[i][0].mean(0)).float().cpu()
                        for i in range(len(model.model.layers))])


@torch.no_grad()
def get_trace_activations(model, tok, prompt: str, generation: str) -> torch.Tensor:
    """Mean residual over the GENERATED tokens (the reasoning trace), every layer.

    Returns [n_layers, d_model] on CPU. One forward over prompt+generation; the
    pooled span is everything after the prompt tokens (feature extraction reads
    the model's own <think> trace, not the question).
    """
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    store: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            store[i] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.model.layers)]
    try:
        ids = tok(prompt + generation, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        for h in handles:
            h.remove()
    vecs = []
    for i in range(len(model.model.layers)):
        span = store[i][0][n_prompt:]
        vecs.append((span.mean(0) if len(span) > 0 else store[i][0][-1]).float().cpu())
    return torch.stack(vecs)


@torch.no_grad()
def get_trace_token_activations_multi(model, tok, prompt: str, generation: str,
                                      layers: list[int]) -> dict[int, torch.Tensor]:
    """Per-token residuals over the GENERATED span at SEVERAL layers, one forward.

    Returns {layer: [n_gen_tokens, d_model] on CPU}. Hooks only `layers`. Used by
    the SAE depth sweep: generate the trace once, read every needed layer in a
    single forward, then encode each through its own SAE.
    """
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    store: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            store[i] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        ids = tok(prompt + generation, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for L in layers:
        span = store[L][0][n_prompt:]
        out[L] = (span if len(span) > 0 else store[L][0][-1:]).float().cpu()
    return out


@torch.no_grad()
def get_trace_token_activations(model, tok, prompt: str, generation: str, layer: int) -> torch.Tensor:
    """Per-token residuals at `layer` over the GENERATED span -> [n_gen_tokens, d_model] on CPU.

    Unlike get_trace_activations (which mean-pools), this keeps every trace
    token, so an SAE can encode each token and the feature activations be pooled
    correctly (encode-then-mean, not mean-then-encode — the SAE is nonlinear).
    """
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    store = {}

    def hook(_m, _inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        ids = tok(prompt + generation, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        handle.remove()
    span = store["h"][0][n_prompt:]
    return (span if len(span) > 0 else store["h"][0][-1:]).float().cpu()


if __name__ == "__main__":
    # Only the pad-decision logic is model-free; exercise it directly.
    assert MAX_NEW_TOKENS > 0 and 0 < DEFAULT_TOP_P <= 1
    print("general.inference import + constants OK (model-dependent paths need the model)")
