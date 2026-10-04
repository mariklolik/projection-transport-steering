from __future__ import annotations

import torch


DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

MAX_NEW_TOKENS = 320


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, seed: int | None = None,
             do_sample: bool = False, temperature: float = DEFAULT_TEMPERATURE,
             stop_strings: list[str] | None = None) -> str:

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

    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    return model(**ids).logits[0, -1].float().cpu()


@torch.no_grad()
def _forward_last_logits(model, ids):

    try:
        return model(**ids, logits_to_keep=1).logits[:, -1]
    except TypeError:
        pass
    try:
        return model(**ids, num_logits_to_keep=1).logits[:, -1]
    except TypeError:
        return model(**ids).logits[:, -1]


def _length_buckets(tok, texts: list[str], batch_size: int) -> list[list[int]]:

    order = sorted(range(len(texts)), key=lambda i: len(tok(texts[i], add_special_tokens=False).input_ids))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def next_token_logits_batch(model, tok, texts: list[str], batch_size: int = 64) -> torch.Tensor:

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

    out: list[str | None] = [None] * len(prompts)
    for idx in _length_buckets(tok, prompts, batch_size):
        res = generate_batch(model, tok, [prompts[j] for j in idx], **kw)
        for r, j in zip(res, idx):
            out[j] = r
    return out


@torch.no_grad()
def get_activations(model, tok, text: str, layer: int, pos: str = "last") -> torch.Tensor:

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

    assert MAX_NEW_TOKENS > 0 and 0 < DEFAULT_TOP_P <= 1
    print("general.inference import + constants OK (model-dependent paths need the model)")
