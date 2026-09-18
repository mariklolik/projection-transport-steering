from collections.abc import Mapping
from contextlib import nullcontext

import torch
from torch import Tensor

from projection_transport_steering.invariantback_runner import candidate_mean_log_likelihood
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

MAX_CANDIDATE_TOKENS = 256


def chat_prefix_ids(tokenizer: object, prompt: str) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()
    if not token_ids or isinstance(token_ids[0], list):
        raise ValueError("chat template returned invalid token identifiers")
    return [int(token) for token in token_ids]


def candidate_score_mask(candidate_mask: Tensor, last_token: bool) -> Tensor:
    if candidate_mask.ndim != 2 or torch.any(candidate_mask.sum(dim=1) == 0):
        raise ValueError("candidate mask is invalid")
    if not last_token:
        return candidate_mask
    result = torch.zeros_like(candidate_mask)
    result.scatter_(1, candidate_mask.sum(dim=1, keepdim=True) - 1, True)
    return result


def encode_candidates(
    tokenizer: object,
    prompt: str,
    candidates: list[str],
    device: torch.device,
) -> tuple[dict[str, Tensor], list[int]]:
    return encode_prompt_candidates(
        tokenizer,
        [prompt] * len(candidates),
        candidates,
        device,
    )


def encode_prompt_candidates(
    tokenizer: object,
    prompts: list[str],
    candidates: list[str],
    device: torch.device,
) -> tuple[dict[str, Tensor], list[int]]:
    if not prompts or len(prompts) != len(candidates):
        raise ValueError("prompts and candidates must be nonempty and aligned")
    prefixes = [chat_prefix_ids(tokenizer, prompt) for prompt in prompts]
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in candidates]
    original_lengths = [len(tokens) for tokens in encoded]
    encoded = [tokens[:MAX_CANDIDATE_TOKENS] for tokens in encoded]
    if any(not tokens for tokens in encoded):
        raise ValueError("candidate continuation is empty")
    width = max(len(prefix) + len(tokens) for prefix, tokens in zip(prefixes, encoded))
    candidate_width = max(map(len, encoded))
    input_ids = torch.full(
        (len(encoded), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    candidate_ids = torch.zeros(
        (len(encoded), candidate_width),
        dtype=torch.long,
        device=device,
    )
    candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
    prefix_lengths = []
    for index, (prefix, tokens) in enumerate(zip(prefixes, encoded, strict=True)):
        full = prefix + tokens
        prefix_lengths.append(len(prefix))
        input_ids[index, : len(full)] = torch.tensor(full, device=device)
        attention_mask[index, : len(full)] = 1
        candidate_ids[index, : len(tokens)] = torch.tensor(tokens, device=device)
        candidate_mask[index, : len(tokens)] = True
    return (
        {
            "attention_mask": attention_mask,
            "candidate_ids": candidate_ids,
            "candidate_mask": candidate_mask,
            "input_ids": input_ids,
            "prefix_lengths": torch.tensor(prefix_lengths, dtype=torch.long, device=device),
        },
        original_lengths,
    )


def sequence_scores(
    model: object,
    tokenizer: object,
    layer: int,
    prompt: str,
    candidates: list[str],
    device: torch.device,
    vector: Tensor | None = None,
    last_token: bool = False,
) -> tuple[list[Tensor], list[int]]:
    batch, lengths = encode_candidates(tokenizer, prompt, candidates, device)
    context = (
        TorchLayerAction(model, layer, AdditiveAction(vector)).installed()
        if vector is not None
        else nullcontext()
    )
    with context:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    score_mask = candidate_score_mask(batch["candidate_mask"], last_token)
    scores = [
        candidate_mean_log_likelihood(
            logits[index : index + 1],
            batch["candidate_ids"][index : index + 1],
            score_mask[index : index + 1],
            batch["prefix_lengths"][index : index + 1],
        )
        for index in range(len(candidates))
    ]
    return scores, lengths


def score_gradient(
    model: object,
    tokenizer: object,
    layer: int,
    prompt: str,
    candidates: list[str],
    device: torch.device,
    last_token: bool = False,
) -> tuple[Tensor, list[float], list[int]]:
    delta = torch.zeros(model.config.hidden_size, device=device, requires_grad=True)
    scores, lengths = sequence_scores(
        model,
        tokenizer,
        layer,
        prompt,
        candidates,
        device,
        vector=delta,
        last_token=last_token,
    )
    objective = scores[0] if len(scores) == 1 else scores[0] - scores[1]
    gradient = torch.autograd.grad(objective, delta)[0].float()
    return gradient, [float(score.detach()) for score in scores], lengths


def generate(
    model: object,
    tokenizer: object,
    layer: int,
    prompt: str,
    device: torch.device,
    vector: Tensor | None = None,
) -> str:
    return generate_many(model, tokenizer, layer, [prompt], device, vector)[0]


def generate_many(
    model: object,
    tokenizer: object,
    layer: int,
    prompts: list[str],
    device: torch.device,
    vector: Tensor | None = None,
) -> list[str]:
    prefixes = [chat_prefix_ids(tokenizer, prompt) for prompt in prompts]
    if not prefixes:
        raise ValueError("generation prompts are empty")
    width = max(map(len, prefixes))
    input_ids = torch.full(
        (len(prefixes), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, prefix in enumerate(prefixes):
        input_ids[index, -len(prefix) :] = torch.tensor(prefix, device=device)
        attention_mask[index, -len(prefix) :] = 1
    context = (
        TorchLayerAction(model, layer, AdditiveAction(vector)).installed()
        if vector is not None
        else nullcontext()
    )
    with torch.inference_mode(), context:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_CANDIDATE_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(row[width:], skip_special_tokens=True).strip() for row in output
    ]
