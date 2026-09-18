from contextlib import contextmanager

import torch
from torch import Tensor, nn

from projection_transport_steering.torch_runtime import TorchLayerAction


def supervised_example(
    tokenizer,
    messages,
    solution: str,
    answer: str,
    max_length: int,
    eos_token_ids: tuple[int, ...] = (),
) -> dict:
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    completed = tokenizer.apply_chat_template(
        messages
        + [
            {
                "role": "assistant",
                "reasoning_content": solution,
                "content": "\\boxed{" + answer + "}",
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    prompt = tokenizer.encode(prefix, add_special_tokens=False)
    tokens = tokenizer.encode(completed, add_special_tokens=False)
    if not prompt or tokens[: len(prompt)] != prompt:
        raise ValueError("native target changed the prompt prefix")
    if not solution or solution not in completed[len(prefix) :]:
        raise ValueError("native template changed the source solution")
    endings_allowed = {tokenizer.eos_token_id, *eos_token_ids}
    endings = [i for i in range(len(prompt), len(tokens)) if tokens[i] in endings_allowed]
    if len(endings) != 1 or tokenizer.decode(tokens[endings[0] + 1 :]).strip():
        raise ValueError("native target must have one EOS and only whitespace afterward")
    tokens = tokens[: endings[0] + 1]
    if len(tokens) > max_length:
        raise ValueError("native target exceeds the training length ceiling")
    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "labels": [-100] * len(prompt) + tokens[len(prompt) :],
        "prompt_length": len(prompt),
    }


def position_mask(attention_mask: Tensor, prompt_lengths: Tensor, application: str) -> Tensor:
    if (
        attention_mask.ndim != 2
        or prompt_lengths.shape != (len(attention_mask),)
        or prompt_lengths.dtype not in {torch.int32, torch.int64}
        or application not in {"prompt", "full"}
        or not torch.all((attention_mask == 0) | (attention_mask == 1))
        or torch.any(prompt_lengths <= 0)
        or torch.any(prompt_lengths > attention_mask.sum(-1))
    ):
        raise ValueError("invalid prompt or application dimensions")
    positions = attention_mask.cumsum(-1) - 1
    lengths = prompt_lengths[:, None]
    window = (lengths // 2).clamp_max(7)
    selected = (positions < window) | ((positions >= lengths - window) & (positions < lengths))
    if application == "full":
        selected |= positions >= lengths
    return selected & attention_mask.bool()


class MaskedAction(nn.Module):
    def __init__(self, action: nn.Module, mask: Tensor) -> None:
        super().__init__()
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("action mask must be a boolean matrix")
        self.action = action
        self.register_buffer("mask", mask)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[:2] != self.mask.shape:
            raise ValueError("action mask and hidden shape differ")
        if not torch.any(self.mask):
            return hidden
        output = hidden.clone()
        output[self.mask] = self.action(hidden[self.mask])
        return output


def epoch_batches(examples: list[dict], seed: int, epoch: int, batch_size: int) -> list[list[int]]:
    if not examples or batch_size < 1 or epoch < 0:
        raise ValueError("invalid epoch schedule")
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(examples), generator=generator).tolist()
    return [
        sorted(order[start : start + batch_size], key=lambda i: (-len(examples[i]["input_ids"]), i))
        for start in range(0, len(order), batch_size)
    ]


@contextmanager
def generation_intervention(model, action, site: int, application: str):
    if model.training or application not in {"prompt", "full"}:
        raise ValueError("invalid generation intervention policy")
    masked = MaskedAction(action, torch.empty(0, 0, dtype=torch.bool))

    def prepare(module, args, kwargs):
        valid = kwargs["attention_mask"]
        if kwargs.get("past_key_values") is None:
            masked.mask = position_mask(valid, valid.sum(-1), application)
        else:
            if kwargs["input_ids"].shape[1] != 1:
                raise ValueError("cached intervention requires one pending token")
            masked.mask = valid[:, -1:].bool() & (application == "full")

    handle = model.register_forward_pre_hook(prepare, with_kwargs=True)
    try:
        with TorchLayerAction(model, site, masked).installed():
            yield
    finally:
        handle.remove()


def fit_step(
    model,
    action,
    batch: dict | list[dict],
    site: int,
    application: str,
    optimizer,
    max_grad_norm: float | None = None,
) -> dict:
    parameters = list(action.parameters())
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if (
        {id(p) for p in owned} != {id(p) for p in parameters}
        or len(owned) != len(parameters)
        or any(p.requires_grad for p in model.parameters())
    ):
        raise ValueError("optimizer ownership must contain only the adapter")
    if max_grad_norm is not None and not 0 < max_grad_norm < float("inf"):
        raise ValueError("gradient norm limit must be finite and positive")
    batches = [batch] if isinstance(batch, dict) else list(batch)
    masks = [position_mask(b["attention_mask"], b["prompt_length"], application) for b in batches]
    count = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
    if count == 0:
        raise ValueError("the optimizer batch has no supervised targets")
    before = [p.detach().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for microbatch, mask in zip(batches, masks, strict=True):
        inputs = {key: value for key, value in microbatch.items() if key != "prompt_length"}
        with TorchLayerAction(model, site, MaskedAction(action, mask)).installed():
            loss = model(**inputs, use_cache=False, num_items_in_batch=count).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite fit loss")
            loss.backward()
        total_loss += float(loss.detach())
    gradients = [p.grad for p in parameters]
    if not all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in gradients):
        raise ValueError("missing, nonfinite or zero adapter gradient")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("frozen base accumulated a gradient")
    norm = float(torch.stack([g.float().norm() for g in gradients]).norm())
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    changed = sum(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True))
    if (not changed and any(group["lr"] != 0 for group in optimizer.param_groups)) or not all(
        torch.isfinite(p).all() for p in parameters
    ):
        raise ValueError("optimizer failed to make a finite adapter update")
    return {
        "loss": total_loss,
        "target_tokens": count,
        "active_positions": sum(int(mask.sum()) for mask in masks),
        "gradient_norm": norm,
        "changed_parameters": changed,
        "microbatches": len(batches),
    }
