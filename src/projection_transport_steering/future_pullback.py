from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from transformers import DynamicCache


class CachePathDownstream(nn.Module):
    def __init__(
        self,
        downstream: Callable[[Tensor], Tensor],
        base_state: Tensor,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"reset", "cache_only"}:
            raise ValueError("cache path mode is invalid")
        self.downstream = downstream
        self.mode = mode
        self.register_buffer("base_outputs", downstream(base_state).detach())

    def forward(self, state: Tensor) -> Tensor:
        outputs = self.downstream(state)
        if self.mode == "reset":
            return torch.cat((outputs[:1], self.base_outputs[1:]))
        return torch.cat((self.base_outputs[:1], outputs[1:]))


class GPT2TeacherForcedDownstream(nn.Module):
    def __init__(
        self,
        model: Any,
        layer: int,
        hidden_states: Tensor,
        position: int,
        attention_mask: Tensor | None,
    ) -> None:
        super().__init__()
        if (
            hidden_states.ndim != 3
            or not 0 <= position < hidden_states.shape[1]
            or not 0 <= layer < len(model.transformer.h)
        ):
            raise ValueError("model, layer, hidden states, or position is invalid")
        self.model = model
        self.layer = layer
        self.position = position
        self.register_buffer("hidden_states", hidden_states.detach())
        self.register_buffer(
            "attention_mask",
            None if attention_mask is None else attention_mask.detach(),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.shape != (self.hidden_states.shape[2],) or not torch.all(torch.isfinite(state)):
            raise ValueError("state is invalid")
        hidden = torch.cat(
            (
                self.hidden_states[:, : self.position],
                state.reshape(1, 1, -1).expand(self.hidden_states.shape[0], -1, -1),
                self.hidden_states[:, self.position + 1 :],
            ),
            dim=1,
        )
        for block in self.model.transformer.h[self.layer + 1 :]:
            output = block(hidden, None, self.attention_mask, use_cache=False)
            hidden = output if isinstance(output, Tensor) else output[0]
        return self.model.transformer.ln_f(hidden)[:, self.position :].flatten(0, 1)


class GPT2IncrementalDownstream(nn.Module):
    def __init__(
        self,
        model: Any,
        layer: int,
        hidden_states: Tensor,
        position: int,
        attention_mask: Tensor | None,
    ) -> None:
        super().__init__()
        if hidden_states.shape[0] != 1:
            raise ValueError("incremental cache replay requires one trajectory")
        self.model = model
        self.layer = layer
        self.position = position
        self.register_buffer("hidden_states", hidden_states.detach())
        self.register_buffer(
            "attention_mask",
            None if attention_mask is None else attention_mask.detach(),
        )

    def _mask(self, query: slice, key_stop: int) -> Tensor | None:
        if self.attention_mask is None:
            return None
        return self.attention_mask[..., query, :key_stop]

    def forward(self, state: Tensor) -> Tensor:
        cache = DynamicCache(config=self.model.config)
        hidden = torch.cat(
            (
                self.hidden_states[:, : self.position],
                state.reshape(1, 1, -1),
            ),
            dim=1,
        )
        prefix_stop = self.position + 1
        for block in self.model.transformer.h[self.layer + 1 :]:
            hidden = block(
                hidden,
                cache,
                self._mask(slice(0, prefix_stop), prefix_stop),
                use_cache=True,
            )
        outputs = [self.model.transformer.ln_f(hidden[:, -1:])]
        for offset in range(1, self.hidden_states.shape[1] - self.position):
            hidden = self.hidden_states[:, self.position + offset : self.position + offset + 1]
            key_stop = self.position + offset + 1
            for block in self.model.transformer.h[self.layer + 1 :]:
                hidden = block(
                    hidden,
                    cache,
                    self._mask(slice(self.position + offset, key_stop), key_stop),
                    use_cache=True,
                )
            outputs.append(self.model.transformer.ln_f(hidden))
        return torch.cat(outputs, dim=1)[0]


def prepare_gpt2_teacher_forced(
    model: Any,
    input_ids: Tensor,
    layer: int,
    position: int,
) -> dict[str, Any]:
    if input_ids.ndim != 2 or input_ids.shape[0] < 1:
        raise ValueError("input ids are invalid")
    captured: list[tuple[Tensor, Tensor | None]] = []

    def capture(_module: Any, inputs: Any, kwargs: Any, output: Any) -> None:
        attention_mask = inputs[2] if len(inputs) > 2 else kwargs.get("attention_mask")
        captured.append((output if isinstance(output, Tensor) else output[0], attention_mask))

    handle = model.transformer.h[layer].register_forward_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"captured {len(captured)} layer outputs")
    hidden_states, attention_mask = captured[0]
    states = hidden_states[:, position]
    if not torch.allclose(states, states[:1].expand_as(states), atol=1e-6, rtol=1e-5):
        raise RuntimeError("teacher-forced trajectories do not share the intervention state")
    downstream = GPT2TeacherForcedDownstream(
        model,
        layer,
        hidden_states,
        position,
        attention_mask,
    )
    incremental_downstream = None
    if input_ids.shape[0] == 1:
        incremental_downstream = GPT2IncrementalDownstream(
            model,
            layer,
            hidden_states,
            position,
            attention_mask,
        )
    state = states[0].detach()
    offset_count = input_ids.shape[1] - position
    base_logits = outputs.logits[:, position:].flatten(0, 1).detach()
    with torch.no_grad():
        replayed = model.lm_head(downstream(state))
    tolerance = 1e-4 + 1e-5 * base_logits.abs().max()
    if (replayed - base_logits).abs().max() > tolerance:
        raise RuntimeError("teacher-forced downstream replay failed")
    return {
        "base_logits": base_logits,
        "downstream": downstream,
        "incremental_downstream": incremental_downstream,
        "offset_count": offset_count,
        "state": state,
        "trajectory_count": input_ids.shape[0],
    }


def full_vocabulary_fisher(representation: Tensor, unembedding: Tensor) -> dict[str, Tensor]:
    if (
        representation.ndim != 1
        or unembedding.ndim != 2
        or unembedding.shape[1] != representation.shape[0]
        or not torch.all(torch.isfinite(representation))
        or not torch.all(torch.isfinite(unembedding))
    ):
        raise ValueError("representation or unembedding is invalid")
    representation = representation.double()
    unembedding = unembedding.double()
    logits = unembedding @ representation
    probabilities = torch.softmax(logits, dim=0)
    mean = probabilities @ unembedding
    centered = unembedding - mean
    metric = centered.transpose(0, 1) @ (probabilities.unsqueeze(1) * centered)
    return {
        "logits": logits,
        "metric": 0.5 * (metric + metric.transpose(0, 1)),
        "probabilities": probabilities,
    }


def topk_vocabulary_fisher(
    representation: Tensor,
    unembedding: Tensor,
    top_k: int,
) -> dict[str, Tensor]:
    if (
        representation.ndim != 1
        or unembedding.ndim != 2
        or unembedding.shape[1] != representation.shape[0]
        or not 1 < top_k <= unembedding.shape[0]
        or not torch.all(torch.isfinite(representation))
        or not torch.all(torch.isfinite(unembedding))
    ):
        raise ValueError("representation, unembedding, or top-k is invalid")
    representation = representation.double()
    unembedding = unembedding.double()
    logits = unembedding @ representation
    selected_logits, selected_ids = logits.topk(top_k)
    probabilities = torch.softmax(selected_logits, dim=0)
    selected = unembedding[selected_ids]
    mean = probabilities @ selected
    centered = selected - mean
    metric = centered.transpose(0, 1) @ (probabilities.unsqueeze(1) * centered)
    return {
        "logits": logits,
        "metric": 0.5 * (metric + metric.transpose(0, 1)),
        "probabilities": probabilities,
        "selected_ids": selected_ids,
    }


def future_pullback_geometry(
    downstream: Callable[[Tensor], Tensor],
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    top_k: int | None = None,
) -> dict[str, Any]:
    representations = downstream(state)
    if (
        state.ndim != 1
        or representations.ndim != 2
        or beta.shape != (representations.shape[1],)
        or unembedding.ndim != 2
        or unembedding.shape[1] != representations.shape[1]
        or not torch.all(torch.isfinite(state))
        or not torch.all(torch.isfinite(representations))
        or not torch.all(torch.isfinite(beta))
        or not torch.all(torch.isfinite(unembedding))
    ):
        raise ValueError("future pullback inputs are invalid")
    try:
        jacobians = torch.func.jacfwd(downstream)(state).double()
        automatic_differentiation = "forward_mode"
    except NotImplementedError:
        jacobians = torch.autograd.functional.jacobian(
            downstream,
            state,
            vectorize=True,
        ).double()
        automatic_differentiation = "reverse_mode_fallback"
    fisher = full_vocabulary_fisher if top_k is None else (
        lambda value, matrix: topk_vocabulary_fisher(value, matrix, top_k)
    )
    fishers = torch.stack([fisher(value, unembedding)["metric"] for value in representations])
    terms = torch.stack(
        [
            jacobian.transpose(0, 1) @ fisher @ jacobian
            for jacobian, fisher in zip(jacobians, fishers, strict=True)
        ]
    )
    terms = 0.5 * (terms + terms.transpose(1, 2))
    return {
        "automatic_differentiation": automatic_differentiation,
        "covector": jacobians[0].transpose(0, 1) @ beta.double(),
        "fishers": fishers,
        "jacobians": jacobians,
        "representations": representations,
        "terms": terms,
        "top_k": top_k,
    }
