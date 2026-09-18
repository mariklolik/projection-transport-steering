from collections.abc import Iterator
from contextlib import contextmanager
import math

import torch
from torch import Tensor, nn


def _resolve_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers"))
    for path in candidates:
        value: object = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if isinstance(value, nn.ModuleList):
            return value
    raise ValueError("no supported ordered decoder layer collection was found")


def _hidden(output: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("decoder layer output does not contain hidden states")


def _replace_hidden(
    output: Tensor | tuple[Tensor, ...] | list[Tensor],
    hidden: Tensor,
) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return [hidden, *output[1:]]


class AdditiveAction(nn.Module):
    def __init__(self, vector: Tensor) -> None:
        super().__init__()
        if vector.ndim != 1:
            raise ValueError("additive action must be a vector")
        self.vector: Tensor
        self.register_buffer("vector", vector)

    def forward(self, hidden: Tensor) -> Tensor:
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if vector.shape[0] != hidden.shape[-1]:
            raise ValueError("action dimension does not match hidden dimension")
        return hidden + vector


class NormMatchedDirectionAction(nn.Module):
    def __init__(self, reference: nn.Module, direction: Tensor) -> None:
        super().__init__()
        if direction.ndim != 1 or direction.norm() == 0:
            raise ValueError("null direction must be a nonzero vector")
        self.reference = reference
        self.direction: Tensor
        self.register_buffer("direction", direction / direction.norm())

    def forward(self, hidden: Tensor) -> Tensor:
        reference_delta = self.reference(hidden) - hidden
        magnitudes = reference_delta.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        return hidden + magnitudes * direction


class ProjectedActionValueObserver(nn.Module):
    def __init__(
        self,
        projection: Tensor,
        mean: Tensor,
        scale: Tensor,
        coefficients: Tensor,
        intercept: Tensor,
        reference_action: int = 0,
    ) -> None:
        super().__init__()
        if (
            projection.ndim != 2
            or mean.shape != (projection.shape[0],)
            or scale.shape != mean.shape
            or coefficients.ndim != 2
            or coefficients.shape[1] != projection.shape[0]
            or intercept.shape != (coefficients.shape[0],)
            or torch.any(scale <= 0)
            or reference_action < 0
            or reference_action > coefficients.shape[0]
        ):
            raise ValueError("projected action-value parameters are invalid")
        self.projection: Tensor
        self.mean: Tensor
        self.scale: Tensor
        self.coefficients: Tensor
        self.intercept: Tensor
        self.reference_action = reference_action
        self.register_buffer("projection", projection.float())
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float())
        self.register_buffer("coefficients", coefficients.float())
        self.register_buffer("intercept", intercept.float())

    def forward(self, hidden: Tensor) -> Tensor:
        values = hidden.float()
        projected = values @ self.projection.to(values.device).transpose(0, 1)
        standardized = (projected - self.mean.to(values.device)) / self.scale.to(values.device)
        differences = standardized @ self.coefficients.to(values.device).transpose(
            0, 1
        ) + self.intercept.to(values.device)
        losses = torch.empty(
            (len(values), differences.shape[1] + 1),
            device=values.device,
        )
        losses[:, self.reference_action] = 0.0
        actions = torch.arange(losses.shape[1], device=values.device)
        losses[:, actions != self.reference_action] = differences
        return losses


class ThresholdedActionValueObserver(nn.Module):
    def __init__(
        self,
        observer: nn.Module,
        reference_action: int,
        threshold: float,
    ) -> None:
        super().__init__()
        if reference_action < 0 or math.isnan(threshold) or threshold < 0.0:
            raise ValueError("thresholded observer parameters are invalid")
        self.observer = observer
        self.reference_action = reference_action
        self.threshold = threshold

    def forward(self, hidden: Tensor) -> Tensor:
        losses = self.observer(hidden)
        if (
            losses.ndim != 2
            or self.reference_action >= losses.shape[1]
            or not torch.all(torch.isfinite(losses))
        ):
            raise ValueError("thresholded observer output is invalid")
        rows = torch.arange(len(losses), device=losses.device)
        best = torch.argmin(losses, dim=1)
        gain = losses[:, self.reference_action] - losses[rows, best]
        fallback = gain < self.threshold
        if not torch.any(fallback):
            return losses
        result = losses.clone()
        result[fallback] = torch.inf
        result[fallback, self.reference_action] = losses[fallback, self.reference_action]
        return result


class TorchLayerAction:
    def __init__(self, model: nn.Module, site: int, action: nn.Module) -> None:
        layers = _resolve_layers(model)
        if site < 0 or site >= len(layers):
            raise ValueError("action site is outside the decoder")
        self.layer = layers[site]
        self.action = action

    def _act(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
        hidden = _hidden(output)
        steered = self.action(hidden)
        if steered.shape != hidden.shape:
            raise ValueError("action changed the hidden-state shape")
        return _replace_hidden(output, steered)

    @contextmanager
    def installed(self) -> Iterator["TorchLayerAction"]:
        handle = self.layer.register_forward_hook(self._act)
        try:
            yield self
        finally:
            handle.remove()


class TorchSameForwardController:
    def __init__(
        self,
        model: nn.Module,
        observer_site: int,
        action_site: int,
        observer: nn.Module,
        action_vectors: Tensor,
    ) -> None:
        layers = _resolve_layers(model)
        if observer_site < 0 or action_site <= observer_site or action_site >= len(layers):
            raise ValueError("observer site must precede a valid action site")
        if action_vectors.ndim != 2 or action_vectors.shape[0] == 0:
            raise ValueError("action vectors must be a nonempty matrix")
        self.model = model
        self.observer = observer
        self.action_vectors = action_vectors
        self.observer_layer = layers[observer_site]
        self.action_layer = layers[action_site]
        self.trajectory_count = 0
        self.forward_count = 0
        self._selected_actions: Tensor | None = None

    @property
    def selected_actions(self) -> Tensor:
        if self._selected_actions is None:
            raise RuntimeError("no action has been selected")
        return self._selected_actions

    def start_trajectory(self) -> None:
        self.trajectory_count += 1
        self.forward_count = 0
        self._selected_actions = None

    def _count_forward(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        self.forward_count += 1

    def _observe(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> None:
        if self._selected_actions is not None:
            return
        hidden = _hidden(output)
        predicted_losses = self.observer(hidden[:, -1, :])
        expected = (hidden.shape[0], self.action_vectors.shape[0])
        if predicted_losses.shape != expected:
            raise ValueError("observer output does not match batch and action counts")
        self._selected_actions = torch.argmin(predicted_losses, dim=1)

    def _act(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
        hidden = _hidden(output)
        selected = self.selected_actions
        if selected.shape[0] != hidden.shape[0]:
            raise ValueError("decode batch differs from prefill batch")
        vectors = self.action_vectors.to(device=hidden.device, dtype=hidden.dtype)[selected]
        if vectors.shape[1] != hidden.shape[2]:
            raise ValueError("action dimension does not match hidden dimension")
        return _replace_hidden(output, hidden + vectors[:, None, :])

    @contextmanager
    def installed(self) -> Iterator["TorchSameForwardController"]:
        handles = (
            self.model.register_forward_pre_hook(self._count_forward),
            self.observer_layer.register_forward_hook(self._observe),
            self.action_layer.register_forward_hook(self._act),
        )
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()


class TorchMaskedActionController:
    def __init__(
        self,
        model: nn.Module,
        observer_site: int,
        action_site: int,
        observer: nn.Module,
        actions: list[nn.Module],
    ) -> None:
        layers = _resolve_layers(model)
        if observer_site < 0 or action_site <= observer_site or action_site >= len(layers):
            raise ValueError("observer site must precede a valid action site")
        if len(actions) < 2:
            raise ValueError("masked controller requires at least two actions")
        self.model = model
        self.observer = observer
        self.actions = nn.ModuleList(actions)
        self.observer_layer = layers[observer_site]
        self.action_layer = layers[action_site]
        self.trajectory_count = 0
        self.forward_count = 0
        self._prompt_indices: Tensor | None = None
        self._action_mask: Tensor | None = None
        self._selection_groups: Tensor | None = None
        self._selected_actions: Tensor | None = None

    @property
    def selected_actions(self) -> Tensor:
        if self._selected_actions is None:
            raise RuntimeError("no action has been selected")
        return self._selected_actions

    def start_trajectory(
        self,
        prompt_indices: Tensor,
        action_mask: Tensor,
        selection_groups: Tensor | None = None,
    ) -> None:
        groups = (
            torch.arange(len(prompt_indices), device=prompt_indices.device)
            if selection_groups is None
            else selection_groups
        )
        if (
            prompt_indices.ndim != 1
            or action_mask.ndim != 2
            or prompt_indices.shape[0] != action_mask.shape[0]
            or action_mask.dtype != torch.bool
            or groups.shape != prompt_indices.shape
        ):
            raise ValueError("prompt indices, action mask, and selection groups are invalid")
        self.trajectory_count += 1
        self.forward_count = 0
        self._prompt_indices = prompt_indices
        self._action_mask = action_mask
        self._selection_groups = groups
        self._selected_actions = None

    def _count_forward(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        self.forward_count += 1

    def _observe(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> None:
        if self._selected_actions is not None:
            return
        if (
            self._prompt_indices is None
            or self._action_mask is None
            or self._selection_groups is None
        ):
            raise RuntimeError("trajectory has not been started")
        hidden = _hidden(output)
        indices = self._prompt_indices.to(hidden.device)
        if (
            self._action_mask.shape != hidden.shape[:2]
            or torch.any(indices < 0)
            or torch.any(indices >= hidden.shape[1])
        ):
            raise ValueError("trajectory mask does not match hidden states")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        predicted_losses = self.observer(hidden[rows, indices])
        expected = (hidden.shape[0], len(self.actions))
        if predicted_losses.shape != expected:
            raise ValueError("observer output does not match batch and action counts")
        selected = torch.argmin(predicted_losses, dim=1)
        groups = self._selection_groups.to(hidden.device)
        for group in torch.unique(groups):
            rows = groups == group
            selected[rows] = selected[rows][0]
        self._selected_actions = selected

    def _act(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
        if self._action_mask is None:
            raise RuntimeError("trajectory has not been started")
        hidden = _hidden(output)
        selected = self.selected_actions.to(hidden.device)
        mask = self._action_mask.to(hidden.device)
        if selected.shape != (hidden.shape[0],) or mask.shape != hidden.shape[:2]:
            raise ValueError("selected actions do not match hidden states")
        steered = hidden.clone()
        for action_index, action in enumerate(self.actions):
            rows = selected == action_index
            if not torch.any(rows):
                continue
            candidate = action(hidden[rows])
            if candidate.shape != hidden[rows].shape:
                raise ValueError("action changed the hidden-state shape")
            steered[rows] = torch.where(
                mask[rows].unsqueeze(-1),
                candidate,
                hidden[rows],
            )
        return _replace_hidden(output, steered)

    @contextmanager
    def installed(self) -> Iterator["TorchMaskedActionController"]:
        handles = (
            self.model.register_forward_pre_hook(self._count_forward),
            self.observer_layer.register_forward_hook(self._observe),
            self.action_layer.register_forward_hook(self._act),
        )
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
