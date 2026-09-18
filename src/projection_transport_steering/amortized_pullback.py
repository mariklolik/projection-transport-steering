from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from projection_transport_steering.torch_transport import restricted_metric_steering


class GPT2Downstream(nn.Module):
    def __init__(self, model: Any, layer: int, hidden_states: Tensor) -> None:
        super().__init__()
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("hidden states are invalid")
        self.model = model
        self.layer = layer
        self.register_buffer("prefix", hidden_states[:, :-1].detach())

    def forward(self, last_state: Tensor) -> Tensor:
        hidden = torch.cat((self.prefix, last_state.reshape(1, 1, -1)), dim=1)
        for block in self.model.transformer.h[self.layer + 1 :]:
            output = block(hidden, use_cache=False)
            hidden = output if isinstance(output, Tensor) else output[0]
        return self.model.transformer.ln_f(hidden)[0, -1]


def build_subspace(covector: Tensor, shared_basis: Tensor) -> Tensor:
    if (
        covector.ndim != 1
        or shared_basis.ndim != 2
        or shared_basis.shape[0] != covector.shape[0]
        or not torch.all(torch.isfinite(covector))
        or not torch.all(torch.isfinite(shared_basis))
        or covector.norm() == 0
    ):
        raise ValueError("covector and shared basis are invalid")
    matrix = torch.cat(((covector / covector.norm()).unsqueeze(1), shared_basis), dim=1)
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = singular_values[0] * max(matrix.shape) * torch.finfo(singular_values.dtype).eps
    rank = int((singular_values > tolerance).sum())
    return left[:, :rank]


def topk_fisher_factors(
    representation: Tensor,
    unembedding: Tensor,
    top_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        representation.ndim != 1
        or unembedding.ndim != 2
        or unembedding.shape[1] != representation.shape[0]
        or not isinstance(top_k, int)
        or not 1 < top_k <= unembedding.shape[0]
    ):
        raise ValueError("representation, unembedding, or top_k is invalid")
    logits = unembedding @ representation
    probabilities = torch.softmax(logits.float(), dim=-1).to(representation.dtype)
    selected_probabilities, selected_ids = probabilities.topk(top_k)
    selected_probabilities = selected_probabilities / selected_probabilities.sum()
    selected_unembedding = unembedding[selected_ids]
    mean = selected_probabilities @ selected_unembedding
    centered = selected_unembedding - mean
    return logits, selected_probabilities, centered


def full_pullback_geometry(
    downstream: Callable[[Tensor], Tensor],
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    top_k: int,
) -> dict[str, Tensor]:
    representation = downstream(state)
    logits, probabilities, centered = topk_fisher_factors(
        representation,
        unembedding,
        top_k,
    )
    jacobian = torch.autograd.functional.jacobian(
        downstream,
        state,
        vectorize=True,
    )
    fisher = centered.transpose(0, 1) @ (probabilities.unsqueeze(1) * centered)
    covector = jacobian.transpose(0, 1) @ beta
    metric = jacobian.transpose(0, 1) @ fisher @ jacobian
    return {
        "covector": covector,
        "fisher": fisher,
        "jacobian": jacobian,
        "logits": logits,
        "metric": 0.5 * (metric + metric.transpose(0, 1)),
        "representation": representation,
    }


def matrix_free_pullback_system(
    downstream: Callable[[Tensor], Tensor],
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    top_k: int,
) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    if (
        state.ndim != 1
        or beta.ndim != 1
        or not torch.all(torch.isfinite(state))
        or not torch.all(torch.isfinite(beta))
    ):
        raise ValueError("state or beta is invalid")
    representation, vjp = torch.func.vjp(downstream, state)
    if beta.shape != representation.shape:
        raise ValueError("beta shape is invalid")
    covector = vjp(beta)[0]
    _, probabilities, centered = topk_fisher_factors(representation, unembedding, top_k)

    def metric_product(vector: Tensor) -> Tensor:
        if vector.shape != state.shape or not torch.all(torch.isfinite(vector)):
            raise ValueError("metric vector is invalid")
        tangent = torch.func.jvp(downstream, (state,), (vector,))[1]
        fisher_tangent = centered.transpose(0, 1) @ (
            probabilities * (centered @ tangent)
        )
        product = vjp(fisher_tangent)[0]
        if product.shape != state.shape or not torch.all(torch.isfinite(product)):
            raise RuntimeError("metric product is invalid")
        return product

    return covector, metric_product


def restricted_pullback_direction(
    downstream: Callable[[Tensor], Tensor],
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    shared_basis: Tensor,
    regularization: float,
    top_k: int,
) -> dict[str, Tensor]:
    representation, vjp = torch.func.vjp(downstream, state)
    covector = vjp(beta)[0]
    basis = build_subspace(covector, shared_basis)
    _, probabilities, centered = topk_fisher_factors(representation, unembedding, top_k)
    tangents = torch.vmap(
        lambda vector: torch.func.jvp(downstream, (state,), (vector,))[1]
    )(basis.transpose(0, 1))
    fisher_tangents = (tangents @ centered.transpose(0, 1) * probabilities) @ centered
    metric_products = torch.vmap(vjp)(fisher_tangents)[0]
    metric = basis.transpose(0, 1) @ metric_products.transpose(0, 1)
    identity = torch.eye(metric.shape[0], device=metric.device, dtype=metric.dtype)
    metric = 0.5 * (metric + metric.transpose(0, 1)) + regularization * identity
    direction, inverse_quadratic = restricted_metric_steering(
        basis,
        metric,
        covector,
        1.0,
    )
    return {
        "covector": covector,
        "basis": basis,
        "direction": direction,
        "inverse_quadratic": inverse_quadratic,
        "metric": metric,
        "representation": representation,
    }
