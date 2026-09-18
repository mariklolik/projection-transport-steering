import time
from typing import Any

import torch
from torch import Tensor

from projection_transport_steering.amortized_pullback import (
    GPT2Downstream,
    build_subspace,
    full_pullback_geometry,
    restricted_pullback_direction,
)
from projection_transport_steering.amortized_pullback_experiment import (
    median_positive_eigenvalue,
    random_orthonormal_basis,
)
from projection_transport_steering.torch_transport import restricted_metric_steering


LAYERS = (3, 6, 9, 11)
RANKS = (8, 16, 32, 64)
TOP_K = 5000
SEED = 20260906


def exact_direction(
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    fixed_regularization: float | None,
) -> dict[str, Any]:
    geometry = full_pullback_geometry(downstream, state, beta, unembedding, TOP_K)
    regularization = (
        median_positive_eigenvalue(geometry["metric"])
        if fixed_regularization is None
        else fixed_regularization
    )
    identity = torch.eye(state.numel(), device=state.device, dtype=state.dtype)
    direction = torch.linalg.solve(geometry["metric"] + regularization * identity, geometry["covector"])
    return {
        "covector": geometry["covector"],
        "direction": direction,
        "diagnostics": {
            "backward_calls": 1,
            "backward_equivalents": state.numel(),
            "inverse_quadratic": float(torch.dot(geometry["covector"], direction)),
            "jacobian_rows": state.numel(),
            "metric_vector_products": state.numel(),
            "partial_forward_calls": 2,
            "partial_forward_equivalents": 2,
            "regularization": regularization,
            "subspace_dimension": state.numel(),
        },
    }


def local_direction(
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    shared_basis: Tensor,
    regularization: float,
) -> dict[str, Any]:
    result = restricted_pullback_direction(
        downstream,
        state,
        beta,
        unembedding,
        shared_basis,
        regularization,
        TOP_K,
    )
    dimension = result["basis"].shape[1]
    return {
        "covector": result["covector"],
        "direction": result["direction"],
        "diagnostics": {
            "backward_calls": 2,
            "backward_equivalents": dimension + 1,
            "inverse_quadratic": float(result["inverse_quadratic"]),
            "jacobian_rows": 0,
            "metric_vector_products": dimension,
            "partial_forward_calls": 2,
            "partial_forward_equivalents": dimension + 1,
            "regularization": regularization,
            "subspace_dimension": dimension,
        },
    }


def proxy_direction(
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    shared_basis: Tensor | None,
    pooled: Tensor | None,
    regularization: float,
) -> dict[str, Any]:
    _, vjp = torch.func.vjp(downstream, state)
    covector = vjp(beta)[0]
    if shared_basis is None:
        return {
            "covector": covector,
            "direction": covector,
            "diagnostics": {
                "backward_calls": 1,
                "backward_equivalents": 1,
                "jacobian_rows": 0,
                "metric_vector_products": 0,
                "partial_forward_calls": 1,
                "partial_forward_equivalents": 1,
                "regularization": regularization,
                "subspace_dimension": 1,
            },
        }
    basis = build_subspace(covector, shared_basis)
    identity = torch.eye(state.numel(), device=state.device, dtype=state.dtype)
    metric = basis.T @ (pooled + regularization * identity) @ basis
    direction, inverse_quadratic = restricted_metric_steering(
        basis,
        metric,
        covector,
        1.0,
    )
    return {
        "covector": covector,
        "direction": direction,
        "diagnostics": {
            "backward_calls": 1,
            "backward_equivalents": 1,
            "inverse_quadratic": float(inverse_quadratic),
            "jacobian_rows": 0,
            "metric_vector_products": 0,
            "partial_forward_calls": 1,
            "partial_forward_equivalents": 1,
            "regularization": regularization,
            "subspace_dimension": basis.shape[1],
        },
    }


def method_specs(layer_packet: dict[str, Any], layer: int, device: torch.device) -> list[dict[str, Any]]:
    shared = layer_packet["basis"].to(device)
    pooled = layer_packet["pooled_metric"].to(device)
    random = random_orthonormal_basis(shared.shape[0], max(RANKS), SEED, shared.dtype, device)
    specs: list[dict[str, Any]] = [
        {"kind": "exact_fixed", "name": "exact_same_metric"},
        {"kind": "exact_local", "name": "fishback_published_reg"},
    ]
    specs.extend(
        {"basis": shared[:, :rank], "kind": "local", "name": f"shared_local_r{rank}"}
        for rank in RANKS
    )
    specs.extend(
        {"basis": random[:, :rank], "kind": "local", "name": f"random_local_r{rank}"}
        for rank in RANKS
    )
    specs.append({"kind": "q_only", "name": "q_only"})
    specs.extend(
        {
            "basis": shared[:, :rank],
            "kind": "pooled",
            "name": f"shared_pooled_r{rank}",
            "pooled": pooled,
        }
        for rank in RANKS
    )
    return specs


def method_direction(
    spec: dict[str, Any],
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    regularization: float,
) -> dict[str, Any]:
    if spec["kind"] == "exact_fixed":
        return exact_direction(downstream, state, beta, unembedding, regularization)
    if spec["kind"] == "exact_local":
        return exact_direction(downstream, state, beta, unembedding, None)
    if spec["kind"] == "local":
        return local_direction(
            downstream,
            state,
            beta,
            unembedding,
            spec["basis"],
            regularization,
        )
    return proxy_direction(
        downstream,
        state,
        beta,
        spec.get("basis"),
        spec.get("pooled"),
        regularization,
    )


def base_geometry(
    downstream: GPT2Downstream,
    state: Tensor,
    beta: Tensor,
    unembedding: Tensor,
    specs: list[dict[str, Any]],
    regularization: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    geometry = full_pullback_geometry(downstream, state, beta, unembedding, TOP_K)
    identity = torch.eye(state.numel(), device=state.device, dtype=state.dtype)
    metric = geometry["metric"] + regularization * identity
    full_direction = torch.linalg.solve(metric, geometry["covector"])
    full_inverse = torch.dot(geometry["covector"], full_direction)
    captures = {}
    for spec in specs:
        if spec["kind"] != "local":
            continue
        basis = build_subspace(geometry["covector"], spec["basis"])
        restricted = basis.T @ metric @ basis
        coordinates = basis.T @ geometry["covector"]
        inverse = torch.dot(coordinates, torch.linalg.solve(restricted, coordinates))
        captures[spec["name"]] = float(inverse / full_inverse)
    tensors = {
        "covector": geometry["covector"].cpu(),
        "metric": geometry["metric"].cpu(),
    }
    summary = {
        "captured_energy": captures,
        "full_inverse_quadratic": float(full_inverse),
        "geometry_seconds": time.perf_counter() - started,
        "metric_rank": int(torch.linalg.matrix_rank(geometry["metric"])),
    }
    return tensors, summary
