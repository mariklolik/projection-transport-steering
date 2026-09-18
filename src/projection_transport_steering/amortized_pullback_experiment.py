import math
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def concept_probability(logits: Tensor, base_ids: list[int], target_ids: list[int]) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    base = probabilities[base_ids].sum()
    target = probabilities[target_ids].sum()
    return float(target / (base + target))


def off_target_kl(base_logits: Tensor, steered_logits: Tensor, concept_ids: list[int]) -> float:
    mask = torch.ones(base_logits.shape[-1], dtype=torch.bool, device=base_logits.device)
    mask[concept_ids] = False
    base = torch.log_softmax(base_logits.float()[mask], dim=-1)
    steered = torch.log_softmax(steered_logits.float()[mask], dim=-1)
    return float((base.exp() * (base - steered)).sum())


def fixed_efficiency_direction(raw_direction: Tensor, covector: Tensor, efficiency: float) -> Tensor:
    if (
        raw_direction.ndim != 1
        or covector.shape != raw_direction.shape
        or not 0.0 < efficiency <= 1.0
        or not torch.all(torch.isfinite(raw_direction))
        or not torch.all(torch.isfinite(covector))
        or covector.norm() == 0
    ):
        raise ValueError("direction, covector, or efficiency is invalid")
    normalized_covector = covector / covector.norm()
    orthogonal = raw_direction - torch.dot(raw_direction, normalized_covector) * normalized_covector
    if orthogonal.norm() < 1e-10:
        return normalized_covector
    return efficiency * normalized_covector + math.sqrt(1.0 - efficiency**2) * (
        orthogonal / orthogonal.norm()
    )


def pooled_metric(gradients: Tensor) -> tuple[Tensor, float]:
    if gradients.ndim != 2 or not torch.all(torch.isfinite(gradients)):
        raise ValueError("gradients are invalid")
    metric = gradients.transpose(0, 1) @ gradients / gradients.shape[0]
    return metric, median_positive_eigenvalue(metric)


def median_positive_eigenvalue(metric: Tensor) -> float:
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric is invalid")
    eigenvalues = torch.linalg.eigvalsh(0.5 * (metric + metric.transpose(0, 1)))
    tolerance = eigenvalues[-1] * metric.shape[0] * torch.finfo(eigenvalues.dtype).eps
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.numel() == 0:
        raise ValueError("metric has no positive spectrum")
    return float(positive.median())


def random_orthonormal_basis(
    dimension: int,
    rank: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> Tensor:
    if not 0 < rank <= dimension:
        raise ValueError("dimension and rank are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
    basis, upper = torch.linalg.qr(values, mode="reduced")
    signs = torch.sign(torch.diag(upper))
    signs[signs == 0] = 1
    return (basis * signs).to(device=device, dtype=dtype)


def select_step_size(calibrations: dict[float, list[dict[str, Any]]]) -> float:
    if not calibrations:
        raise ValueError("calibrations are empty")
    reaching = [
        step_size
        for step_size, trajectory in calibrations.items()
        if max(row["concept_probability"] for row in trajectory) >= 0.9
    ]
    if reaching:
        return min(reaching)
    return min(
        calibrations,
        key=lambda step_size: (-calibrations[step_size][-1]["concept_probability"], step_size),
    )


def localize_targets(
    evaluate: Callable[[Tensor], dict[str, float]],
    trajectory: list[dict[str, Any]],
    states: list[Tensor],
    targets: tuple[float, ...],
    iterations: int,
) -> list[dict[str, Any]]:
    if len(trajectory) != len(states) or iterations < 1:
        raise ValueError("trajectory, states, or iterations are invalid")
    rows: list[dict[str, Any]] = []
    for target in targets:
        crossing = next(
            (
                index
                for index in range(1, len(trajectory))
                if trajectory[index - 1]["concept_probability"] < target
                <= trajectory[index]["concept_probability"]
            ),
            None,
        )
        if crossing is None:
            rows.append({"reached": False, "target": target})
            continue
        lower = states[crossing - 1]
        upper = states[crossing]
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            if evaluate(middle)["concept_probability"] >= target:
                upper = middle
            else:
                lower = middle
        evaluation = evaluate(upper)
        rows.append(
            {
                "crossing_step": crossing,
                "reached": True,
                "target": target,
                **evaluation,
            }
        )
    return rows


def run_trajectory(
    initial_state: Tensor,
    step_size: float,
    direction: Callable[[Tensor], dict[str, Any]],
    evaluate: Callable[[Tensor], dict[str, float]],
    steps: int,
    efficiency: float,
    stop_probability: float | None = None,
) -> tuple[list[dict[str, Any]], list[Tensor]]:
    if stop_probability is not None and not 0 < stop_probability <= 1:
        raise ValueError("stop probability is invalid")
    state = initial_state.detach()
    records: list[dict[str, Any]] = [{"step": 0, **evaluate(state)}]
    states = [state]
    for step in range(1, steps + 1):
        if state.is_cuda:
            torch.cuda.synchronize(state.device)
        started = time.perf_counter()
        result = direction(state)
        applied = fixed_efficiency_direction(
            result["direction"],
            result["covector"],
            efficiency,
        )
        if state.is_cuda:
            torch.cuda.synchronize(state.device)
        elapsed = time.perf_counter() - started
        state = (state + step_size * applied).detach()
        records.append(
            {
                "direction_seconds": elapsed,
                "step": step,
                **result.get("diagnostics", {}),
                **evaluate(state),
            }
        )
        states.append(state)
        if (
            stop_probability is not None
            and records[-1]["concept_probability"] >= stop_probability
        ):
            break
    return records, states
