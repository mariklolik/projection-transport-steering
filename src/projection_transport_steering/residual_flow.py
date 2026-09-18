from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from projection_transport_steering.claim_relative_geometry import (
    diagonal_metric_robust_action,
    reduced_identity_robust_action,
)
from projection_transport_steering.semantic_outcome_evaluation import match_metric_cost
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

FloatArray = NDArray[np.float64]
SEED = 20260907


def residual_deficits(changes: ArrayLike, quantile: float) -> tuple[float, FloatArray]:
    values = np.asarray(changes, dtype=np.float64)
    if (
        values.ndim != 1
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
        or not np.isfinite(quantile)
        or quantile < 0.0
        or quantile > 1.0
    ):
        raise ValueError("changes or quantile are invalid")
    target = max(0.0, float(np.quantile(values, quantile)))
    return target, np.maximum(target - values, 0.0)


def fit_residual_actions(
    score_covectors: ArrayLike,
    metric_diagonal: ArrayLike,
    construction_changes: ArrayLike,
    quantile: float,
    concept_id: int,
) -> dict[str, object]:
    covectors = np.asarray(score_covectors, dtype=np.float64)
    metric = np.asarray(metric_diagonal, dtype=np.float64)
    changes = np.asarray(construction_changes, dtype=np.float64)
    if (
        covectors.ndim != 2
        or metric.shape != (covectors.shape[0],)
        or changes.shape != (covectors.shape[1],)
        or not np.all(np.isfinite(covectors))
        or not np.all(np.isfinite(metric))
        or np.any(metric <= 0.0)
        or not isinstance(concept_id, int)
    ):
        raise ValueError("residual geometry is invalid")
    target, deficits = residual_deficits(changes, quantile)
    if np.max(deficits) <= 0.0:
        raise ValueError("residual deficits are all zero")
    candidate = diagonal_metric_robust_action(metric, covectors, deficits)
    euclidean = reduced_identity_robust_action(covectors, deficits)
    pooled_covector = covectors @ deficits / np.sum(deficits)
    pooled = diagonal_metric_robust_action(metric, pooled_covector[:, None], 1.0)
    permutation = np.random.default_rng(
        np.random.SeedSequence([SEED, concept_id])
    ).permutation(len(deficits))
    shuffled = diagonal_metric_robust_action(metric, covectors, deficits[permutation])
    solvers = {
        "candidate_solver": candidate,
        "euclidean_solver": euclidean,
        "pooled_solver": pooled,
        "shuffled_solver": shuffled,
    }
    if not all(bool(solver["feasible"]) for solver in solvers.values()):
        raise RuntimeError("a frozen residual construction is infeasible")
    reference = np.asarray(candidate["action"])
    controls = {
        "euclidean_residual": np.asarray(euclidean["action"]),
        "pooled_residual_metric": np.asarray(pooled["action"]),
        "shuffled_deficits": np.asarray(shuffled["action"]),
    }
    actions = {"residual_metric": reference}
    actions.update(
        {
            name: match_metric_cost(reference, action, metric)
            for name, action in controls.items()
        }
    )
    return {
        **solvers,
        "actions": actions,
        "deficits": deficits,
        "permutation": permutation,
        "target": target,
    }


def _expand_batch(value: Tensor, batch_size: int) -> Tensor:
    if value.ndim < 2 or value.shape[0] not in (1, batch_size):
        raise ValueError("FLAS concept state does not match batch")
    if value.shape[0] == batch_size:
        return value
    return value.expand(batch_size, *value.shape[1:]).contiguous()


@contextmanager
def flas_teacher_forcing(
    generator: object,
    batch: dict[str, Tensor],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    vector: Tensor | None,
    flowtime: float,
    n_steps: int,
    intervention: dict[str, object] | None = None,
) -> Iterator[Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or not np.isfinite(flowtime)
        or flowtime < 0.0
        or n_steps <= 0
        or getattr(generator, "_hook_handle", None) is not None
    ):
        raise ValueError("FLAS teacher-forcing state is invalid")
    batch_size = input_ids.shape[0]
    generator._n_steps = n_steps
    generator._concept_hidden = _expand_batch(concept_hidden, batch_size)
    generator._concept_mask = _expand_batch(concept_mask, batch_size)
    generator._flowtimes = torch.full(
        (batch_size,), flowtime, device=input_ids.device, dtype=torch.float32
    )
    generator._padding_mask = attention_mask.float()
    generator._sa_caches = [None] * n_steps
    generator._is_prefill = True
    generator._past_len = 0
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    generator._position_ids = position_ids
    correction = (
        TorchLayerAction(generator.llm, generator.layer, AdditiveAction(vector)).installed()
        if vector is not None
        else nullcontext()
    )
    previous_intervention = getattr(generator, "_iv", None)
    installed = False
    try:
        generator._iv = intervention
        generator._install_hook()
        installed = True
        generator._active = True
        with correction:
            yield position_ids
    finally:
        generator._iv = previous_intervention
        generator._active = False
        generator._sa_caches = None
        generator._concept_hidden = None
        generator._concept_mask = None
        generator._flowtimes = None
        generator._padding_mask = None
        generator._position_ids = None
        if installed:
            generator._remove_hook()


def flas_model_logits(
    generator: object,
    batch: dict[str, Tensor],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    vector: Tensor | None = None,
    flowtime: float = 2.0,
    n_steps: int = 3,
    intervention: dict[str, object] | None = None,
) -> Tensor:
    with flas_teacher_forcing(
        generator,
        batch,
        concept_hidden,
        concept_mask,
        vector,
        flowtime,
        n_steps,
        intervention,
    ) as position_ids:
        return generator.llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=position_ids,
            use_cache=False,
        ).logits
