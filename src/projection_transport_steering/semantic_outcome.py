import numpy as np
from numpy.typing import ArrayLike, NDArray

from projection_transport_steering.claim_relative_geometry import (
    diagonal_metric_robust_action,
    reduced_identity_robust_action,
)

FloatArray = NDArray[np.float64]


def sequence_score_metric(
    score_gradients: ArrayLike,
    ridge_fraction: float = 0.01,
    floor: float = 1e-12,
) -> FloatArray:
    gradients = np.asarray(score_gradients, dtype=np.float64)
    if (
        gradients.ndim != 2
        or gradients.shape[0] == 0
        or gradients.shape[1] == 0
        or not np.all(np.isfinite(gradients))
        or not np.isfinite(ridge_fraction)
        or ridge_fraction <= 0.0
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError("score gradients or ridge parameters are invalid")
    second_moment = np.mean(gradients**2, axis=0)
    ridge = max(floor, ridge_fraction * float(np.mean(second_moment)))
    return second_moment + ridge


def unit_covectors(covectors: ArrayLike) -> FloatArray:
    views = np.asarray(covectors, dtype=np.float64)
    if (
        views.ndim != 2
        or views.shape[0] == 0
        or views.shape[1] == 0
        or not np.all(np.isfinite(views))
    ):
        raise ValueError("covectors are invalid")
    norms = np.linalg.norm(views, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("covectors are invalid")
    return views / norms


def _minimum_diagonal_action(diagonal: FloatArray, covector: FloatArray) -> FloatArray:
    inverse = covector / diagonal
    reachability = float(covector @ inverse)
    if reachability <= 0.0:
        raise ValueError("pooled covector is unreachable")
    return inverse / reachability


def fit_semantic_outcome_actions(
    score_covectors: ArrayLike,
    neutral_score_gradients: ArrayLike,
) -> dict[str, object]:
    return fit_semantic_outcome_actions_from_metric(
        score_covectors,
        sequence_score_metric(neutral_score_gradients),
    )


def fit_semantic_outcome_actions_from_metric(
    score_covectors: ArrayLike,
    metric_diagonal: ArrayLike,
) -> dict[str, object]:
    views = unit_covectors(score_covectors)
    metric = np.asarray(metric_diagonal, dtype=np.float64)
    if metric.shape != (views.shape[0],):
        raise ValueError("metric and covectors do not match")
    if not np.all(np.isfinite(metric)) or np.any(metric <= 0.0):
        raise ValueError("metric and covectors do not match")
    pooled = np.mean(views, axis=1)
    pooled_norm = float(pooled @ pooled)
    pooled_euclidean = pooled / pooled_norm if pooled_norm > 0.0 else np.full_like(pooled, np.nan)
    pooled_sequence_metric = (
        _minimum_diagonal_action(metric, pooled)
        if pooled_norm > 0.0
        else np.full_like(pooled, np.nan)
    )
    return {
        "metric_diagonal": metric,
        "pooled_euclidean": pooled_euclidean,
        "pooled_sequence_metric": pooled_sequence_metric,
        "robust_euclidean": reduced_identity_robust_action(views, target=1.0),
        "semantic_outcome": diagonal_metric_robust_action(metric, views, target=1.0),
        "unit_covectors": views,
    }
