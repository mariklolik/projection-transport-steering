from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog, minimize

FloatArray = NDArray[np.float64]


def _spd_matrix(values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T)
    ):
        raise ValueError("metric must be symmetric positive definite")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("metric must be symmetric positive definite") from error
    return matrix


def _covector_matrix(values: ArrayLike, dimension: int) -> FloatArray:
    covectors = np.asarray(values, dtype=np.float64)
    if (
        covectors.ndim != 2
        or covectors.shape[0] != dimension
        or covectors.shape[1] == 0
        or not np.all(np.isfinite(covectors))
        or np.any(np.linalg.norm(covectors, axis=0) == 0.0)
    ):
        raise ValueError("covectors must be finite nonzero columns")
    return covectors


def _target_vector(values: ArrayLike, count: int) -> FloatArray:
    target = np.asarray(values, dtype=np.float64)
    if target.ndim == 0:
        target = np.full(count, float(target))
    if (
        target.shape != (count,)
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or np.max(target) <= 0.0
    ):
        raise ValueError("target must be finite, nonnegative, and nonzero")
    return target


def robust_metric_action(
    metric: ArrayLike,
    covectors: ArrayLike,
    target: ArrayLike,
    slack_penalty: float | None = None,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    system = _spd_matrix(metric)
    views = _covector_matrix(covectors, system.shape[0])
    target_vector = _target_vector(target, views.shape[1])
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if slack_penalty is not None and (not np.isfinite(slack_penalty) or slack_penalty <= 0.0):
        raise ValueError("slack penalty must be positive and finite")
    view_count = views.shape[1]
    if slack_penalty is None:
        feasibility = linprog(
            np.zeros(system.shape[0]),
            A_ub=-views.T,
            b_ub=-target_vector,
            bounds=[(None, None)] * system.shape[0],
            method="highs",
        )
        if not feasibility.success:
            return {
                "action": np.full(system.shape[0], np.nan),
                "complementarity_residual": np.inf,
                "dual": np.zeros(view_count),
                "feasible": False,
                "margins": np.full(view_count, np.nan),
                "objective": np.nan,
                "slack": np.full(view_count, np.nan),
                "stationarity_residual": np.inf,
                "status": "infeasible",
            }
    inverse_views = np.linalg.solve(system, views)
    gram = views.T @ inverse_views
    gram = 0.5 * (gram + gram.T)
    gram_scale = float(np.max(np.diag(gram)))
    target_scale = float(np.max(target_vector))
    scaled_gram = gram / gram_scale
    scaled_target = target_vector / target_scale
    bounds = (
        [(0.0, slack_penalty * gram_scale / target_scale)] * view_count
        if slack_penalty is not None
        else [(0.0, None)] * view_count
    )
    result = minimize(
        lambda beta: 0.5 * beta @ scaled_gram @ beta - scaled_target @ beta,
        np.zeros(view_count),
        jac=lambda beta: scaled_gram @ beta - scaled_target,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "ftol": max(tolerance**2, np.finfo(np.float64).eps),
            "gtol": tolerance,
            "maxiter": 20_000,
        },
    )
    scaled_dual = np.asarray(result.x, dtype=np.float64)
    if slack_penalty is None:
        active = scaled_dual > tolerance
        polished = np.zeros(view_count)
        polished[active] = np.linalg.lstsq(
            scaled_gram[np.ix_(active, active)],
            scaled_target[active],
            rcond=None,
        )[0]
        gradient = scaled_gram @ polished - scaled_target
        polished_kkt = (
            np.all(polished >= 0.0)
            and np.all(np.abs(gradient[active]) <= 10.0 * tolerance)
            and np.all(gradient[~active] >= -tolerance)
        )
        if polished_kkt:
            scaled_dual = polished
        elif not result.success:
            raise RuntimeError(f"dual solve failed: {result.message}")
    elif not result.success:
        raise RuntimeError(f"dual solve failed: {result.message}")
    dual = target_scale * scaled_dual / gram_scale
    action = inverse_views @ dual
    margins = views.T @ action
    slack = np.maximum(target_vector - margins, 0.0)
    stationarity = np.linalg.norm(system @ action - views @ dual, ord=np.inf)
    primal_complementarity = np.max(np.abs(dual * (margins + slack - target_vector)))
    if slack_penalty is None:
        complementarity = primal_complementarity
        feasible = bool(np.max(slack) <= 10.0 * tolerance)
        status = "hard" if feasible else "infeasible"
        objective = 0.5 * action @ system @ action
    else:
        slack_complementarity = np.max(np.abs((slack_penalty - dual) * slack))
        complementarity = max(primal_complementarity, slack_complementarity)
        feasible = True
        status = "soft"
        objective = 0.5 * action @ system @ action + slack_penalty * slack.sum()
    return {
        "action": action,
        "complementarity_residual": float(complementarity),
        "dual": dual,
        "feasible": feasible,
        "margins": margins,
        "objective": float(objective),
        "slack": slack,
        "stationarity_residual": float(stationarity),
        "status": status,
    }


def reduced_identity_robust_action(
    covectors: ArrayLike,
    target: ArrayLike,
    slack_penalty: float | None = None,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    views = np.asarray(covectors, dtype=np.float64)
    if (
        views.ndim != 2
        or views.shape[0] == 0
        or views.shape[1] == 0
        or not np.all(np.isfinite(views))
        or np.any(np.linalg.norm(views, axis=0) == 0.0)
    ):
        raise ValueError("covectors must be finite nonzero columns")
    left, singular_values, _ = np.linalg.svd(views, full_matrices=False)
    threshold = singular_values[0] * max(views.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > threshold))
    basis = left[:, :rank]
    reduced_views = basis.T @ views
    result = robust_metric_action(
        np.eye(rank),
        reduced_views,
        target,
        slack_penalty=slack_penalty,
        tolerance=tolerance,
    )
    reduced_action = np.asarray(result["action"])
    return {
        **result,
        "action": basis @ reduced_action,
        "reduced_rank": rank,
    }


def diagonal_metric_robust_action(
    metric_diagonal: ArrayLike,
    covectors: ArrayLike,
    target: ArrayLike,
    slack_penalty: float | None = None,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    diagonal = np.asarray(metric_diagonal, dtype=np.float64)
    if diagonal.ndim != 1 or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("metric diagonal must be finite and positive")
    views = _covector_matrix(covectors, diagonal.shape[0])
    roots = np.sqrt(diagonal)
    result = reduced_identity_robust_action(
        views / roots[:, None],
        target,
        slack_penalty=slack_penalty,
        tolerance=tolerance,
    )
    action = np.asarray(result["action"]) / roots
    if result["feasible"]:
        dual = np.asarray(result["dual"])
        stationarity = np.linalg.norm(diagonal * action - views @ dual, ord=np.inf)
        metric_cost = 0.5 * action @ (diagonal * action)
    else:
        stationarity = np.inf
        metric_cost = np.nan
    return {
        **result,
        "action": action,
        "metric_cost": float(metric_cost),
        "stationarity_residual": float(stationarity),
    }


def nested_sequence_metrics(
    terms: ArrayLike,
    horizons: Iterable[int],
    tolerance: float = 1e-10,
) -> dict[int, FloatArray]:
    values = np.asarray(terms, dtype=np.float64)
    selected = tuple(horizons)
    if (
        values.ndim != 3
        or values.shape[1] != values.shape[2]
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
        or not selected
        or any(
            not isinstance(horizon, int) or not 0 <= horizon < values.shape[0]
            for horizon in selected
        )
        or len(set(selected)) != len(selected)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("sequence terms or horizons are invalid")
    symmetric = 0.5 * (values + values.transpose(0, 2, 1))
    if not np.allclose(values, symmetric, atol=tolerance, rtol=tolerance):
        raise ValueError("sequence terms must be symmetric positive semidefinite")
    for term in symmetric:
        scale = max(1.0, float(np.linalg.norm(term, ord=2)))
        if float(np.linalg.eigvalsh(term)[0]) < -tolerance * scale:
            raise ValueError("sequence terms must be symmetric positive semidefinite")
    cumulative = np.cumsum(symmetric, axis=0)
    return {horizon: cumulative[horizon].copy() for horizon in selected}


def minimum_sequence_metric_action(
    metric: ArrayLike,
    covector: ArrayLike,
    target: float,
    regularization: float,
) -> dict[str, object]:
    geometry = np.asarray(metric, dtype=np.float64)
    direction = np.asarray(covector, dtype=np.float64)
    if (
        geometry.ndim != 2
        or geometry.shape[0] != geometry.shape[1]
        or direction.shape != (geometry.shape[0],)
        or not np.all(np.isfinite(geometry))
        or not np.all(np.isfinite(direction))
        or not np.allclose(geometry, geometry.T)
        or np.linalg.norm(direction) == 0.0
        or not np.isfinite(target)
        or target <= 0.0
        or not np.isfinite(regularization)
        or regularization <= 0.0
    ):
        raise ValueError("metric, covector, target, or regularization is invalid")
    system = _spd_matrix(geometry + regularization * np.eye(geometry.shape[0]))
    inverse = np.linalg.solve(system, direction)
    reachability = float(direction @ inverse)
    if reachability <= 0.0:
        raise ValueError("covector is not reachable under the regularized metric")
    action = target * inverse / reachability
    return {
        "action": action,
        "quadratic_cost": float(0.5 * action @ system @ action),
        "reachability": reachability,
    }
