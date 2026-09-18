import numpy as np

from projection_transport_steering.statistics import paired_bca_interval

FACTORS = (0.25, 0.5, 1.0, 1.5)
METHODS = (
    "pooled_euclidean",
    "pooled_sequence_metric",
    "robust_euclidean",
    "semantic_outcome",
    "last_token_semantic_outcome",
    "diffmean",
)
PRIMARY_BASELINES = ("pooled_sequence_metric", "robust_euclidean")


def _condition(concept: dict[str, object], method: str, factor: float) -> dict[str, object]:
    key = f"{method}@{factor:g}"
    condition = concept["conditions"].get(key)
    if condition is None:
        raise ValueError(f"missing condition {key}")
    heldout = np.asarray(condition["heldout_score_changes"], dtype=np.float64)
    neutral = np.asarray(condition["neutral_score_changes"], dtype=np.float64)
    divergence = np.asarray(condition["neutral_forward_kl"], dtype=np.float64)
    if (
        heldout.shape != (8,)
        or neutral.shape != (16,)
        or divergence.shape != (16,)
        or not np.all(np.isfinite([*heldout, *neutral, *divergence]))
        or np.any(divergence < -1e-7)
    ):
        raise ValueError(f"invalid condition {key}")
    return condition


def _solver_pass(solver: dict[str, object]) -> bool:
    return bool(
        solver["feasible"]
        and solver["status"] == "hard"
        and solver["stationarity_residual"] < 1e-5
        and solver["complementarity_residual"] < 1e-5
        and np.all(np.isfinite(solver["margins"]))
        and min(solver["margins"]) >= 1.0 - 1e-7
    )


def _concept_invariants(concept: dict[str, object]) -> bool:
    if concept.get("status") != "pass":
        return False
    if set(concept["action_metric_costs"]) != set(METHODS):
        return False
    costs = np.asarray(list(concept["action_metric_costs"].values()), dtype=np.float64)
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0.0):
        return False
    if not np.allclose(costs, costs[0], rtol=1e-8, atol=1e-15):
        return False
    geometry = concept["geometry"]
    return all(
        _solver_pass(geometry[key])
        for key in ("mean_solver", "robust_euclidean_solver", "last_token_solver")
    )


def _method_summary(concepts, method, factor):
    heldout = [
        np.asarray(_condition(concept, method, factor)["heldout_score_changes"])
        for concept in concepts
    ]
    neutral = [
        np.asarray(_condition(concept, method, factor)["neutral_score_changes"])
        for concept in concepts
    ]
    divergence = [
        np.asarray(_condition(concept, method, factor)["neutral_forward_kl"])
        for concept in concepts
    ]
    return {
        "concept_mean_changes": [float(np.mean(values)) for values in heldout],
        "concept_neutral_medians": [float(np.median(values)) for values in neutral],
        "concept_neutral_kl_medians": [float(np.median(values)) for values in divergence],
        "concept_worst_changes": [float(np.min(values)) for values in heldout],
        "mean_change": float(np.mean(heldout)),
        "mean_worst_change": float(np.mean([np.min(values) for values in heldout])),
        "median_neutral_change": float(np.median(neutral)),
        "median_neutral_kl": float(np.median(divergence)),
    }


def analyze_concepts(
    concepts: list[dict[str, object]],
    resamples: int = 10_000,
) -> dict[str, object]:
    complete = [concept for concept in concepts if _concept_invariants(concept)]
    concept_ids = [int(concept["concept_id"]) for concept in complete]
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("concept IDs are duplicated")
    factors = {}
    passing_factors = []
    for factor in FACTORS:
        methods = {
            method: _method_summary(complete, method, factor) for method in METHODS
        }
        contrasts = {}
        for baseline in PRIMARY_BASELINES:
            candidate = np.asarray(methods["semantic_outcome"]["concept_worst_changes"])
            comparator = np.asarray(methods[baseline]["concept_worst_changes"])
            differences = candidate - comparator
            neutral = np.asarray(
                methods["semantic_outcome"]["concept_neutral_medians"]
            ) - np.asarray(methods[baseline]["concept_neutral_medians"])
            interval = paired_bca_interval(
                differences,
                resamples=resamples,
                seed=20260907,
            )
            contrasts[baseline] = {
                "concept_differences": differences.tolist(),
                "mean_difference": float(np.mean(differences)),
                "mean_difference_bca95": list(interval),
                "neutral_median_difference": float(np.median(neutral)),
                "wins": int(np.count_nonzero(differences > 0.0)),
            }
        factor_pass = all(
            contrasts[baseline]["wins"] >= 9
            and contrasts[baseline]["mean_difference"] > 0.0
            and contrasts[baseline]["neutral_median_difference"] >= -0.02
            for baseline in PRIMARY_BASELINES
        )
        key = f"{factor:g}"
        factors[key] = {
            "contrasts": contrasts,
            "methods": methods,
            "premise_pass": factor_pass,
        }
        if factor_pass:
            passing_factors.append(factor)
    algebra_pass = len(complete) >= 11
    return {
        "algebra_and_coverage_pass": algebra_pass,
        "complete_concept_ids": concept_ids,
        "complete_concepts": len(complete),
        "factors": factors,
        "passing_factors": passing_factors,
        "planned_concepts": len(concepts),
        "stage_b_authorized": algebra_pass and bool(passing_factors),
    }
