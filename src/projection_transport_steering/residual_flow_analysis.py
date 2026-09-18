import numpy as np

QUANTILES = (0.5, 0.75, 1.0)
METHODS = (
    "residual_metric",
    "pooled_residual_metric",
    "euclidean_residual",
    "shuffled_deficits",
    "retained_pooled_sequence_metric",
)
SEED = 20260907


def _interval(values: np.ndarray, draws: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _condition(concept: dict[str, object], quantile: float, method: str) -> dict[str, object]:
    return concept["quantiles"][f"{quantile:g}"]["conditions"][method]


def _method_summary(
    concepts: list[dict[str, object]],
    quantile: float,
    method: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    conditions = [_condition(concept, quantile, method) for concept in concepts]
    heldout = [
        np.asarray(condition["heldout_score_changes"], dtype=np.float64)
        for condition in conditions
    ]
    neutral = [
        np.asarray(condition["neutral_score_changes"], dtype=np.float64)
        for condition in conditions
    ]
    divergence = [
        np.asarray(condition["neutral_forward_kl"], dtype=np.float64)
        for condition in conditions
    ]
    if (
        any(values.shape != (8,) for values in heldout)
        or any(values.shape != (16,) for values in neutral)
        or any(values.shape != (16,) for values in divergence)
        or not all(np.all(np.isfinite(values)) for values in [*heldout, *neutral, *divergence])
    ):
        raise ValueError("residual-flow condition is invalid")
    worst = np.asarray([np.min(values) for values in heldout])
    mean = np.asarray([np.mean(values) for values in heldout])
    return {
        "concept_mean_changes": mean.tolist(),
        "concept_worst_changes": worst.tolist(),
        "mean_change": float(np.mean(mean)),
        "mean_change_interval": _interval(mean, draws, seed),
        "mean_worst_change": float(np.mean(worst)),
        "mean_worst_change_interval": _interval(worst, draws, seed + 1),
        "median_neutral_change": float(np.median(np.concatenate(neutral))),
        "median_neutral_forward_kl": float(np.median(np.concatenate(divergence))),
        "wins_over_flas": int(np.count_nonzero(worst > 0.0)),
    }


def _paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, object]:
    differences = left - right
    return {
        "concept_differences": differences.tolist(),
        "mean_difference": float(np.mean(differences)),
        "mean_difference_interval": _interval(differences, draws, seed),
        "wins": int(np.count_nonzero(differences > 0.0)),
    }


def _split_summary(
    values: np.ndarray,
    effects: np.ndarray,
) -> dict[str, object]:
    order = np.argsort(values, kind="stable")
    midpoint = len(order) // 2
    lower = order[:midpoint]
    upper = order[midpoint:]
    return {
        "lower_count": int(len(lower)),
        "lower_mean_candidate_worst_change": float(np.mean(effects[lower])),
        "upper_count": int(len(upper)),
        "upper_mean_candidate_worst_change": float(np.mean(effects[upper])),
    }


def _diagnostics(
    concepts: list[dict[str, object]],
    quantile: float,
    candidate_worst: np.ndarray,
) -> dict[str, object]:
    changes = [np.asarray(concept["construction"]["changes"], dtype=np.float64) for concept in concepts]
    deficits = [
        np.asarray(concept["quantiles"][f"{quantile:g}"]["deficits"], dtype=np.float64)
        for concept in concepts
    ]
    solvers = [concept["quantiles"][f"{quantile:g}"]["candidate_solver"] for concept in concepts]
    conditions = [_condition(concept, quantile, "residual_metric") for concept in concepts]
    margins = [np.asarray(condition["realized_margins"], dtype=np.float64) for condition in conditions]
    if any(value.shape != (8,) for value in [*changes, *deficits, *margins]):
        raise ValueError("residual-flow geometry is invalid")
    means = np.asarray([np.mean(value) for value in changes])
    ranges = np.asarray([np.ptp(value) for value in changes])
    return {
        "action_cosines_to_candidate": {
            method: float(
                np.mean(
                    [
                        _condition(concept, quantile, method)["action_cosine_to_candidate"]
                        for concept in concepts
                    ]
                )
            )
            for method in METHODS
        },
        "construction_change_mean": float(np.mean(np.concatenate(changes))),
        "construction_change_range_mean": float(np.mean(ranges)),
        "construction_negative_fraction": float(np.mean(np.concatenate(changes) < 0.0)),
        "constraint_coverage": float(
            np.mean(
                np.concatenate(
                    [margin >= deficit - 1e-8 for margin, deficit in zip(margins, deficits, strict=True)]
                )
            )
        ),
        "deficit_mean": float(np.mean(np.concatenate(deficits))),
        "mean_active_constraints": float(
            np.mean([solver["active_constraints"] for solver in solvers])
        ),
        "mean_dual_concentration": float(
            np.mean([solver["dual_concentration"] for solver in solvers])
        ),
        "strength_halves": _split_summary(means, candidate_worst),
        "dispersion_halves": _split_summary(ranges, candidate_worst),
    }


def analyze_design(
    receipts: list[dict[str, object]],
    expected_concepts: int = 12,
    bootstrap_draws: int = 10_000,
    seed: int = SEED,
) -> dict[str, object]:
    concepts = [
        concept
        for receipt in receipts
        for concept in receipt["concepts"]
        if concept["status"] == "pass"
    ]
    concept_ids = [concept["concept_id"] for concept in concepts]
    planned = sum(receipt.get("planned_concepts", len(receipt["concepts"])) for receipt in receipts)
    replay_pass = all(receipt["replay_max_abs_logit_error"] == 0.0 for receipt in receipts)
    receipt_pass = all(receipt["status"] == "pass" for receipt in receipts)
    unique_pass = len(concept_ids) == len(set(concept_ids))
    solver_pass = all(
        concept["quantiles"][f"{quantile:g}"]["candidate_solver"]["feasible"]
        for concept in concepts
        for quantile in QUANTILES
    )
    invariants_pass = bool(
        receipt_pass
        and replay_pass
        and unique_pass
        and planned == expected_concepts
        and solver_pass
    )
    quantile_results = {}
    for quantile_index, quantile in enumerate(QUANTILES):
        methods = {
            method: _method_summary(
                concepts,
                quantile,
                method,
                bootstrap_draws,
                seed + 100 * quantile_index + method_index,
            )
            for method_index, method in enumerate(METHODS)
        }
        candidate = np.asarray(methods["residual_metric"]["concept_worst_changes"])
        pooled = np.asarray(methods["pooled_residual_metric"]["concept_worst_changes"])
        versus_flas = _paired_summary(
            candidate,
            np.zeros_like(candidate),
            bootstrap_draws,
            seed + 1000 + quantile_index,
        )
        versus_pooled = _paired_summary(
            candidate,
            pooled,
            bootstrap_draws,
            seed + 2000 + quantile_index,
        )
        neutral = methods["residual_metric"]["median_neutral_change"]
        divergence = methods["residual_metric"]["median_neutral_forward_kl"]
        gate = {
            "candidate_vs_flas_mean": versus_flas["mean_difference"],
            "candidate_vs_flas_wins": versus_flas["wins"],
            "candidate_vs_pooled_mean": versus_pooled["mean_difference"],
            "candidate_vs_pooled_wins": versus_pooled["wins"],
            "complete_concepts": len(concepts),
            "median_neutral_forward_kl": divergence,
            "median_neutral_likelihood_change": neutral,
        }
        gate["pass"] = bool(
            invariants_pass
            and len(concepts) >= 11
            and gate["candidate_vs_flas_wins"] >= 9
            and gate["candidate_vs_pooled_wins"] >= 9
            and gate["candidate_vs_flas_mean"] > 0.0
            and gate["candidate_vs_pooled_mean"] > 0.0
            and neutral >= -0.02
            and divergence <= 0.002
        )
        quantile_results[f"{quantile:g}"] = {
            "diagnostics": _diagnostics(concepts, quantile, candidate),
            "gate": gate,
            "methods": methods,
            "versus_flas": versus_flas,
            "versus_pooled_residual": versus_pooled,
        }
    selected = next(
        (
            quantile
            for quantile in QUANTILES
            if quantile_results[f"{quantile:g}"]["gate"]["pass"]
        ),
        None,
    )
    return {
        "bootstrap_draws": bootstrap_draws,
        "complete_concepts": len(concepts),
        "decision": (
            "advance_residual_flow_v1" if selected is not None else "close_residual_flow_v1"
        ),
        "expected_concepts": expected_concepts,
        "invariants_pass": invariants_pass,
        "planned_concepts": planned,
        "quantiles": quantile_results,
        "seed": seed,
        "selected_quantile": selected,
    }
