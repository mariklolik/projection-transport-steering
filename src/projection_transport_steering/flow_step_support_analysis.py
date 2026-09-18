import math
from itertools import combinations

import numpy as np

from projection_transport_steering.flow_step_support import condition_specifications

SEED = 20260907


def _interval(values: np.ndarray, draws: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired(left: np.ndarray, right: np.ndarray, draws: int, seed: int) -> dict[str, object]:
    differences = left - right
    return {
        "concept_differences": differences.tolist(),
        "mean_difference": float(np.mean(differences)),
        "mean_difference_interval": _interval(differences, draws, seed),
        "wins": int(np.count_nonzero(differences > 0.0)),
    }


def shapley_values(values: dict[tuple[int, ...], float]) -> dict[str, float]:
    players = (0, 1, 2)
    expected = {
        subset
        for size in range(4)
        for subset in combinations(players, size)
    }
    if set(values) != expected or not np.all(np.isfinite(list(values.values()))):
        raise ValueError("three-step set function is invalid")
    allocations = {}
    for player in players:
        others = tuple(value for value in players if value != player)
        allocation = 0.0
        for size in range(3):
            weight = math.factorial(size) * math.factorial(2 - size) / 6
            for subset in combinations(others, size):
                included = tuple(sorted((*subset, player)))
                allocation += weight * (values[included] - values[subset])
        allocations[str(player)] = float(allocation)
    return allocations


def _condition_arrays(concept: dict[str, object], name: str) -> dict[str, np.ndarray]:
    condition = concept["conditions"][name]
    arrays = {
        key: np.asarray(condition[key], dtype=np.float64)
        for key in (
            "construction_score_changes",
            "heldout_score_changes",
            "neutral_forward_kl",
            "neutral_score_changes",
        )
    }
    if (
        arrays["construction_score_changes"].shape != (8,)
        or arrays["heldout_score_changes"].shape != (8,)
        or arrays["neutral_forward_kl"].shape != (16,)
        or arrays["neutral_score_changes"].shape != (16,)
        or not all(np.all(np.isfinite(values)) for values in arrays.values())
    ):
        raise ValueError(f"invalid condition {name}")
    return arrays


def _condition_summary(
    concepts: list[dict[str, object]],
    name: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    rows = [_condition_arrays(concept, name) for concept in concepts]
    construction = np.asarray(
        [np.mean(row["construction_score_changes"]) for row in rows]
    )
    heldout_mean = np.asarray([np.mean(row["heldout_score_changes"]) for row in rows])
    heldout_worst = np.asarray([np.min(row["heldout_score_changes"]) for row in rows])
    neutral_likelihood = np.asarray(
        [np.median(row["neutral_score_changes"]) for row in rows]
    )
    neutral_kl = np.asarray([np.median(row["neutral_forward_kl"]) for row in rows])
    return {
        "concept_construction_means": construction.tolist(),
        "concept_heldout_means": heldout_mean.tolist(),
        "concept_heldout_worst": heldout_worst.tolist(),
        "concept_neutral_likelihood_medians": neutral_likelihood.tolist(),
        "concept_neutral_kl_medians": neutral_kl.tolist(),
        "mean_construction_change": float(np.mean(construction)),
        "mean_heldout_change": float(np.mean(heldout_mean)),
        "mean_heldout_worst": float(np.mean(heldout_worst)),
        "mean_heldout_worst_interval": _interval(heldout_worst, draws, seed),
        "mean_neutral_likelihood_median": float(np.mean(neutral_likelihood)),
        "mean_neutral_kl_median": float(np.mean(neutral_kl)),
    }


def _raw_values(concept: dict[str, object], endpoint: str) -> dict[tuple[int, ...], float]:
    values = {(): 0.0}
    names = {
        (0,): "raw_0",
        (1,): "raw_1",
        (2,): "raw_2",
        (0, 1): "raw_01",
        (0, 2): "raw_02",
        (1, 2): "raw_12",
        (0, 1, 2): "full",
    }
    for subset, name in names.items():
        arrays = _condition_arrays(concept, name)
        values[subset] = {
            "construction_mean": float(np.mean(arrays["construction_score_changes"])),
            "heldout_mean": float(np.mean(arrays["heldout_score_changes"])),
            "heldout_worst": float(np.min(arrays["heldout_score_changes"])),
            "neutral_kl_median": float(np.median(arrays["neutral_forward_kl"])),
            "neutral_likelihood_median": float(np.median(arrays["neutral_score_changes"])),
        }[endpoint]
    return values


def _shapley(concepts: list[dict[str, object]]) -> dict[str, object]:
    endpoints = (
        "construction_mean",
        "heldout_mean",
        "heldout_worst",
        "neutral_likelihood_median",
        "neutral_kl_median",
    )
    per_concept = {}
    errors = []
    for concept in concepts:
        allocations = {}
        for endpoint in endpoints:
            values = _raw_values(concept, endpoint)
            result = shapley_values(values)
            errors.append(abs(sum(result.values()) - values[(0, 1, 2)]))
            allocations[endpoint] = result
        per_concept[str(concept["concept_id"])] = allocations
    means = {
        endpoint: {
            player: float(
                np.mean(
                    [rows[endpoint][player] for rows in per_concept.values()]
                )
            )
            for player in ("0", "1", "2")
        }
        for endpoint in endpoints
    }
    return {
        "max_reconstruction_error": max(errors, default=0.0),
        "mean_allocations": means,
        "per_concept": per_concept,
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
    planned = sum(receipt.get("planned_concepts", len(receipt["concepts"])) for receipt in receipts)
    concept_ids = [concept["concept_id"] for concept in concepts]
    specifications = condition_specifications()
    expected_names = [str(specification["name"]) for specification in specifications]
    complete_conditions = all(
        concept["condition_order"] == expected_names
        and set(concept["conditions"]) == set(expected_names)
        for concept in concepts
    )
    invariants_pass = bool(
        all(receipt["status"] == "pass" for receipt in receipts)
        and all(receipt["replay_max_abs_logit_error"] == 0.0 for receipt in receipts)
        and all(receipt["condition_serialization_exact"] for receipt in receipts)
        and len(concept_ids) == len(set(concept_ids))
        and planned == expected_concepts
        and complete_conditions
    )
    summaries = {
        name: _condition_summary(concepts, name, bootstrap_draws, seed + index)
        for index, name in enumerate(expected_names)
    }
    candidate = summaries["equal_12"]
    full = summaries["full"]
    early = summaries["equal_01"]
    candidate_worst = np.asarray(candidate["concept_heldout_worst"])
    full_worst = np.asarray(full["concept_heldout_worst"])
    early_worst = np.asarray(early["concept_heldout_worst"])
    candidate_kl = np.asarray(candidate["concept_neutral_kl_medians"])
    full_kl = np.asarray(full["concept_neutral_kl_medians"])
    candidate_likelihood = np.asarray(candidate["concept_neutral_likelihood_medians"])
    full_likelihood = np.asarray(full["concept_neutral_likelihood_medians"])
    valid_denominators = bool(np.all(full_kl > 0.0))
    fractional_kl = (full_kl - candidate_kl) / full_kl if valid_denominators else np.full_like(full_kl, -np.inf)
    versus_full = _paired(candidate_worst, full_worst, bootstrap_draws, seed + 100)
    versus_early = _paired(candidate_worst, early_worst, bootstrap_draws, seed + 101)
    likelihood = _paired(candidate_likelihood, full_likelihood, bootstrap_draws, seed + 102)
    neutral_kl = {
        "concept_fractional_reductions": fractional_kl.tolist(),
        "mean_fractional_reduction": float(np.mean(fractional_kl)),
        "valid_denominators": valid_denominators,
        "wins": int(np.count_nonzero(candidate_kl < full_kl)),
    }
    gate = {
        "candidate_vs_early": versus_early,
        "candidate_vs_full": versus_full,
        "complete_concepts": len(concepts),
        "neutral_kl": neutral_kl,
        "neutral_likelihood": likelihood,
    }
    gate["pass"] = bool(
        invariants_pass
        and len(concepts) >= 11
        and versus_full["wins"] >= 9
        and versus_full["mean_difference"] > 0.0
        and neutral_kl["wins"] >= 9
        and neutral_kl["mean_fractional_reduction"] >= 0.25
        and likelihood["wins"] >= 9
        and likelihood["mean_difference"] > 0.0
        and versus_early["wins"] >= 9
        and versus_early["mean_difference"] > 0.0
    )
    shapley = _shapley(concepts)
    return {
        "bootstrap_draws": bootstrap_draws,
        "complete_concepts": len(concepts),
        "conditions": summaries,
        "decision": (
            "advance_flow_step_support_v1"
            if gate["pass"]
            else "close_flow_step_support_v1"
        ),
        "expected_concepts": expected_concepts,
        "gate": gate,
        "invariants_pass": invariants_pass,
        "planned_concepts": planned,
        "seed": seed,
        "shapley": shapley,
    }
