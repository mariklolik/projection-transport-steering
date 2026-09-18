import numpy as np
from scipy.stats import spearmanr

from projection_transport_steering.flow_step_support import condition_specifications

SEED = 20260907


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    result = float(spearmanr(left, right).statistic)
    return (result, True) if np.isfinite(result) else (-1.0, False)


def _interval(values: np.ndarray, draws: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(
    correlations: list[float],
    valid: list[bool],
    median_threshold: float,
    positive_threshold: int,
    lower_threshold: float,
    draws: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(correlations, dtype=np.float64)
    interval = _interval(values, draws, seed)
    summary = {
        "bootstrap_mean_interval": interval,
        "concept_correlations": correlations,
        "invalid_concepts": int(len(valid) - sum(valid)),
        "mean_correlation": float(np.mean(values)),
        "median_correlation": float(np.median(values)),
        "positive_concepts": int(np.count_nonzero(values > 0.0)),
    }
    summary["pass"] = bool(
        all(valid)
        and summary["median_correlation"] >= median_threshold
        and summary["positive_concepts"] >= positive_threshold
        and interval[0] > lower_threshold
    )
    return summary


def _generated_arrays(
    rows: list[dict[str, object]],
    concept_ids: list[int],
    condition_names: list[str],
) -> tuple[dict[tuple[int, str], dict[str, float]], bool]:
    expected = {
        (concept_id, condition, prompt_index)
        for concept_id in concept_ids
        for condition in condition_names
        for prompt_index in range(8)
    }
    indexed: dict[tuple[int, str, int], dict[str, object]] = {}
    valid = True
    for row in rows:
        key = (int(row["concept_id"]), str(row["condition"]), int(row["prompt_index"]))
        scores = row.get("scores")
        valid = bool(
            valid
            and key in expected
            and key not in indexed
            and isinstance(scores, dict)
            and all(
                name in scores and np.isfinite(float(scores[name]))
                for name in ("concept", "fluency", "harmonic_mean", "instruction")
            )
        )
        indexed[key] = row
    valid = bool(valid and set(indexed) == expected)
    aggregated = {}
    if not valid:
        return aggregated, False
    for concept_id in concept_ids:
        for condition in condition_names:
            scores = [indexed[(concept_id, condition, index)]["scores"] for index in range(8)]
            aggregated[(concept_id, condition)] = {
                name: float(np.mean([float(score[name]) for score in scores]))
                for name in ("concept", "fluency", "harmonic_mean", "instruction")
            }
    return aggregated, True


def _teacher_array(
    teacher: dict[str, object],
    condition_names: list[str],
    field: str,
    concept_index: int,
) -> np.ndarray:
    values = np.asarray(
        [teacher["conditions"][name][field][concept_index] for name in condition_names],
        dtype=np.float64,
    )
    if values.shape != (len(condition_names),) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid teacher array {field}")
    return values


def _sign_agrees(left: float, right: float) -> bool:
    return bool(
        (left > 0.0 and right > 0.0)
        or (left < 0.0 and right < 0.0)
        or (left == 0.0 and right == 0.0)
    )


def analyze_proxy_validity(
    teacher: dict[str, object],
    rows: list[dict[str, object]],
    concept_ids: list[int],
    invariants_pass: bool,
    bootstrap_draws: int = 10_000,
    seed: int = SEED,
) -> dict[str, object]:
    condition_names = [str(specification["name"]) for specification in condition_specifications()]
    generated, rows_valid = _generated_arrays(rows, concept_ids, condition_names)
    teacher_valid = True
    correlations = {"pv2": [], "pv3": [], "pv5": []}
    valid = {"pv2": [], "pv3": [], "pv5": []}
    agreements = []
    per_concept_agreement = {}
    try:
        for concept_index, concept_id in enumerate(concept_ids):
            teacher_mean = _teacher_array(
                teacher, condition_names, "concept_heldout_means", concept_index
            )
            teacher_worst = _teacher_array(
                teacher, condition_names, "concept_heldout_worst", concept_index
            )
            teacher_kl = _teacher_array(
                teacher, condition_names, "concept_neutral_kl_medians", concept_index
            )
            generated_concept = np.asarray(
                [generated[(concept_id, name)]["concept"] for name in condition_names]
            )
            generated_hmean = np.asarray(
                [generated[(concept_id, name)]["harmonic_mean"] for name in condition_names]
            )
            generated_fluency_loss = -np.asarray(
                [generated[(concept_id, name)]["fluency"] for name in condition_names]
            )
            for key, left, right in (
                ("pv2", teacher_mean, generated_concept),
                ("pv3", teacher_worst, generated_hmean),
                ("pv5", teacher_kl, generated_fluency_loss),
            ):
                correlation, is_valid = _correlation(left, right)
                correlations[key].append(correlation)
                valid[key].append(is_valid)
            full_index = condition_names.index("full")
            concept_agreements = [
                _sign_agrees(
                    teacher_worst[index] - teacher_worst[full_index],
                    generated_hmean[index] - generated_hmean[full_index],
                )
                for index, name in enumerate(condition_names)
                if name != "full"
            ]
            agreements.extend(concept_agreements)
            per_concept_agreement[str(concept_id)] = float(np.mean(concept_agreements))
    except (KeyError, IndexError, TypeError, ValueError):
        teacher_valid = False
    complete = bool(
        invariants_pass
        and rows_valid
        and teacher_valid
        and len(concept_ids) == 12
        and len(set(concept_ids)) == 12
    )
    if teacher_valid:
        pv2 = _summary(correlations["pv2"], valid["pv2"], 0.50, 9, 0.20, bootstrap_draws, seed)
        pv3 = _summary(correlations["pv3"], valid["pv3"], 0.50, 9, 0.20, bootstrap_draws, seed + 1)
        pv5 = _summary(correlations["pv5"], valid["pv5"], 0.30, 8, 0.0, bootstrap_draws, seed + 2)
    else:
        pv2 = pv3 = pv5 = {"pass": False}
    pv4 = {
        "overall_agreement": float(np.mean(agreements)) if agreements else 0.0,
        "per_concept_agreement": per_concept_agreement,
    }
    pv4["pass"] = bool(
        agreements
        and pv4["overall_agreement"] >= 0.70
        and min(per_concept_agreement.values(), default=0.0) >= 0.60
    )
    gate = {"pv1": {"pass": complete}, "pv2": pv2, "pv3": pv3, "pv4": pv4, "pv5": pv5}
    gate["pass"] = bool(all(gate[key]["pass"] for key in ("pv1", "pv2", "pv3", "pv4", "pv5")))
    return {
        "bootstrap_draws": bootstrap_draws,
        "condition_order": condition_names,
        "decision": "retain_teacher_forced_screen"
        if gate["pass"]
        else "retire_teacher_forced_screen",
        "gate": gate,
        "invariants_pass": complete,
        "seed": seed,
    }
