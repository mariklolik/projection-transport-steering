import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from projection_transport_steering.statistics import (
    paired_bca_interval,
    resampled_bca_interval,
)


LAYERS = (3, 6, 9)
REGULARIZATION_MULTIPLIERS = (0.01, 0.1, 1.0)
DOSES = (0.125, 0.25, 0.5)
METHODS = (
    "cacheback_h0",
    "cacheback_h2",
    "cacheback_h4",
    "cacheback_h8",
    "euclidean",
    "caa",
)
HORIZONS = (2, 4, 8)
SEED = 20260906


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjusted(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def holm_alpha_levels(
    p_values: dict[str, float],
    family_alpha: float = 0.05,
) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    return {
        name: family_alpha / (len(ordered) - index)
        for index, name in enumerate(ordered)
    }


def common_doses(records: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    coverage = {}
    eligible = []
    for dose in DOSES:
        dose_rows = [row for row in records if row["dose"] == dose]
        methods = {}
        for method in METHODS:
            method_rows = [row for row in dose_rows if row["method"] == method]
            by_concept = {
                concept: float(
                    np.mean(
                        [row["reached"] for row in method_rows if row["concept"] == concept]
                    )
                )
                for concept in ("third", "ing", "past")
            }
            methods[method] = {
                "by_concept": by_concept,
                "overall": float(np.mean([row["reached"] for row in method_rows])),
            }
        coverage[str(dose)] = methods
        if all(
            values["overall"] >= 0.8
            and min(values["by_concept"].values()) >= 0.8
            for values in methods.values()
        ):
            eligible.append(dose)
    return eligible, coverage


def load_packet(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    receipts = []
    for index in range(8):
        shard = root / f"shard_{index:02d}"
        receipt_path = shard / "receipt.json"
        result_path = shard / "results.jsonl"
        receipt = json.loads(receipt_path.read_text())
        shard_rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        if (
            receipt["shard_index"] != index
            or receipt["shard_count"] != 8
            or receipt["pilot_states_observed"] != 0
            or receipt["failed_rows"] != 0
            or receipt["passed_rows"] != 9
            or receipt["results_sha256"] != file_sha256(result_path)
            or len(shard_rows) != 9
            or any(row["status"] != "pass" for row in shard_rows)
        ):
            raise ValueError(f"invalid development shard {index}")
        rows.extend(shard_rows)
        receipts.append(receipt)
    identities = {(row["case_id"], row["layer"]) for row in rows}
    if len(rows) != 72 or len(identities) != 72:
        raise ValueError("development packet is incomplete")
    return rows, {
        "automatic_differentiation": sorted(
            {receipt["automatic_differentiation"] for receipt in receipts}
        ),
        "maximum_peak_memory_allocated_bytes": max(
            receipt["maximum_peak_memory_allocated_bytes"] for receipt in receipts
        ),
        "pilot_states_observed": 0,
        "receipt_sha256": [
            file_sha256(root / f"shard_{index:02d}" / "receipt.json")
            for index in range(8)
        ],
        "shard_runner_seconds": [receipt["full_runner_seconds"] for receipt in receipts],
        "total_derivative_rows": sum(receipt["total_derivative_rows"] for receipt in receipts),
        "total_jvp_count": sum(receipt["total_jvp_count"] for receipt in receipts),
        "vjp_count": sum(receipt["vjp_count"] for receipt in receipts),
    }


def flatten_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        for configuration in row["configurations"]:
            for method, values in configuration["methods"].items():
                matching = configuration["effect_matching"]["methods"][method]
                flattened.append(
                    {
                        "action_cosine_vs_h0": configuration["action_cosine_vs_h0"].get(
                            method.removeprefix("cacheback_h")
                        ),
                        "case_id": row["case_id"],
                        "concept": row["concept"],
                        "dose": configuration["target"],
                        "endpoint_effect": matching["endpoint_effect"],
                        "immediate_effect": values["immediate_semantic_effect_mean"],
                        "layer": row["layer"],
                        "method": method,
                        "offset_kl": values["offset_forward_kl_mean"],
                        "predicted_kl": values["predicted_cumulative_quadratic_cost"],
                        "realized_kl": values["cumulative_forward_kl_mean"],
                        "reached": matching["reached"],
                        "regularization_multiplier": configuration[
                            "regularization_multiplier"
                        ],
                    }
                )
    return flattened


def interval(values: list[float], resamples: int, alpha: float = 0.05) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(finite) < 2:
        return {"count": len(finite), "estimate": None, "lower": None, "upper": None}
    estimate, lower, upper = paired_bca_interval(
        finite,
        resamples=resamples,
        seed=SEED,
        alpha=alpha,
    )
    return {"count": len(finite), "estimate": estimate, "lower": lower, "upper": upper}


def median_interval(values: list[float], resamples: int) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(finite) < 2:
        return {"count": len(finite), "estimate": None, "lower": None, "upper": None}
    estimate, lower, upper = resampled_bca_interval(
        len(finite),
        lambda indices: float(np.median(finite[indices])),
        resamples=resamples,
        seed=SEED,
    )
    return {"count": len(finite), "estimate": estimate, "lower": lower, "upper": upper}


def contrast_values(
    records: list[dict[str, Any]],
    horizon: int,
) -> dict[str, list[float]]:
    lookup = {(row["case_id"], row["method"]): row for row in records}
    reductions = []
    retentions = []
    concentrations = []
    angles = []
    concept_reductions: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted({row["case_id"] for row in records}):
        baseline = lookup[(case_id, "cacheback_h0")]
        candidate = lookup[(case_id, f"cacheback_h{horizon}")]
        cosine = candidate["action_cosine_vs_h0"]
        if cosine is not None:
            angles.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        if not baseline["reached"] or not candidate["reached"] or baseline["realized_kl"] < 1e-8:
            continue
        reduction = 1.0 - candidate["realized_kl"] / baseline["realized_kl"]
        reductions.append(reduction)
        retentions.append(candidate["immediate_effect"] / baseline["immediate_effect"])
        concept_reductions[candidate["concept"]].append(reduction)
        changes = np.asarray(baseline["offset_kl"][1:]) - np.asarray(candidate["offset_kl"][1:])
        denominator = float(np.abs(changes).sum())
        if denominator > 0:
            concentrations.append(float(np.abs(changes).max() / denominator))
    return {
        "angles": angles,
        "concentrations": concentrations,
        "concept_means": {
            concept: float(np.mean(values))
            for concept, values in sorted(concept_reductions.items())
        },
        "reductions": reductions,
        "retentions": retentions,
    }


def state_spearman(records: list[dict[str, Any]]) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["reached"]:
            grouped[row["case_id"]].append(row)
    values = []
    for rows in grouped.values():
        predicted = [row["predicted_kl"] for row in rows]
        realized = [row["realized_kl"] for row in rows]
        if len(set(predicted)) < 2 or len(set(realized)) < 2:
            continue
        value = float(spearmanr(predicted, realized).statistic)
        if math.isfinite(value):
            values.append(value)
    return values


def dose_diagnostics(records: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    contrasts = {}
    p_values = {}
    reduction_values = {}
    for horizon in HORIZONS:
        values = contrast_values(records, horizon)
        reductions = np.asarray(values["reductions"])
        p_value = (
            float(wilcoxon(reductions - 0.15, alternative="greater").pvalue)
            if len(reductions) >= 2 and not np.all(reductions == 0.15)
            else 1.0
        )
        p_values[f"h{horizon}"] = p_value
        reduction_values[f"h{horizon}"] = values["reductions"]
        contrasts[f"h{horizon}"] = {
            "action_angle": interval(values["angles"], resamples),
            "concept_mean_reduction": values["concept_means"],
            "immediate_effect_retention": interval(values["retentions"], resamples),
            "offset_concentration": interval(values["concentrations"], resamples),
            "reduction": interval(values["reductions"], resamples, alpha=0.1),
        }
    adjusted = holm_adjusted(p_values)
    alpha_levels = holm_alpha_levels(p_values)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p_value_above_15pct"] = value
        contrasts[name]["holm_one_sided_alpha"] = alpha_levels[name]
        contrasts[name]["reduction_holm_one_sided"] = interval(
            reduction_values[name],
            resamples,
            alpha=2.0 * alpha_levels[name],
        )
    return contrasts


def analyze_cell(
    all_records: list[dict[str, Any]],
    layer: int,
    multiplier: float,
    resamples: int,
) -> dict[str, Any]:
    cell = [
        row
        for row in all_records
        if row["layer"] == layer and row["regularization_multiplier"] == multiplier
    ]
    eligible, coverage = common_doses(cell)
    selected_dose = max(eligible) if eligible else None
    diagnostics = {
        str(dose): dose_diagnostics(
            [row for row in cell if row["dose"] == dose],
            resamples,
        )
        for dose in DOSES
    }
    smallest = [row for row in cell if row["dose"] == min(DOSES) and row["reached"]]
    taylor = [
        abs(row["predicted_kl"] - row["realized_kl"]) / row["realized_kl"]
        for row in smallest
        if row["realized_kl"] > 0
    ]
    return {
        "common_doses": eligible,
        "contrasts": diagnostics[str(selected_dose)] if selected_dose is not None else {},
        "coverage": coverage,
        "dose_diagnostics": diagnostics,
        "layer": layer,
        "quadratic_rank_spearman": interval(state_spearman(cell), resamples),
        "regularization_multiplier": multiplier,
        "selected_dose": selected_dose,
        "smallest_dose_taylor_relative_error": median_interval(taylor, resamples),
        "status": "eligible" if selected_dose is not None else "no_common_dose",
    }


def analyze(root: Path, resamples: int) -> dict[str, Any]:
    rows, provenance = load_packet(root)
    records = flatten_results(rows)
    cells = [
        analyze_cell(records, layer, multiplier, resamples)
        for layer in LAYERS
        for multiplier in REGULARIZATION_MULTIPLIERS
    ]
    eligible = [cell for cell in cells if cell["status"] == "eligible"]
    return {
        "cells": cells,
        "development_states": len({row["case_id"] for row in rows}),
        "pilot_states_observed": 0,
        "provenance": provenance,
        "scope": "development-selection-only",
        "selected_cell": None,
        "selection_status": "no_common_dose" if not eligible else "requires_gate_evaluation",
    }


def main() -> None:
    args = parse_args()
    result = analyze(args.input, args.bootstrap_samples)
    args.output.mkdir(parents=True, exist_ok=False)
    analysis_path = args.output / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    receipt = {
        "analysis_sha256": file_sha256(analysis_path),
        "analyzer_sha256": file_sha256(Path(__file__)),
        "input_receipt_sha256": result["provenance"]["receipt_sha256"],
        "pilot_states_observed": 0,
        "selection_status": result["selection_status"],
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
