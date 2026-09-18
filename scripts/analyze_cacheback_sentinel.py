import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_summary(values: list[float], bootstrap_samples: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    random = np.random.default_rng(20260906)
    means = np.mean(
        random.choice(array, size=(bootstrap_samples, len(array)), replace=True),
        axis=1,
    )
    return {
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "mean_ci95_high": float(np.quantile(means, 0.975)),
        "mean_ci95_low": float(np.quantile(means, 0.025)),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
    }


def spearman_summary(
    predicted: list[float],
    realized: list[float],
    bootstrap_samples: int,
) -> dict[str, float]:
    predicted_array = np.asarray(predicted)
    realized_array = np.asarray(realized)
    random = np.random.default_rng(20260906)
    estimates = []
    for _ in range(bootstrap_samples):
        indices = random.integers(0, len(predicted_array), len(predicted_array))
        if len(np.unique(predicted_array[indices])) < 2 or len(
            np.unique(realized_array[indices])
        ) < 2:
            continue
        estimate = spearmanr(predicted_array[indices], realized_array[indices]).statistic
        if np.isfinite(estimate):
            estimates.append(estimate)
    return {
        "estimate": float(spearmanr(predicted_array, realized_array).statistic),
        "bootstrap_ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_ci95_low": float(np.quantile(estimates, 0.025)),
    }


def shard_directories(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.glob("shard_*")) if path.is_dir()]


def summarize_method(
    rows: list[dict[str, Any]],
    method: str,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    reductions = []
    retentions = []
    concentrations = []
    predicted = []
    realized = []
    taylor_errors = []
    concepts: dict[str, list[float]] = {}
    for row in rows:
        baseline = row["methods"]["cacheback_h0"]
        result = row["methods"][method]
        cumulative = result["cumulative_forward_kl_mean"]
        reduction = 1 - cumulative / baseline["cumulative_forward_kl_mean"]
        reductions.append(reduction)
        retentions.append(
            result["immediate_semantic_effect_mean"]
            / baseline["immediate_semantic_effect_mean"]
        )
        concentrations.append(max(result["offset_forward_kl_mean"]) / cumulative)
        predicted.append(result["predicted_cumulative_quadratic_cost"])
        realized.append(cumulative)
        taylor_errors.append(abs(predicted[-1] - cumulative) / cumulative)
        concepts.setdefault(row["concept"], []).append(reduction)
    return {
        "concept_mean_reduction": {
            concept: float(np.mean(values)) for concept, values in sorted(concepts.items())
        },
        "cumulative_kl_reduction": scalar_summary(reductions, bootstrap_samples),
        "immediate_effect_retention": scalar_summary(retentions, bootstrap_samples),
        "offset_concentration": scalar_summary(concentrations, bootstrap_samples),
        "predicted_realized_spearman": spearman_summary(
            predicted,
            realized,
            bootstrap_samples,
        ),
        "predicted_to_full_trajectory_relative_error": scalar_summary(
            taylor_errors,
            bootstrap_samples,
        ),
        "prediction_scope": "full-h8-unregularized-metric",
    }


def load_packet(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    receipts = []
    for shard in shard_directories(directory):
        receipt = json.loads((shard / "receipt.json").read_text())
        result_path = shard / "results.jsonl"
        if receipt["results_sha256"] != file_sha256(result_path):
            raise ValueError(f"result hash mismatch in {shard.name}")
        if receipt["pilot_states_observed"] != 0 or receipt["failed_cases"] != 0:
            raise ValueError(f"invalid sentinel receipt in {shard.name}")
        shard_rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        if len(shard_rows) != receipt["case_count"] or any(
            row["status"] != "pass" for row in shard_rows
        ):
            raise ValueError(f"incomplete sentinel results in {shard.name}")
        rows.extend(shard_rows)
        receipts.append(receipt)
    if len(receipts) != 4 or len(rows) != 8:
        raise ValueError("sentinel packet is incomplete")
    return rows, {
        "case_count": len(rows),
        "pilot_states_observed": 0,
        "receipt_sha256": [file_sha256(path) for path in sorted(directory.glob("shard_*/receipt.json"))],
        "runner_sha256": sorted({receipt["runner_sha256"] for receipt in receipts}),
    }


def analyze(directory: Path, bootstrap_samples: int) -> dict[str, Any]:
    rows, provenance = load_packet(directory)
    methods = {
        method: summarize_method(rows, method, bootstrap_samples)
        for method in ("cacheback_h2", "cacheback_h4", "cacheback_h8", "euclidean")
    }
    h8 = methods["cacheback_h8"]
    return {
        "diagnostic_gates": {
            "h8_all_case_reduction_above_15pct": h8["cumulative_kl_reduction"]["minimum"] > 0.15,
            "h8_all_case_retention_at_least_98pct": h8["immediate_effect_retention"]["minimum"]
            >= 0.98,
            "h8_no_offset_at_least_80pct": h8["offset_concentration"]["maximum"] < 0.8,
            "h8_taylor_error_below_20pct": h8[
                "predicted_to_full_trajectory_relative_error"
            ]["maximum"]
            < 0.2,
            "nested_metrics_psd": min(row["minimum_increment_eigenvalue"] for row in rows)
            >= -1e-8,
        },
        "methods": methods,
        "provenance": provenance,
        "scope": "exposed-sentinel-premise-only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.input, args.bootstrap_samples)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
