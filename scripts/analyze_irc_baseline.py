from collections import Counter

import numpy as np

from projection_transport_steering.statistics import paired_bca_interval


def interval(values: list[float], resamples: int) -> dict:
    if len(values) < 2:
        return {
            "count": len(values),
            "estimate": float(np.mean(values)) if values else None,
            "lower": None,
            "upper": None,
            "degenerate": None,
        }
    estimate, lower, upper = paired_bca_interval(values, resamples=resamples, seed=20260907)
    return {
        "count": len(values),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "degenerate": len(set(values)) == 1,
    }


def condition_summary(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["sensitivity_supported"]]
    tokens = [row["generated_tokens"] for row in rows]
    return {
        "questions": len(rows),
        "primary_correct": sum(row["primary"]["correct"] for row in rows),
        "eos_count": sum(row["terminated"] for row in rows),
        "cap_failures": sum(not row["terminated"] for row in rows),
        "malformed_count": sum(row["primary"]["parse_status"] != "pass" for row in rows),
        "sensitivity_supported": len(supported),
        "supported_primary_correct": sum(row["primary"]["correct"] for row in supported),
        "supported_sensitivity_correct": sum(row["sensitivity"]["correct"] for row in supported),
        "scorer_disagreements": sum(
            row["primary"]["correct"] != row["sensitivity"]["correct"] for row in supported
        ),
        "generated_tokens": {
            "total": sum(tokens),
            "mean": float(np.mean(tokens)),
            "median": float(np.median(tokens)),
            "p95": float(np.quantile(tokens, 0.95)),
            "maximum": max(tokens),
        },
    }


def contrast(
    source: list[dict],
    zero: list[dict],
    rows: list[dict],
    resamples: int,
    reference_name: str = "zero",
) -> dict:
    primary = [
        int(r["primary"]["correct"]) - int(z["primary"]["correct"])
        for z, r in zip(zero, rows, strict=True)
    ]
    supported = [i for i, row in enumerate(zero) if row["sensitivity_supported"]]
    transitions = Counter(
        f"{int(z['primary']['correct'])}{int(r['primary']['correct'])}"
        for z, r in zip(zero, rows, strict=True)
    )
    labels = {
        "subject": [row["type"] for row in source],
        "difficulty": [row["level"] for row in source],
        "prompt_tokens": [
            "<128"
            if r["prompt_tokens"] < 128
            else "128-255"
            if r["prompt_tokens"] < 256
            else ">=256"
            for r in zero
        ],
        "reference_correctness": [str(r["primary"]["correct"]) for r in zero],
        "reference_termination": ["EOS" if r["terminated"] else "cap" for r in zero],
        "reference_output_tokens": [
            "<=512"
            if r["generated_tokens"] <= 512
            else "513-2048"
            if r["generated_tokens"] <= 2048
            else ">2048"
            for r in zero
        ],
        "candidate_output_tokens_post_treatment": [
            "<=512"
            if r["generated_tokens"] <= 512
            else "513-2048"
            if r["generated_tokens"] <= 2048
            else ">2048"
            for r in rows
        ],
    }
    wrong = transitions["00"] + transitions["01"]
    correct = transitions["10"] + transitions["11"]
    sensitivity = [
        int(rows[i]["sensitivity"]["correct"]) - int(zero[i]["sensitivity"]["correct"])
        for i in supported
    ]
    return {
        "reference_condition": reference_name,
        "rescue_denominator": wrong,
        "harm_retention_denominator": correct,
        "rescue_rate": transitions["01"] / wrong if wrong else None,
        "harm_rate": transitions["10"] / correct if correct else None,
        "retention_rate": transitions["11"] / correct if correct else None,
        "primary": interval(primary, resamples),
        "supported_primary": interval([primary[i] for i in supported], resamples),
        "supported_sensitivity": interval(sensitivity, resamples),
        "measurement_direction_reversal": sum(primary[i] for i in supported) * sum(sensitivity) < 0,
        "generated_tokens": interval(
            [
                r["generated_tokens"] - z["generated_tokens"]
                for z, r in zip(zero, rows, strict=True)
            ],
            resamples,
        ),
        "transitions": {key: transitions[key] for key in ("00", "01", "10", "11")},
        "strata": {
            field: {
                label: interval(
                    [d for d, v in zip(primary, values, strict=True) if v == label], resamples
                )
                for label in sorted(set(values))
            }
            for field, values in labels.items()
        },
    }


def summarize(
    source: list[dict],
    zero: list[dict],
    outcomes: list[list[dict]],
    candidates: list[dict],
    resamples: int = 10_000,
) -> dict:
    expected = [row["cluster_id"] for row in source]
    product = {
        (rank, site, application)
        for rank in (4, 8)
        for site in (17, 23)
        for application in ("prompt", "full")
    }
    if (
        len(candidates) != 8
        or len(outcomes) != 8
        or not expected
        or len(set(expected)) != len(expected)
        or resamples < 2
        or {(c["rank"], c["site"], c["application"]) for c in candidates} != product
    ):
        raise ValueError("incomplete frozen candidate or question product")
    for rows in [zero, *outcomes]:
        if [row["cluster_id"] for row in rows] != expected:
            raise ValueError("incomplete or misordered paired identities")
        for z, row in zip(zero, rows, strict=True):
            if (
                row["evaluator_errors"]
                or row["sensitivity_supported"] != z["sensitivity_supported"]
                or not isinstance(row["primary"], dict)
                or row["sensitivity_supported"] != isinstance(row["sensitivity"], dict)
            ):
                raise ValueError("invalid scored row or fixed support membership")
    summaries = [condition_summary(rows) for rows in [zero, *outcomes]]
    selected = {
        application: min(
            (i for i, c in enumerate(candidates) if c["application"] == application),
            key=lambda i: (
                -summaries[i + 1]["primary_correct"],
                candidates[i]["rank"],
                candidates[i]["site"],
            ),
        )
        for application in ("prompt", "full")
    }
    return {
        "evidence_label": "exploratory_selection",
        "uncertainty": {
            "unit": "paired_question",
            "resamples": resamples,
            "seed": 20260907,
            "confidence": 0.95,
            "simultaneous": False,
            "selection_corrected": False,
            "training_seeds": 1,
            "rollouts_per_question_condition": 1,
        },
        "conditions": [{"condition": "zero", **summaries[0]}]
        + [{"index": i, **c, **summaries[i + 1]} for i, c in enumerate(candidates)],
        "contrasts_vs_zero": [contrast(source, zero, rows, resamples) for rows in outcomes],
        "selected_full_vs_selected_prompt": contrast(
            source,
            outcomes[selected["prompt"]],
            outcomes[selected["full"]],
            resamples,
            reference_name="selected_prompt",
        ),
        "selected_indices": selected,
        "baseline_entry_criterion_met": (
            summaries[selected["full"] + 1]["primary_correct"] > summaries[0]["primary_correct"]
        ),
        "response_launch_authorized": False,
    }
