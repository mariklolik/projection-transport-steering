import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "analyze_cacheback_development.py"
    spec = importlib.util.spec_from_file_location("analyze_cacheback_development", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holm_adjustment_preserves_ordered_familywise_bounds():
    module = load_script()

    adjusted = module.holm_adjusted({"h2": 0.01, "h4": 0.03, "h8": 0.04})
    levels = module.holm_alpha_levels({"h2": 0.01, "h4": 0.03, "h8": 0.04})

    assert adjusted == pytest.approx({"h2": 0.03, "h4": 0.06, "h8": 0.06})
    assert levels == pytest.approx({"h2": 0.05 / 3, "h4": 0.05 / 2, "h8": 0.05})


def test_common_doses_require_every_method_overall_and_within_concept():
    module = load_script()
    methods = module.METHODS
    records = []
    for dose in module.DOSES:
        for concept in ("third", "ing", "past"):
            for index in range(5):
                for method in methods:
                    reached = not (
                        dose == 0.5
                        and concept == "past"
                        and method == "euclidean"
                        and index == 4
                    )
                    records.append(
                        {
                            "concept": concept,
                            "dose": dose,
                            "case_id": f"{concept}-{index}",
                            "method": method,
                            "reached": reached,
                        }
                    )

    eligible, coverage = module.common_doses(records)

    assert eligible == [0.125, 0.25, 0.5]
    assert coverage["0.5"]["euclidean"]["by_concept"]["past"] == pytest.approx(0.8)

    next(
        row
        for row in records
        if row["dose"] == 0.5
        and row["concept"] == "past"
        and row["method"] == "euclidean"
        and row["reached"]
    )["reached"] = False
    eligible, coverage = module.common_doses(records)

    assert eligible == [0.125, 0.25]
    assert coverage["0.5"]["euclidean"]["by_concept"]["past"] == pytest.approx(0.6)


def test_dose_diagnostics_remain_available_when_coverage_fails():
    module = load_script()
    records = []
    for index, concept in enumerate(("third", "ing", "past")):
        for method in module.METHODS:
            horizon = method.removeprefix("cacheback_h")
            candidate = method.startswith("cacheback_h") and horizon != "0"
            records.append(
                {
                    "action_cosine_vs_h0": 0.5 if candidate else None,
                    "case_id": f"state-{index}",
                    "concept": concept,
                    "dose": 0.125,
                    "immediate_effect": 0.125,
                    "method": method,
                    "offset_kl": [0.5, 0.5] if candidate else [1.0, 1.0],
                    "predicted_kl": 1.0,
                    "realized_kl": 1.0 if candidate else 2.0,
                    "reached": not (method == "euclidean" and concept == "past"),
                }
            )

    result = module.dose_diagnostics(records, 100)

    assert result["h8"]["reduction"]["estimate"] == pytest.approx(0.5)
    assert result["h8"]["reduction"]["count"] == 3
