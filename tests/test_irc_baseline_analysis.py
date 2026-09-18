import importlib
import sys
from pathlib import Path

import pytest


def module():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("analyze_irc_baseline")


def records(values):
    return [
        {
            "cluster_id": str(i),
            "primary": {"correct": bool(value), "parse_status": "pass"},
            "sensitivity": {"correct": bool(value and i != 0), "parse_status": "pass"}
            if i < 3
            else None,
            "sensitivity_supported": i < 3,
            "terminated": i != 3,
            "generated_tokens": [200, 600, 3000, 8192][i],
            "prompt_tokens": [100, 128, 255, 400][i],
            "evaluator_errors": {},
        }
        for i, value in enumerate(values)
    ]


def packet():
    source = [
        {
            "cluster_id": str(i),
            "type": "Algebra" if i < 2 else "Geometry",
            "level": f"Level {i + 1}",
        }
        for i in range(4)
    ]
    candidates = [
        {"rank": rank, "site": site, "application": application}
        for application in ("prompt", "full")
        for rank in (4, 8)
        for site in (17, 23)
    ]
    zero = records([1, 0, 0, 0])
    outcomes = [records([0, 1, 1, 0]) for _ in candidates]
    return source, zero, outcomes, candidates


def test_complete_selection_reports_all_conditions_and_fixed_ties():
    result = module().summarize(*packet(), resamples=100)
    assert len(result["conditions"]) == 9
    assert result["selected_indices"] == {"prompt": 0, "full": 4}
    assert result["baseline_entry_criterion_met"]
    assert result["evidence_label"] == "exploratory_selection"
    assert not result["response_launch_authorized"]
    contrast = result["contrasts_vs_zero"][0]
    assert contrast["primary"]["estimate"] == 0.25
    assert contrast["primary"]["count"] == 4
    assert contrast["transitions"] == {"00": 1, "01": 2, "10": 1, "11": 0}
    assert contrast["supported_primary"]["estimate"] == pytest.approx(1 / 3)
    assert contrast["supported_sensitivity"]["estimate"] == pytest.approx(2 / 3)
    assert result["conditions"][0]["scorer_disagreements"] == 1
    assert result["conditions"][1]["cap_failures"] == 1
    assert contrast["strata"]["subject"]["Algebra"]["count"] == 2
    assert contrast["strata"]["difficulty"]["Level 1"]["lower"] is None
    assert contrast["reference_condition"] == "zero"
    assert result["selected_full_vs_selected_prompt"]["reference_condition"] == "selected_prompt"
    assert contrast["rescue_rate"] == pytest.approx(2 / 3)
    assert contrast["harm_rate"] == 1
    assert contrast["retention_rate"] == 0


def test_equal_full_and_zero_does_not_open_baseline_entry():
    source, zero, outcomes, candidates = packet()
    outcomes[4:] = [records([1, 0, 0, 0]) for _ in range(4)]
    result = module().summarize(source, zero, outcomes, candidates, resamples=100)
    assert not result["baseline_entry_criterion_met"]


def test_supported_scorer_direction_reversal_is_explicit_without_changing_primary_selection():
    source, zero, outcomes, candidates = packet()
    zero[0]["sensitivity"]["correct"] = True
    for rows in outcomes:
        for row in rows[:3]:
            row["sensitivity"]["correct"] = False
    result = module().summarize(source, zero, outcomes, candidates, resamples=100)
    assert result["baseline_entry_criterion_met"]
    assert all(row["measurement_direction_reversal"] for row in result["contrasts_vs_zero"])


@pytest.mark.parametrize(
    "fault", ["missing_candidate", "missing_question", "duplicate", "support", "error"]
)
def test_incomplete_or_unpaired_results_cannot_select_a_winner(fault):
    source, zero, outcomes, candidates = packet()
    if fault == "missing_candidate":
        outcomes.pop()
    elif fault == "missing_question":
        outcomes[2].pop()
    elif fault == "duplicate":
        outcomes[2][1]["cluster_id"] = "0"
    elif fault == "support":
        outcomes[2][1]["sensitivity_supported"] = False
    else:
        outcomes[2][1]["evaluator_errors"] = {"primary": "TimeoutException"}
    with pytest.raises(ValueError):
        module().summarize(source, zero, outcomes, candidates, resamples=100)
