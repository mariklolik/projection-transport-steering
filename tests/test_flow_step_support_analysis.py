import copy

from projection_transport_steering.flow_step_support import condition_specifications
from projection_transport_steering.flow_step_support_analysis import (
    analyze_design,
    shapley_values,
)


def condition(heldout, neutral_likelihood=-0.01, neutral_kl=0.004):
    return {
        "construction_score_changes": [heldout] * 8,
        "heldout_score_changes": [heldout] * 8,
        "neutral_forward_kl": [neutral_kl] * 16,
        "neutral_score_changes": [neutral_likelihood] * 16,
    }


def concept_result(concept_id):
    conditions = {
        specification["name"]: condition(0.08)
        for specification in condition_specifications()
    }
    conditions["full"] = condition(0.1, -0.02, 0.004)
    conditions["equal_01"] = condition(0.05, -0.015, 0.003)
    conditions["equal_12"] = condition(0.2, -0.005, 0.002)
    return {
        "concept": f"concept {concept_id}",
        "concept_id": concept_id,
        "condition_order": [
            specification["name"] for specification in condition_specifications()
        ],
        "conditions": conditions,
        "status": "pass",
    }


def test_shapley_values_reconstruct_full_set_value():
    values = {
        (): 0.0,
        (0,): 1.0,
        (1,): 2.0,
        (2,): 3.0,
        (0, 1): 4.0,
        (0, 2): 5.0,
        (1, 2): 7.0,
        (0, 1, 2): 10.0,
    }

    allocations = shapley_values(values)

    assert abs(sum(allocations.values()) - 10.0) < 1e-12


def test_analyze_design_advances_only_fixed_candidate():
    receipt = {
        "condition_serialization_exact": True,
        "concepts": [concept_result(index) for index in range(12)],
        "planned_concepts": 12,
        "replay_max_abs_logit_error": 0.0,
        "status": "pass",
    }

    result = analyze_design([receipt], expected_concepts=12, bootstrap_draws=100)

    assert result["decision"] == "advance_flow_step_support_v1"
    assert result["gate"]["candidate_vs_full"]["wins"] == 12
    assert result["gate"]["candidate_vs_early"]["wins"] == 12
    assert result["gate"]["neutral_kl"]["mean_fractional_reduction"] == 0.5
    assert result["gate"]["pass"]
    assert result["shapley"]["max_reconstruction_error"] < 1e-12


def test_analyze_design_fails_closed_on_receipt_invariant():
    receipt = {
        "condition_serialization_exact": True,
        "concepts": [concept_result(index) for index in range(12)],
        "planned_concepts": 12,
        "replay_max_abs_logit_error": 0.0,
        "status": "pass",
    }
    invalid = copy.deepcopy(receipt)
    invalid["replay_max_abs_logit_error"] = 1e-6

    result = analyze_design([invalid], expected_concepts=12, bootstrap_draws=100)

    assert result["decision"] == "close_flow_step_support_v1"
    assert not result["invariants_pass"]
    assert not result["gate"]["pass"]
