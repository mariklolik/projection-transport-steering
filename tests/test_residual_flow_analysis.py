import copy

from projection_transport_steering.residual_flow_analysis import analyze_design


def concept_result(concept_id, passing):
    candidate_change = 0.2 if passing else -0.1
    pooled_change = 0.1
    conditions = {
        method: {
            "action_cosine_to_candidate": 1.0 if method == "residual_metric" else 0.5,
            "heldout_score_changes": [change] * 8,
            "metric_cost": 1.0,
            "neutral_forward_kl": [0.001] * 16,
            "neutral_score_changes": [-0.01] * 16,
            "realized_margins": [0.2] * 8,
        }
        for method, change in {
            "euclidean_residual": 0.05,
            "pooled_residual_metric": pooled_change,
            "residual_metric": candidate_change,
            "retained_pooled_sequence_metric": 0.04,
            "shuffled_deficits": 0.03,
        }.items()
    }
    quantile = {
        "candidate_solver": {
            "active_constraints": 4,
            "dual_concentration": 0.4,
            "feasible": True,
        },
        "conditions": conditions,
        "deficits": [0.2] * 8,
        "target": 0.3,
    }
    return {
        "concept": f"concept {concept_id}",
        "concept_id": concept_id,
        "construction": {
            "changes": [-0.1 + concept_id / 100.0] * 8,
        },
        "quantiles": {
            "0.5": quantile,
            "0.75": copy.deepcopy(quantile),
            "1": copy.deepcopy(quantile),
        },
        "status": "pass",
    }


def test_analyze_design_selects_smallest_passing_quantile():
    concepts = [concept_result(index, passing=index < 10) for index in range(12)]
    for concept in concepts:
        concept["quantiles"]["0.75"]["conditions"]["residual_metric"][
            "heldout_score_changes"
        ] = [-0.1] * 8
        concept["quantiles"]["1"]["conditions"]["residual_metric"][
            "heldout_score_changes"
        ] = [-0.1] * 8
    receipt = {
        "concepts": concepts,
        "replay_max_abs_logit_error": 0.0,
        "status": "pass",
    }

    result = analyze_design([receipt], expected_concepts=12, bootstrap_draws=100)

    assert result["decision"] == "advance_residual_flow_v1"
    assert result["selected_quantile"] == 0.5
    assert result["quantiles"]["0.5"]["gate"]["candidate_vs_flas_wins"] == 10
    assert result["quantiles"]["0.5"]["gate"]["candidate_vs_pooled_wins"] == 10
    assert not result["quantiles"]["0.75"]["gate"]["pass"]


def test_analyze_design_fails_closed_on_receipt_invariant():
    receipt = {
        "concepts": [concept_result(index, passing=True) for index in range(12)],
        "replay_max_abs_logit_error": 1e-6,
        "status": "pass",
    }

    result = analyze_design([receipt], expected_concepts=12, bootstrap_draws=100)

    assert result["decision"] == "close_residual_flow_v1"
    assert result["selected_quantile"] is None
    assert not result["invariants_pass"]
