from projection_transport_steering.semantic_outcome_stage_a_analysis import (
    FACTORS,
    METHODS,
    analyze_concepts,
)


def concept_receipt(concept_id: int, candidate_gain: float = 0.2):
    conditions = {}
    for factor in FACTORS:
        for method in METHODS:
            gain = {
                "semantic_outcome": candidate_gain,
                "pooled_sequence_metric": 0.1,
                "robust_euclidean": 0.05,
            }.get(method, 0.0)
            conditions[f"{method}@{factor:g}"] = {
                "heldout_score_changes": [factor * gain] * 8,
                "neutral_forward_kl": [0.001] * 16,
                "neutral_score_changes": (
                    [-0.01] * 16 if method == "semantic_outcome" else [0.0] * 16
                ),
            }
    solver = {
        "complementarity_residual": 0.0,
        "feasible": True,
        "margins": [1.0] * 8,
        "stationarity_residual": 0.0,
        "status": "hard",
    }
    return {
        "action_metric_costs": {method: 1.0 for method in METHODS},
        "conditions": conditions,
        "concept_id": concept_id,
        "geometry": {
            "last_token_solver": solver,
            "mean_solver": solver,
            "robust_euclidean_solver": solver,
        },
        "status": "pass",
    }


def test_stage_a_authorizes_generation_only_after_both_primary_contrasts_pass():
    concepts = [concept_receipt(index) for index in range(12)]

    result = analyze_concepts(concepts, resamples=100)

    assert result["complete_concepts"] == 12
    assert result["stage_b_authorized"]
    assert result["passing_factors"] == list(FACTORS)
    assert result["factors"]["0.25"]["contrasts"]["robust_euclidean"]["wins"] == 12


def test_stage_a_retains_adverse_concepts_in_win_gate():
    concepts = [concept_receipt(index, -0.1 if index < 4 else 0.2) for index in range(12)]

    result = analyze_concepts(concepts, resamples=100)

    assert not result["stage_b_authorized"]
    assert result["passing_factors"] == []
