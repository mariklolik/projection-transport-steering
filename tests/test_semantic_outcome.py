import numpy as np
import pytest

from projection_transport_steering.semantic_outcome import (
    fit_semantic_outcome_actions,
    fit_semantic_outcome_actions_from_metric,
    sequence_score_metric,
    unit_covectors,
)


def test_sequence_score_metric_uses_frozen_relative_ridge():
    gradients = np.array([[1.0, 2.0], [3.0, 4.0]])
    second_moment = np.mean(gradients**2, axis=0)
    ridge = 0.01 * np.mean(second_moment)

    metric = sequence_score_metric(gradients)

    assert np.allclose(metric, second_moment + ridge)


def test_unit_covectors_normalizes_columns():
    covectors = np.array([[3.0, 0.0], [4.0, 2.0]])

    normalized = unit_covectors(covectors)

    assert np.allclose(np.linalg.norm(normalized, axis=0), 1.0)


def test_semantic_outcome_actions_isolate_metric_and_constraints():
    covectors = np.array([[1.0, 0.2], [0.1, 1.0], [0.0, 0.0]])
    neutral = np.array([[1.0, 0.1, 0.3], [0.2, 0.8, 0.4]])

    result = fit_semantic_outcome_actions(covectors, neutral)

    views = unit_covectors(covectors)
    assert result["semantic_outcome"]["feasible"]
    assert result["robust_euclidean"]["feasible"]
    assert np.min(views.T @ result["semantic_outcome"]["action"]) >= 1.0 - 1e-8
    assert np.min(views.T @ result["robust_euclidean"]["action"]) >= 1.0 - 1e-8
    assert np.mean(views, axis=1) @ result["pooled_sequence_metric"] == pytest.approx(1.0)
    assert np.mean(views, axis=1) @ result["pooled_euclidean"] == pytest.approx(1.0)


def test_semantic_outcome_actions_reuse_frozen_metric():
    covectors = np.array([[1.0, 0.2], [0.1, 1.0], [0.0, 0.0]])
    neutral = np.array([[1.0, 0.1, 0.3], [0.2, 0.8, 0.4]])
    metric = sequence_score_metric(neutral)

    direct = fit_semantic_outcome_actions(covectors, neutral)
    reused = fit_semantic_outcome_actions_from_metric(covectors, metric)

    assert np.array_equal(reused["metric_diagonal"], metric)
    assert np.allclose(
        reused["semantic_outcome"]["action"],
        direct["semantic_outcome"]["action"],
    )


def test_semantic_outcome_actions_preserve_contradictory_views():
    result = fit_semantic_outcome_actions(
        np.array([[1.0, -1.0], [0.0, 0.0]]),
        np.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    assert not result["semantic_outcome"]["feasible"]
    assert not result["robust_euclidean"]["feasible"]
    assert np.all(np.isnan(result["pooled_euclidean"]))
    assert np.all(np.isnan(result["pooled_sequence_metric"]))
