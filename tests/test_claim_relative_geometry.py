import numpy as np
import pytest

from projection_transport_steering.claim_relative_geometry import (
    diagonal_metric_robust_action,
    minimum_sequence_metric_action,
    nested_sequence_metrics,
    reduced_identity_robust_action,
    robust_metric_action,
)


def test_diagonal_metric_robust_action_matches_dense_problem():
    diagonal = np.array([2.0, 3.0, 5.0, 7.0])
    covectors = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.3, 1.0, 0.2],
            [0.1, 0.4, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    dense = robust_metric_action(np.diag(diagonal), covectors, target=0.7)

    reduced = diagonal_metric_robust_action(diagonal, covectors, target=0.7)

    assert reduced["feasible"]
    assert np.allclose(reduced["action"], dense["action"], atol=1e-8)
    assert np.allclose(reduced["margins"], dense["margins"], atol=1e-8)
    assert reduced["metric_cost"] == pytest.approx(dense["objective"])
    assert reduced["stationarity_residual"] < 1e-8
    assert reduced["reduced_rank"] == 3


def test_diagonal_metric_robust_action_preserves_infeasibility():
    result = diagonal_metric_robust_action(
        np.array([2.0, 3.0]),
        np.array([[1.0, -1.0], [0.0, 0.0]]),
        target=1.0,
    )

    assert not result["feasible"]
    assert result["status"] == "infeasible"
    assert np.isnan(result["metric_cost"])


def test_diagonal_metric_robust_action_handles_small_metric_scale():
    generator = np.random.default_rng(0)
    covectors = generator.normal(size=(16, 8))
    covectors /= np.linalg.norm(covectors, axis=0)
    diagonal = 10 ** generator.uniform(-9, -6, size=16)

    result = diagonal_metric_robust_action(diagonal, covectors, target=1.0)

    assert result["feasible"]
    assert np.min(covectors.T @ result["action"]) >= 1.0 - 1e-8
    assert result["stationarity_residual"] < 1e-8


@pytest.mark.parametrize(
    "diagonal",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan]),
        np.ones((2, 1)),
    ],
)
def test_diagonal_metric_robust_action_rejects_invalid_metric(diagonal):
    with pytest.raises(ValueError, match="metric diagonal"):
        diagonal_metric_robust_action(diagonal, np.eye(2), target=1.0)


def test_robust_metric_action_satisfies_orthogonal_active_constraints():
    result = robust_metric_action(
        np.diag([2.0, 1.0]),
        np.eye(2),
        target=1.0,
    )

    assert result["feasible"]
    assert np.allclose(result["action"], [1.0, 1.0], atol=1e-8)
    assert np.allclose(result["margins"], [1.0, 1.0], atol=1e-8)
    assert np.allclose(result["slack"], 0.0, atol=1e-8)
    assert result["stationarity_residual"] < 1e-8
    assert result["complementarity_residual"] < 1e-8


def test_robust_metric_action_vector_target_matches_scalar_target():
    metric = np.diag([2.0, 1.0])
    covectors = np.eye(2)

    scalar = robust_metric_action(metric, covectors, target=0.7)
    vector = robust_metric_action(metric, covectors, target=np.array([0.7, 0.7]))

    assert np.allclose(vector["action"], scalar["action"], atol=1e-10)
    assert np.allclose(vector["dual"], scalar["dual"], atol=1e-10)
    assert vector["objective"] == pytest.approx(scalar["objective"])


def test_robust_metric_action_satisfies_heterogeneous_vector_target():
    result = robust_metric_action(
        np.diag([2.0, 1.0]),
        np.eye(2),
        target=np.array([0.25, 1.5]),
    )

    assert result["feasible"]
    assert np.allclose(result["action"], [0.25, 1.5], atol=1e-8)
    assert np.allclose(result["margins"], [0.25, 1.5], atol=1e-8)
    assert result["stationarity_residual"] < 1e-8
    assert result["complementarity_residual"] < 1e-8


def test_robust_metric_action_reports_infeasible_vector_target():
    result = robust_metric_action(
        np.eye(2),
        np.array([[1.0, -1.0], [0.0, 0.0]]),
        target=np.array([1.0, 2.0]),
    )

    assert not result["feasible"]
    assert result["status"] == "infeasible"


def test_robust_metric_action_handles_redundant_views():
    result = robust_metric_action(
        np.eye(2),
        np.array([[1.0, 1.0], [0.0, 0.0]]),
        target=0.5,
    )

    assert result["feasible"]
    assert np.allclose(result["action"], [0.5, 0.0], atol=1e-8)
    assert np.all(result["dual"] >= 0.0)


def test_robust_metric_action_reports_contradictory_hard_views():
    result = robust_metric_action(
        np.eye(2),
        np.array([[1.0, -1.0], [0.0, 0.0]]),
        target=1.0,
    )

    assert not result["feasible"]
    assert result["status"] == "infeasible"


def test_robust_metric_action_uses_only_predeclared_l1_slack():
    result = robust_metric_action(
        np.eye(2),
        np.array([[1.0, -1.0], [0.0, 0.0]]),
        target=1.0,
        slack_penalty=2.0,
    )

    assert result["feasible"]
    assert result["status"] == "soft"
    assert np.allclose(result["action"], [0.0, 0.0], atol=1e-8)
    assert np.allclose(result["slack"], [1.0, 1.0], atol=1e-8)
    assert np.all(result["dual"] <= 2.0 + 1e-10)


def test_robust_metric_action_rejects_non_spd_metric():
    with pytest.raises(ValueError, match="positive definite"):
        robust_metric_action(
            np.diag([1.0, -1.0]),
            np.eye(2),
            target=1.0,
        )


def test_reduced_identity_robust_action_matches_full_problem():
    covectors = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    full = robust_metric_action(np.eye(4), covectors, target=0.7)

    reduced = reduced_identity_robust_action(covectors, target=0.7)

    assert reduced["feasible"]
    assert np.allclose(reduced["action"], full["action"], atol=1e-8)
    assert np.allclose(reduced["margins"], full["margins"], atol=1e-8)
    assert reduced["reduced_rank"] == 2


def test_nested_sequence_metrics_are_exact_horizon_sums():
    terms = np.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    metrics = nested_sequence_metrics(terms, horizons=(0, 1, 2))

    assert np.allclose(metrics[0], terms[0])
    assert np.allclose(metrics[1], terms[0] + terms[1])
    assert np.allclose(metrics[2], terms.sum(axis=0))


def test_nested_sequence_metrics_reject_non_psd_increment():
    with pytest.raises(ValueError, match="positive semidefinite"):
        nested_sequence_metrics(
            np.array([[[1.0, 0.0], [0.0, -0.1]]]),
            horizons=(0,),
        )


def test_minimum_sequence_metric_action_matches_target_and_closed_form():
    metric = np.diag([3.0, 1.0])
    covector = np.array([1.0, 2.0])
    result = minimum_sequence_metric_action(
        metric,
        covector,
        target=0.7,
        regularization=0.2,
    )
    system = metric + 0.2 * np.eye(2)
    inverse = np.linalg.solve(system, covector)
    expected = 0.7 * inverse / (covector @ inverse)

    assert np.allclose(result["action"], expected)
    assert covector @ result["action"] == pytest.approx(0.7)
    assert result["quadratic_cost"] == pytest.approx(0.5 * expected @ system @ expected)
