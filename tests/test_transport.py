import numpy as np
import pytest

from projection_transport_steering.transport import (
    EmpiricalQuantileTransport,
    minimum_metric_update,
)


def test_quantile_transport_is_monotone():
    transport = EmpiricalQuantileTransport.fit(
        source=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        target=np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        clip=0.1,
    )

    mapped = transport.transform(np.linspace(-4.0, 4.0, 101))

    assert np.all(np.diff(mapped) >= 0.0)


def test_quantile_transport_uses_midranks_for_ties():
    transport = EmpiricalQuantileTransport.fit(
        source=np.array([0.0, 0.0, 1.0, 1.0]),
        target=np.array([10.0, 20.0, 30.0, 40.0]),
        clip=0.1,
    )

    mapped = transport.transform(np.array([0.0, 1.0]))

    assert mapped == pytest.approx(np.array([17.5, 32.5]))


def test_quantile_transport_clips_tails_without_extrapolation():
    transport = EmpiricalQuantileTransport.fit(
        source=np.array([0.0, 1.0, 2.0, 3.0]),
        target=np.array([0.0, 10.0, 20.0, 30.0]),
        clip=0.25,
    )

    mapped = transport.transform(np.array([-100.0, 100.0]))

    assert mapped == pytest.approx(np.array([7.5, 22.5]))


def test_metric_update_reaches_target_coordinates():
    states = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    metric = np.diag([1.0, 2.0, 4.0])
    targets = np.array([[0.5, -0.5], [2.0, 1.0]])

    updates = minimum_metric_update(states, basis, targets, metric)

    assert (states + updates) @ basis == pytest.approx(targets)


def test_metric_update_is_minimum_euclidean_lift():
    states = np.zeros((1, 3))
    basis = np.array([[1.0], [1.0], [0.0]])
    targets = np.array([[2.0]])

    update = minimum_metric_update(states, basis, targets, np.eye(3))[0]
    alternative = update + np.array([1.0, -1.0, 3.0])

    assert update @ update < alternative @ alternative


@pytest.mark.parametrize(
    ("basis", "metric"),
    [
        (np.array([[1.0, 1.0], [0.0, 0.0]]), np.eye(2)),
        (np.eye(2), np.array([[1.0, 0.0], [0.0, 0.0]])),
    ],
)
def test_metric_update_rejects_nonidentified_inputs(basis, metric):
    with pytest.raises(ValueError):
        minimum_metric_update(np.zeros((1, 2)), basis, np.zeros((1, 2)), metric)
