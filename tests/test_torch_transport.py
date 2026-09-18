import numpy as np
import pytest
import torch

from projection_transport_steering.torch_transport import (
    FullCovarianceTransportAction,
    GaussianCoordinateTransportAction,
    ProjectionTransportAction,
    ScoreConditionedAction,
    SphericalSteeringAction,
)
from projection_transport_steering.transport import EmpiricalQuantileTransport


def test_projection_transport_action_reaches_mapped_coordinate():
    empirical = EmpiricalQuantileTransport.fit(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        0.1,
    )
    basis = torch.tensor([[1.0], [1.0], [0.0]])
    action = ProjectionTransportAction.from_empirical(
        basis=basis,
        metric=torch.eye(3),
        transport=empirical,
        strength=1.0,
    )
    hidden = torch.tensor([[[0.5, -0.5, 4.0], [1.0, 0.0, 3.0]]])

    steered = action(hidden)
    expected = empirical.transform((hidden @ basis).numpy().reshape(-1))

    assert torch.allclose((steered @ basis).reshape(-1), torch.tensor(expected).float())


def test_projection_transport_action_interpolates_strength():
    empirical = EmpiricalQuantileTransport.fit(
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        0.1,
    )
    basis = torch.tensor([[1.0], [0.0]])
    action = ProjectionTransportAction.from_empirical(
        basis=basis,
        metric=torch.eye(2),
        transport=empirical,
        strength=0.5,
    )
    hidden = torch.tensor([[[0.0, 2.0]]])

    steered = action(hidden)

    assert torch.allclose((steered @ basis).reshape(-1), torch.tensor([1.0]))


def test_projection_transport_euclidean_constructor_avoids_dense_metric():
    empirical = EmpiricalQuantileTransport.fit(
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        0.1,
    )
    basis = torch.tensor([[2.0], [0.0]])

    action = ProjectionTransportAction.euclidean(basis, empirical, 1.0)
    steered = action(torch.tensor([[[0.0, 4.0]]]))

    assert torch.allclose((steered @ basis).reshape(-1), torch.tensor([2.0]))


def test_score_conditioned_action_is_exact_noop_below_threshold():
    action = torch.nn.Linear(2, 2, bias=False)
    action.weight.data = 2.0 * torch.eye(2)
    gated = ScoreConditionedAction(
        action,
        direction=torch.tensor([1.0, 0.0]),
        slope=4.0,
        intercept=0.0,
        threshold=0.5,
    )
    hidden = torch.tensor([[[-1.0, 3.0], [1.0, 3.0]]])

    steered = gated(hidden)

    assert torch.equal(steered[0, 0], hidden[0, 0])
    assert torch.equal(steered[0, 1], 2.0 * hidden[0, 1])


def test_gaussian_coordinate_transport_matches_target_moments():
    source = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
    target = torch.tensor([[8.0, 3.0], [12.0, 7.0]])
    action = GaussianCoordinateTransportAction.fit(source, target, strength=1.0)

    transformed = action(source)

    assert transformed.mean(0) == pytest.approx(target.mean(0))
    assert transformed.std(0) == pytest.approx(target.std(0))


def test_spherical_steering_preserves_norm_and_respects_gate():
    action = SphericalSteeringAction(
        target=torch.tensor([1.0, 0.0]),
        source=torch.tensor([-1.0, 0.0]),
        kappa=20.0,
        alpha=0.5,
        beta=0.0,
    )
    hidden = torch.tensor([[[-1.0, 1.0], [1.0, 0.0]]])

    steered = action(hidden)

    assert steered.norm(dim=-1) == pytest.approx(hidden.norm(dim=-1))
    assert steered[0, 0, 0] > hidden[0, 0, 0]
    assert torch.equal(steered[0, 1], hidden[0, 1])


def test_full_covariance_transport_matches_regularized_moments():
    source = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -2.0], [0.0, 2.0]],
        dtype=torch.float64,
    )
    transform = torch.tensor([[1.5, 0.5], [0.5, 1.0]], dtype=torch.float64)
    target = source @ transform + torch.tensor([3.0, -2.0])
    action = FullCovarianceTransportAction.fit(
        source,
        target,
        strength=1.0,
        regularization=1e-8,
    )

    transformed = action(source)

    assert transformed.mean(0) == pytest.approx(target.mean(0), abs=1e-6)
    assert torch.cov(transformed.T) == pytest.approx(torch.cov(target.T), abs=1e-6)
