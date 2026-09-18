from types import SimpleNamespace

import pytest
import torch

from projection_transport_steering import torch_transport


def test_fit_fisher_basis_returns_leading_uncentered_score_subspace():
    fit = getattr(torch_transport, "fit_fisher_basis", None)
    assert fit is not None
    gradients = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    basis = fit(gradients, rank=2)

    assert basis.shape == (3, 2)
    expected = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(basis @ basis.T, expected)


def test_fisher_protected_velocity_preserves_progress_and_minimizes_metric_cost():
    project = getattr(torch_transport, "fisher_protected_velocity", None)
    assert project is not None
    velocity = torch.tensor(
        [[[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]]],
        dtype=torch.float64,
    )
    basis = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    penalty = 3.0

    projected = project(velocity, basis, penalty)

    progress = (velocity * projected).sum(-1)
    raw_progress = velocity.square().sum(-1)
    metric = torch.eye(3, dtype=torch.float64) + penalty * basis @ basis.T
    orthogonal = torch.stack(
        (-velocity[..., 1], velocity[..., 0], torch.zeros_like(velocity[..., 0])),
        dim=-1,
    )
    alternative = projected + 0.4 * orthogonal
    projected_cost = torch.einsum("...i,ij,...j->...", projected, metric, projected)
    alternative_cost = torch.einsum("...i,ij,...j->...", alternative, metric, alternative)

    assert torch.allclose(progress, raw_progress)
    assert torch.all(projected_cost < alternative_cost)


def test_fisher_protected_velocity_keeps_zero_exactly_zero():
    project = getattr(torch_transport, "fisher_protected_velocity", None)
    assert project is not None
    velocity = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    basis = torch.eye(4, dtype=torch.bfloat16)[:, :2]

    projected = project(velocity, basis, 1.0)

    assert torch.equal(projected, velocity)


def test_fisher_metric_filtered_velocity_applies_inverse_without_progress_rescaling():
    filter_velocity = getattr(torch_transport, "fisher_metric_filtered_velocity", None)
    assert filter_velocity is not None
    velocity = torch.tensor([[1.0, 2.0]])
    basis = torch.tensor([[1.0], [0.0]])

    filtered = filter_velocity(velocity, basis, 1.0)

    assert torch.allclose(filtered, torch.tensor([[0.5, 2.0]]))


@pytest.mark.parametrize(
    ("basis", "penalty"),
    (
        (torch.tensor([[1.0], [1.0]]), 1.0),
        (torch.eye(2), -1.0),
    ),
)
def test_fisher_protected_velocity_rejects_invalid_metric(basis, penalty):
    project = getattr(torch_transport, "fisher_protected_velocity", None)
    assert project is not None

    with pytest.raises(ValueError):
        project(torch.ones((1, 2)), basis, penalty)


def test_fisher_protected_flow_forwards_inputs_and_preserves_cache_identity():
    wrapper_type = getattr(torch_transport, "FisherProtectedFlow", None)
    assert wrapper_type is not None

    class Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.received = None

        def forward(self, hidden, concept, **kwargs):
            self.received = SimpleNamespace(hidden=hidden, concept=concept, kwargs=kwargs)
            return hidden * self.weight, kwargs["cache"]

    flow = Flow()
    wrapper = wrapper_type(flow, torch.eye(2)[:, :1], 2.0)
    hidden = torch.tensor([[[1.0, 2.0]]])
    concept = torch.tensor([[[3.0, 4.0]]])
    cache = object()

    velocity, returned_cache = wrapper(hidden, concept, cache=cache, t=0.5)

    assert flow.received.hidden is hidden
    assert flow.received.concept is concept
    assert flow.received.kwargs == {"cache": cache, "t": 0.5}
    assert returned_cache is cache
    assert velocity.shape == hidden.shape
    assert velocity.dtype == hidden.dtype
