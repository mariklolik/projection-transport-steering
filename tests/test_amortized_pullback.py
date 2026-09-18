import pytest
import torch

from projection_transport_steering import torch_transport


def test_score_gradient_basis_recovers_leading_pullback_fisher_direction():
    weights = torch.tensor(
        [[2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    probabilities = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.float64)
    mean = probabilities @ weights
    scores = weights - mean
    exact_metric = scores.T @ (probabilities[:, None] * scores)
    expected = torch.linalg.eigh(exact_metric).eigenvectors[:, -1]
    generator = torch.Generator().manual_seed(20260906)
    sampled = torch.multinomial(probabilities, 20_000, replacement=True, generator=generator)

    fitted = torch_transport.fit_fisher_basis(scores[sampled], rank=1).squeeze(1)

    assert torch.abs(torch.dot(fitted, expected)) > 0.995


def test_restricted_metric_steering_is_feasible_and_minimal_in_subspace():
    solve = getattr(torch_transport, "restricted_metric_steering", None)
    assert solve is not None
    basis = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    metric = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    covector = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)

    update, inverse_quadratic = solve(basis, metric, covector, progress=1.5)

    coordinates = basis.T @ update
    alternative_coordinates = coordinates + 0.4 * torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert covector @ update == pytest.approx(1.5)
    assert torch.dot(coordinates, metric @ coordinates) < torch.dot(
        alternative_coordinates,
        metric @ alternative_coordinates,
    )
    assert inverse_quadratic == pytest.approx(
        torch.dot(basis.T @ covector, torch.linalg.solve(metric, basis.T @ covector))
    )


def test_restricted_metric_steering_matches_full_solution_when_contained():
    solve = getattr(torch_transport, "restricted_metric_steering", None)
    assert solve is not None
    full_metric = torch.tensor(
        [[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=torch.float64,
    )
    covector = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    full_direction = torch.linalg.solve(full_metric, covector)
    first = full_direction / full_direction.norm()
    second = torch.tensor([-first[1], first[0], 0.0], dtype=torch.float64)
    basis = torch.stack((first, second), dim=1)
    restricted_metric = basis.T @ full_metric @ basis

    update, inverse_quadratic = solve(basis, restricted_metric, covector, progress=0.7)
    expected_energy = covector @ full_direction
    expected = 0.7 * full_direction / expected_energy

    assert torch.allclose(update, expected)
    assert inverse_quadratic == pytest.approx(expected_energy)


def test_restricted_metric_steering_q_only_is_euclidean_direction():
    solve = getattr(torch_transport, "restricted_metric_steering", None)
    assert solve is not None
    covector = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    basis = (covector / covector.norm()).unsqueeze(1)
    restricted_metric = torch.tensor([[7.0]], dtype=torch.float64)

    update, _ = solve(basis, restricted_metric, covector, progress=2.5)

    assert torch.allclose(update, 2.5 * covector / covector.square().sum())


@pytest.mark.parametrize(
    ("basis", "metric", "covector"),
    (
        (torch.ones(3), torch.eye(1), torch.ones(3)),
        (torch.eye(3)[:, :2], torch.ones((2, 3)), torch.ones(3)),
        (torch.eye(3)[:, :2], torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.ones(3)),
        (torch.eye(3)[:, :2], torch.eye(2), torch.tensor([0.0, 0.0, 1.0])),
    ),
)
def test_restricted_metric_steering_rejects_invalid_inputs(basis, metric, covector):
    solve = getattr(torch_transport, "restricted_metric_steering", None)
    assert solve is not None

    with pytest.raises(ValueError):
        solve(basis, metric, covector, progress=1.0)
