import numpy as np
import pytest
import torch

from projection_transport_steering.semantic_outcome_evaluation import (
    match_metric_cost,
    row_forward_kl,
    row_mean_log_likelihood,
)


def test_row_mean_log_likelihood_keeps_rows_separate():
    logits = torch.zeros((2, 4, 3))
    logits[0, 1, 1] = 2.0
    logits[0, 2, 2] = 1.0
    logits[1, 0, 0] = 3.0
    batch = {
        "candidate_ids": torch.tensor([[1, 2], [0, 0]]),
        "candidate_mask": torch.tensor([[True, True], [True, False]]),
        "prefix_lengths": torch.tensor([2, 1]),
    }

    scores = row_mean_log_likelihood(logits, batch)

    expected_first = (
        torch.log_softmax(logits[0, 1], dim=-1)[1]
        + torch.log_softmax(logits[0, 2], dim=-1)[2]
    ) / 2
    expected_second = torch.log_softmax(logits[1, 0], dim=-1)[0]
    assert torch.allclose(scores, torch.stack([expected_first, expected_second]))


def test_row_forward_kl_is_zero_only_for_equal_candidate_distributions():
    base = torch.zeros((1, 3, 2))
    steered = base.clone()
    batch = {
        "candidate_ids": torch.tensor([[0, 1]]),
        "candidate_mask": torch.tensor([[True, True]]),
        "prefix_lengths": torch.tensor([1]),
    }

    assert row_forward_kl(base, steered, batch).item() == pytest.approx(0.0)
    steered[0, 0, 0] = 2.0
    assert row_forward_kl(base, steered, batch).item() > 0.0


def test_row_scores_ignore_masked_positions_beyond_a_short_row():
    logits = torch.zeros((2, 4, 2))
    batch = {
        "candidate_ids": torch.tensor([[1, 0, 0], [0, 1, 0]]),
        "candidate_mask": torch.tensor(
            [[True, False, False], [True, True, True]]
        ),
        "prefix_lengths": torch.tensor([4, 1]),
    }

    scores = row_mean_log_likelihood(logits, batch)
    divergences = row_forward_kl(logits, logits, batch)

    assert torch.all(torch.isfinite(scores))
    assert torch.equal(divergences, torch.zeros(2))


def test_match_metric_cost_preserves_direction_and_matches_reference():
    metric = np.array([2.0, 5.0])
    reference = np.array([1.0, 2.0])
    action = np.array([3.0, 1.0])

    matched = match_metric_cost(reference, action, metric)

    assert action[0] * matched[1] - action[1] * matched[0] == pytest.approx(0.0)
    assert matched @ (metric * matched) == pytest.approx(
        reference @ (metric * reference)
    )
