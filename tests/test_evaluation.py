import numpy as np
import pytest

from projection_transport_steering.evaluation import (
    compose_action_probabilities,
    contrast_direction,
    format_multiple_choice,
    fit_truthfulqa_temperature,
    multiclass_brier,
    temperature_probabilities,
    truthfulqa_mc,
)


def test_format_multiple_choice_uses_declared_alphabet():
    prompt = format_multiple_choice("Capital of France?", ["Paris", "Rome"], ["X", "Y"])

    assert prompt == "Capital of France?\nX. Paris\nY. Rome\nAnswer:"


def test_format_multiple_choice_rejects_mismatched_alphabet():
    with pytest.raises(ValueError):
        format_multiple_choice("Question", ["one", "two"], ["A"])


def test_multiclass_brier_returns_paired_item_losses():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])

    losses = multiclass_brier(probabilities, np.array([0, 1]))

    assert losses == pytest.approx(np.array([0.08, 0.32]))


def test_contrast_direction_is_unit_mean_difference():
    positive = np.array([[3.0, 1.0], [1.0, 1.0]])
    negative = np.array([[0.0, 0.0], [0.0, 0.0]])

    direction = contrast_direction(positive, negative)

    assert direction == pytest.approx(np.array([2.0, 1.0]) / np.sqrt(5.0))


def test_contrast_direction_rejects_zero_difference():
    with pytest.raises(ValueError):
        contrast_direction(np.ones((2, 3)), np.ones((2, 3)))


def test_compose_action_probabilities_selects_each_rows_action():
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.3, 0.7], [0.4, 0.6]],
        ]
    )

    composed = compose_action_probabilities(probabilities, np.array([0, 1]))

    assert composed == pytest.approx(np.array([[0.9, 0.1], [0.4, 0.6]]))


def test_truthfulqa_mc_matches_reference_definitions():
    scores = np.log(np.array([0.5, 0.2, 0.3]))

    metrics = truthfulqa_mc(scores, np.array([1, 1, 0]), best_index=0)

    assert metrics == pytest.approx({"mc1": 1.0, "mc2": 0.7, "mc3": 0.5})


def test_truthfulqa_mc_requires_true_false_and_truthful_best():
    with pytest.raises(ValueError):
        truthfulqa_mc(np.array([0.0, 1.0]), np.array([1, 1]), best_index=0)
    with pytest.raises(ValueError):
        truthfulqa_mc(np.array([0.0, 1.0]), np.array([1, 0]), best_index=1)


def test_temperature_probabilities_are_normalized_and_rank_preserving():
    probabilities = temperature_probabilities(np.array([2.0, 0.0, -1.0]), 2.0)

    assert probabilities.sum() == pytest.approx(1.0)
    assert np.argmax(probabilities) == 0


def test_truthfulqa_temperature_softens_overconfident_wrong_scores():
    temperature = fit_truthfulqa_temperature(
        [np.log(np.array([0.99, 0.01]))],
        [np.array([0, 1])],
    )

    assert temperature > 1.0
