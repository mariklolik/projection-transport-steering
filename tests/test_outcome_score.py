import numpy as np
import pytest

from projection_transport_steering.outcome_score import (
    allocate_aime_rows,
    extract_aime_answer,
    extract_aime_benchmark_answer,
    extract_aime_reward_answer,
    minimum_score_update,
    partition_aime_rows,
    score_aime_benchmark_completion,
    score_aime_completion,
    score_aime_math_verify_completion,
    score_aime_reward_completion,
)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Work\nFinal Answer: 0", 0),
        ("Work\nFinal Answer: 042", 42),
        ("Work\nFinal Answer: 999\n", 999),
        ("Work\nFinal Answer: -1", None),
        ("Work\nFinal Answer: 1000", None),
        ("Work\nFinal Answer: 42\nmore", None),
        ("Work\nFinal Answer: 41\nFinal Answer: 42", None),
        ("Work\n\\boxed{42}", None),
    ],
)
def test_extract_aime_answer_requires_one_terminal_declared_form(completion, expected):
    assert extract_aime_answer(completion) == expected


def test_score_aime_completion_keeps_parse_status_separate_from_correctness():
    assert score_aime_completion("Final Answer: 42", 42) == {
        "correct": True,
        "extracted_answer": 42,
        "parse_status": "pass",
    }


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Work\nFinal Answer: 042", 42),
        ("Work\n$$\n\\boxed{042}\n$$", 42),
        ("Work \\(\\boxed{999}\\)", 999),
        ("Work \\boxed{-1}", None),
        ("Work \\boxed{1000}", None),
        ("Work \\boxed{42}\nmore", None),
        ("Work \\boxed{41}\nthen \\boxed{42}", 42),
        ("Work says 42", None),
    ],
)
def test_benchmark_answer_accepts_only_a_terminal_bounded_integer(completion, expected):
    assert extract_aime_benchmark_answer(completion) == expected


def test_benchmark_scorer_keeps_parse_separate_from_exact_correctness():
    assert score_aime_benchmark_completion("Work\n\\boxed{42}", 42) == {
        "correct": True,
        "extracted_answer": 42,
        "parse_status": "pass",
    }
    assert score_aime_benchmark_completion("Work\n\\boxed{41}", 42) == {
        "correct": False,
        "extracted_answer": 41,
        "parse_status": "pass",
    }
    assert score_aime_completion("answer is 42", 42) == {
        "correct": False,
        "extracted_answer": None,
        "parse_status": "fail",
    }


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Work\nFinal Answer: -1", -1),
        ("Work\n\\boxed{1786}", 1786),
        ("Work\n$$\n\\boxed{-999999999999}\n$$", -999999999999),
        ("Work\n\\boxed{1000000000000}", None),
        ("Work\n\\boxed{1786}\nmore", None),
        ("Work says 1786", None),
    ],
)
def test_reward_answer_accepts_any_terminal_bounded_integer(completion, expected):
    assert extract_aime_reward_answer(completion) == expected


def test_reward_scorer_maps_out_of_range_prediction_to_incorrect_label():
    assert score_aime_reward_completion("Work\n\\boxed{1786}", 504) == {
        "correct": False,
        "extracted_answer": 1786,
        "parse_status": "pass",
    }


@pytest.mark.parametrize(
    ("completion", "answer", "expected"),
    [
        ("Result \\[ \\boxed{307} \\]", 307, True),
        ("Result \\(\\boxed{1504}\\).", 504, False),
        ("Result \\(\\boxed{15}\\).", 15, True),
        ("Result \\[\\boxed{50}.\\]", 600, False),
        ("Result \\(\\boxed{88}\\).", 32, False),
    ],
)
def test_math_verify_scorer_handles_observed_terminal_forms(completion, answer, expected):
    score = score_aime_math_verify_completion(completion, answer)

    assert score["parse_status"] == "pass"
    assert score["correct"] is expected


@pytest.mark.parametrize(
    ("completion", "expected_status", "expected_correct"),
    [
        ("mentions 504 then final \\(\\boxed{1504}\\).", "pass", False),
        ("intermediate \\(\\boxed{504}\\), corrected to \\(\\boxed{1504}\\).", "pass", False),
        ("final \\(\\boxed{504}\\). trailing prose 1504", "pass", True),
        ("unfinished reasoning says 504 but no boxed answer", "fail", False),
    ],
)
def test_math_verify_scorer_is_terminal_and_fail_closed(
    completion, expected_status, expected_correct
):
    score = score_aime_math_verify_completion(completion, 504)

    assert score["parse_status"] == expected_status
    assert score["correct"] is expected_correct


def test_allocate_aime_rows_is_deterministic_and_year_separated():
    rows = [
        {"year": 2000 + index // 50, "index": index, "problem": f"p{index}", "answer": index % 1000}
        for index in range(975)
    ]
    rows.extend(
        {"year": year, "index": index, "problem": f"p{year}-{index}", "answer": index}
        for year in (2024, 2025)
        for index in range(30)
    )

    first = allocate_aime_rows(rows, "dataset-sha")
    second = allocate_aime_rows(reversed(rows), "dataset-sha")

    first_ids = {(row["group_id"], row["allocation"]) for row in first}
    second_ids = {(row["group_id"], row["allocation"]) for row in second}
    counts = {
        stage: sum(row["allocation"] == stage for row in first)
        for stage in {
            "basis",
            "fit",
            "calibration",
            "validation",
            "reserve",
            "development",
            "pilot",
        }
    }
    assert first_ids == second_ids
    assert counts == {
        "basis": 32,
        "fit": 256,
        "calibration": 128,
        "validation": 128,
        "reserve": 431,
        "development": 30,
        "pilot": 30,
    }


def test_allocate_aime_rows_rejects_wrong_historical_count():
    with pytest.raises(ValueError, match="975"):
        allocate_aime_rows([], "dataset-sha")


def test_partition_aime_rows_keeps_stage_payloads_disjoint():
    rows = [
        {"year": 2000 + index // 50, "index": index, "problem": f"p{index}", "answer": index % 1000}
        for index in range(975)
    ]
    rows.extend(
        {"year": year, "index": index, "problem": f"p{year}-{index}", "answer": index}
        for year in (2024, 2025)
        for index in range(30)
    )

    packets = partition_aime_rows(rows, "dataset-sha")

    assert set(packets) == {
        "basis",
        "fit",
        "calibration",
        "validation",
        "reserve",
        "development",
        "pilot",
    }
    all_ids = [row["group_id"] for packet in packets.values() for row in packet]
    assert len(all_ids) == len(set(all_ids)) == 1035


def test_minimum_score_update_reaches_increment_and_beats_nullspace_alternative():
    gradient = np.array([1.0, 2.0, 0.0])
    metric = np.diag([1.0, 4.0, 2.0])

    update = minimum_score_update(gradient, metric, 0.3)
    alternative = update + np.array([2.0, -1.0, 3.0])

    assert update @ gradient == pytest.approx(0.3)
    assert update @ metric @ update < alternative @ metric @ alternative


def test_minimum_score_update_returns_exact_zero_for_zero_increment():
    assert minimum_score_update(np.ones(3), np.eye(3), 0.0) == pytest.approx(np.zeros(3))


@pytest.mark.parametrize(
    ("gradient", "metric", "increment"),
    [
        (np.zeros(3), np.eye(3), 0.1),
        (np.ones(3), np.zeros((3, 3)), 0.1),
        (np.ones(3), np.eye(3), -0.1),
    ],
)
def test_minimum_score_update_rejects_invalid_action(gradient, metric, increment):
    with pytest.raises(ValueError):
        minimum_score_update(gradient, metric, increment)
