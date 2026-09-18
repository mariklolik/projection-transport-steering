import hashlib

import pytest

from projection_transport_steering import splits


def test_problem_normalization_preserves_math_symbols_and_case():
    assert hasattr(splits, "normalize_problem")
    assert splits.normalize_problem("  cafe\u0301\n  X = 2. ") == "café X = 2."
    assert splits.normalize_problem("x = 2") != splits.normalize_problem("X = 2")


def test_near_problem_clusters_are_transitive_and_order_invariant():
    assert hasattr(splits, "problem_clusters")
    shared = "Find the total number of possible arrangements of the distinct objects. " * 20
    rows = [shared + "abcdefghij", shared + "abcdefghik", shared + "abcdefghil", "Other x=5"]
    clusters = splits.problem_clusters(rows)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[0] != clusters[3]
    assert clusters == splits.problem_clusters(rows[::-1])[::-1]
    expected = min(
        hashlib.sha256(splits.normalize_problem(row).encode()).hexdigest() for row in rows[:3]
    )
    assert clusters[0] == expected


def test_problem_clusters_handle_duplicates_short_questions_and_empty_frame():
    assert hasattr(splits, "problem_clusters")
    assert splits.problem_clusters([]) == []
    labels = splits.problem_clusters(["x", " x ", "y", "Longer problem"])
    assert labels[0] == labels[1]
    assert len(set(labels)) == 3


@pytest.mark.parametrize("rows", [[""], ["  "], [None]])
def test_problem_clusters_reject_missing_text(rows):
    assert hasattr(splits, "problem_clusters")
    with pytest.raises((TypeError, ValueError)):
        splits.problem_clusters(rows)


def test_query_limited_graph_preserves_math_exposure_decisions():
    import inspect

    assert "query_count" in inspect.signature(splits.problem_clusters).parameters
    shared = "Solve the described mathematical arrangement problem carefully. " * 20
    problems = ["An unrelated primary question x=4", shared + "1", shared + "2"]
    full = splits.problem_clusters(problems)
    limited = splits.problem_clusters(problems, query_count=2)
    assert limited == full
