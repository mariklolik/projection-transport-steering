import hashlib
import runpy
from functools import partial
from pathlib import Path

import pytest

from projection_transport_steering import outcome_score


@pytest.fixture
def scorer():
    source = Path(__file__).parents[1] / ".external/math-author-source/modeling/dataset/util.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "9183a9ea7bce3c126ac33f993d21bf85cffbdb7bab10088216d286699c9e084a"
    )
    extract_answer = runpy.run_path(str(source))["last_boxed_only_string"]
    return partial(outcome_score.score_math_completion, extract_answer=extract_answer)


def test_general_math_score_accepts_noninteger_gold(scorer):
    score = scorer(
        r"Therefore $\boxed{0.5}$.", r"Work gives $\boxed{\frac{1}{2}}$.", terminated=True
    )

    assert score == {"correct": True, "extracted_answer": "1/2", "parse_status": "pass"}


@pytest.mark.parametrize(
    ("completion", "gold", "correct", "status"),
    [
        (r"\boxed{2x+2}", r"\boxed{2(x+1)}", True, "pass"),
        (r"\boxed{c + b + a}", r"\boxed{a + b + c}", True, "pass"),
        (r"\boxed{1-\cos^2 t}", r"\boxed{\sin^2 t}", True, "pass"),
        (r"\boxed{\sin^2 u}", r"\boxed{\sin^2 t}", False, "pass"),
        (r"\boxed{3}", r"\boxed{3m}", False, "pass"),
        (r"\boxed{\{2,1\}}", r"\boxed{\{1,2\}}", True, "pass"),
        (r"\boxed{(2,1)}", r"\boxed{(1,2)}", False, "pass"),
        (r"\boxed{(1,2)}", r"\boxed{(1,2)}", True, "pass"),
        (r"\boxed{-4}", r"\boxed{4}", False, "pass"),
        (r"\boxed{\sqrt{4}}.", r"\boxed{2}", True, "pass"),
        (r"\fbox{2}", r"\boxed{2}", True, "pass"),
        (r"\boxed{2}", r"\fbox{2}", True, "pass"),
        (r"\boxed{\text{No solution}}", r"\boxed{\text{no solution}}", True, "pass"),
        (r"First \boxed{2}, corrected to \boxed{3}.", r"\boxed{2}", False, "pass"),
        (r"\boxed{3}", r"First \boxed{2}, corrected to \boxed{3}.", True, "pass"),
        ("Unfinished work mentions 2, but no final answer.", r"\boxed{2}", False, "fail"),
        ("", r"\boxed{2}", False, "fail"),
    ],
)
def test_general_math_score_preserves_answer_semantics(scorer, completion, gold, correct, status):
    score = scorer(completion, gold, terminated=True)

    assert score["correct"] is correct
    assert score["parse_status"] == status


def test_general_math_score_never_rewards_budget_exhaustion(scorer):
    score = scorer(r"\boxed{2}", r"\boxed{2}", terminated=False)

    assert score == {"correct": False, "extracted_answer": "2", "parse_status": "pass"}


@pytest.mark.parametrize("gold", ["", "unparseable reference solution"])
def test_general_math_score_rejects_invalid_gold_even_for_empty_predictions(scorer, gold):
    with pytest.raises(ValueError, match="gold"):
        scorer("", gold, terminated=True)


@pytest.mark.parametrize(
    ("completion", "gold", "terminated"),
    [(None, r"\boxed{2}", True), (r"\boxed{2}", 2, True), (r"\boxed{2}", r"\boxed{2}", 1)],
)
def test_general_math_score_rejects_invalid_input_types(scorer, completion, gold, terminated):
    with pytest.raises(TypeError):
        scorer(completion, gold, terminated=terminated)


@pytest.fixture(scope="module")
def author_loader():
    scripts = Path(__file__).parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        return runpy.run_path(str(scripts / "audit_irc_sources.py"))["load_math_author"]


@pytest.fixture(scope="module")
def native_scorer(author_loader):
    extract, equivalent = author_loader(Path(__file__).parents[1])
    return partial(
        outcome_score.score_math_author_completion, extract_answer=extract, equivalent=equivalent
    )


def test_pinned_native_author_scorer_is_callable(native_scorer):
    assert callable(native_scorer)


@pytest.mark.parametrize(
    ("completion", "gold", "correct"),
    [
        (r"\boxed{0.5}", r"\boxed{\frac{1}{2}}", True),
        (r"\boxed{05:00}", r"\boxed{05\!:\!00}", True),
        (r"\boxed{2(x+1)}", r"\boxed{2x+2}", False),
        (r"\boxed{3m}", r"\boxed{3}", False),
        (r"\boxed{3\text{ mm}}", r"\boxed{3\text{ cm}}", True),
        (r"\boxed{y=2}", r"\boxed{x=2}", True),
        (r"\boxed{(1,2)}", r"\boxed{\{1,2\}}", False),
        (r"\boxed{1}, corrected to \boxed{2}", r"\boxed{2}", True),
        (r"\boxed{1} \fbox{2}", r"\boxed{1}", True),
        (r"\fbox{2}", r"\boxed{2}", False),
        (r"\boxed{}", r"\boxed{2}", False),
        ("", r"\boxed{2}", False),
    ],
)
def test_author_score_matches_unchanged_upstream_including_known_limits(
    native_scorer, completion, gold, correct
):
    assert native_scorer(completion, gold, terminated=True)["correct"] is correct


@pytest.mark.parametrize("gold", ["", r"\boxed{}", r"\boxed 2", r"\fbox{2}"])
def test_author_score_rejects_unextractable_reference(native_scorer, gold):
    with pytest.raises(ValueError, match="gold"):
        native_scorer(gold, gold, terminated=True)


def test_author_score_requires_termination(native_scorer):
    assert native_scorer(r"\boxed{2}", r"\boxed{2}", terminated=False) == {
        "correct": False,
        "extracted_answer": "2",
        "parse_status": "pass",
    }
