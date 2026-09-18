# The 4-state overconfidence labeling system.
#
# Every (confidence, correctness) pair maps to one of four states, worst (0) to
# best (3):
#
#     0  overconfident_wrong   (confident AND wrong)   <- the failure we target
#     1  nonconfident_wrong    (unsure   AND wrong)
#     2  nonconfident_right    (unsure   AND right)
#     3  confident_right       (confident AND right)   <- the goal
#
# "Positive behavior change" = the state's rank goes UP after an intervention;
# no change or a drop is negative.
#
# `python -m behaviour_specific.overconfidence.labeling` runs the self-tests.

from __future__ import annotations

# ordered worst -> best; the index IS the rank
STATES = ["overconfident_wrong", "nonconfident_wrong", "nonconfident_right", "confident_right"]
RANK = {s: i for i, s in enumerate(STATES)}

# Default: a continuous confidence in [0, 1] counts as "confident" at/above this.
CONF_THRESHOLD = 0.5


def label(is_confident: bool, is_correct: bool) -> str:
    """Return the 4-state label for a (confident?, correct?) pair."""
    if is_correct:
        return "confident_right" if is_confident else "nonconfident_right"
    return "overconfident_wrong" if is_confident else "nonconfident_wrong"


def label_from_score(confidence: float, is_correct: bool, threshold: float = CONF_THRESHOLD) -> str:
    """Threshold a continuous confidence, then label."""
    return label(confidence >= threshold, is_correct)


def rank(state: str) -> int:
    """Rank of a state (0=worst overconfident-wrong ... 3=best confident-right)."""
    return RANK[state]


def behavior_change(before: str, after: str) -> str:
    """'positive' if the state's rank strictly improved, else 'negative'."""
    return "positive" if RANK[after] > RANK[before] else "negative"


if __name__ == "__main__":
    assert label(True, False) == "overconfident_wrong"
    assert label(True, True) == "confident_right"
    assert label(False, True) == "nonconfident_right"
    assert label(False, False) == "nonconfident_wrong"
    assert rank("overconfident_wrong") == 0 < rank("confident_right") == 3
    assert label_from_score(0.9, is_correct=False) == "overconfident_wrong"
    assert label_from_score(0.1, is_correct=False) == "nonconfident_wrong"
    assert behavior_change("overconfident_wrong", "nonconfident_wrong") == "positive"
    assert behavior_change("confident_right", "confident_right") == "negative"
    print("behaviour.overconfidence.labeling self-tests passed")
