from __future__ import annotations


STATES = ["overconfident_wrong", "nonconfident_wrong", "nonconfident_right", "confident_right"]
RANK = {s: i for i, s in enumerate(STATES)}


CONF_THRESHOLD = 0.5


def label(is_confident: bool, is_correct: bool) -> str:

    if is_correct:
        return "confident_right" if is_confident else "nonconfident_right"
    return "overconfident_wrong" if is_confident else "nonconfident_wrong"


def label_from_score(confidence: float, is_correct: bool, threshold: float = CONF_THRESHOLD) -> str:

    return label(confidence >= threshold, is_correct)


def rank(state: str) -> int:

    return RANK[state]


def behavior_change(before: str, after: str) -> str:

    return "positive" if RANK[after] > RANK[before] else "negative"
