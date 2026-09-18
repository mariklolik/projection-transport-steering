# Reasoning-trace formatting helpers, shared by every benchmark/behaviour.
#
# The whole project runs the model in reasoning mode: it thinks inside literal
# <think>...</think> tags and then gives a final answer. These helpers split the
# trace from the answer and pull the boxed letter out — the parsing rules that
# keep letter/word noise inside the reasoning from being mistaken for the answer.
#
# `python -m general.reasoning` runs the self-tests (no model).

from __future__ import annotations

import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_think(text: str) -> tuple[str, str]:
    """Split a generation into (reasoning, after_reasoning) at </think>.

    If the trace never closes, everything is reasoning and `after` is "".
    A leading <think> is stripped from the reasoning part.
    """
    open_idx = text.find(THINK_OPEN)
    body = text[open_idx + len(THINK_OPEN):] if open_idx != -1 else text
    close_idx = body.find(THINK_CLOSE)
    if close_idx == -1:
        return body.strip(), ""
    return body[:close_idx].strip(), body[close_idx + len(THINK_CLOSE):].strip()


def parse_boxed_letter(text: str) -> str | None:
    """Extract the FINAL answer letter from a reasoning generation.

    Takes the LAST \\boxed{X} — the model may float earlier candidates
    mid-thought. Fallback: a standalone A-D letter in the after-</think> region
    ONLY; never in the reasoning itself, where option letters appear constantly.
    """
    matches = re.findall(r"\\?boxed\s*\{\s*([A-Da-d])", text)
    if matches:
        return matches[-1].upper()
    _, after = split_think(text)
    m = re.search(r"\b([A-Da-d])\b", after)
    return m.group(1).upper() if m else None


if __name__ == "__main__":
    think, after = split_think("<think>step 1. step 2.</think>\n\\boxed{B}")
    assert think == "step 1. step 2." and after == "\\boxed{B}"
    think, after = split_think("<think>never closes...")
    assert after == "" and think.startswith("never")
    think, after = split_think("no tags at all")
    assert think == "no tags at all" and after == ""

    assert parse_boxed_letter("<think>maybe \\boxed{A}? no.</think> \\boxed{C}") == "C"
    assert parse_boxed_letter("<think>A is wrong, B too</think>\nThe answer is D") == "D"
    assert parse_boxed_letter("<think>options A and B look plausible") is None
    assert parse_boxed_letter("\\boxed{ b }") == "B"
    print("general.reasoning self-tests passed")
