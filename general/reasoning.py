from __future__ import annotations

import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_think(text: str) -> tuple[str, str]:

    open_idx = text.find(THINK_OPEN)
    body = text[open_idx + len(THINK_OPEN):] if open_idx != -1 else text
    close_idx = body.find(THINK_CLOSE)
    if close_idx == -1:
        return body.strip(), ""
    return body[:close_idx].strip(), body[close_idx + len(THINK_CLOSE):].strip()


def parse_boxed_letter(text: str) -> str | None:

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
