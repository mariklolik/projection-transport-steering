# M3 — self-reported confidence.
#
# HYPOTHESIS: just ASK the model. After its <think>...</think> reasoning and the
# \boxed{letter} answer, it reports Low/Medium/High; we parse both. The level is
# read from the AFTER-</think> region only — the reasoning may muse about
# confidence, and that is not the final self-report.
#
#   level in {Low, Medium, High} -> {0.0, 0.5, 1.0}; "confident" iff High
#
# One greedy rollout per question. The full generation is saved for hand-check.
#
# `python -m behaviour_specific.overconfidence.confidence_self_reported`

from __future__ import annotations

import re

from behaviour_specific.overconfidence.labeling import label
from behaviour_specific.overconfidence.mmlu.data import (
    correct_letter, load_mmlu, mcq_prompt, parse_boxed_letter, split_think,
)
from general.inference import generate
from models_specific.active import chat_prompt

CONF_SUFFIX = (
    "\n\nAdditionally, after your \\boxed{[letter]} answer, report your confidence "
    "in the answer using this template:\n"
    "Confidence: [Low/Medium/High]"
)
LEVEL_TO_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}
CONFIDENT_LEVELS = {"high"}
MAX_NEW_TOKENS = 384  # trace + answer + confidence line


def parse_confidence_level(text: str) -> str | None:
    """Pull the Low/Medium/High self-report from the after-</think> region.

    Tolerates brackets/asterisks the model adds ("Confidence: [Low]", "**High**").
    Returns None when the trace never closes or no report follows it.
    """
    _, after = split_think(text)
    m = re.search(r"confidence\s*:?\s*[\*\[\(]*\s*(low|medium|high)", after, re.IGNORECASE)
    return m.group(1).lower() if m else None


def measure(model, tok, record: dict) -> dict:
    """One reasoning rollout with the confidence suffix; score the self-report."""
    user = mcq_prompt(record) + CONF_SUFFIX
    prompt = chat_prompt(tok, user)
    text = generate(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    pred = parse_boxed_letter(text)
    level = parse_confidence_level(text)
    is_correct = pred == correct_letter(record)
    is_confident = level in CONFIDENT_LEVELS
    return {
        "id": record["id"], "subject": record["subject"], "method": "self_reported",
        "system_prompt": None, "user_prompt": user,
        "final_answer": pred, "gold": correct_letter(record),
        "level": level, "confidence": LEVEL_TO_SCORE.get(level, 0.0),
        "is_correct": is_correct, "state": label(is_confident, is_correct),
        "generations": [{"role": "answer", "prompt": prompt, "text": text}],
    }


def _selftest():
    assert parse_confidence_level("<think>hm</think>\n\\boxed{B}\nConfidence: High") == "high"
    assert parse_confidence_level("<think>x</think>Confidence: **Medium**") == "medium"
    assert parse_confidence_level("<think>x</think>\n[A]\nConfidence: [Low]") == "low"
    assert parse_confidence_level("<think>my confidence: high maybe?</think>\\boxed{A}") is None
    assert parse_confidence_level("Confidence: High") is None  # no closed trace
    assert label("high" in CONFIDENT_LEVELS, False) == "overconfident_wrong"
    print("confidence_self_reported self-test passed")


if __name__ == "__main__":
    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    for r in load_mmlu(n=2, seed=7):
        res = measure(model, tok, r)
        print(f"{res['id']}: pred={res['final_answer']} gold={res['gold']} "
              f"level={res['level']} -> {res['state']}")
