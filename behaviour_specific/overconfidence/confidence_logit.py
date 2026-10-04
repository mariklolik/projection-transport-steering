from __future__ import annotations

import torch

from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence.mmlu.data import (
    LETTERS, correct_letter, mcq_prompt, parse_boxed_letter,
)
from general.inference import generate, next_token_logits
from models_specific.active import chat_prompt

LOGIT_CONF_THRESHOLD = 0.5
BOX = "\\boxed{"


def letter_token_ids(tok) -> list[int]:

    return [tok.encode("{" + L, add_special_tokens=False)[-1] for L in LETTERS]


def probs_over_letters(logits: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:

    return torch.softmax(logits[letter_ids], dim=-1)


def cut_at_box(generation: str) -> tuple[str, bool]:

    idx = generation.rfind(BOX)
    if idx == -1:
        return generation.rstrip() + "\n" + BOX, True
    return generation[: idx + len(BOX)], False


def measure(model, tok, record: dict, letter_ids: list[int] | None = None) -> dict:

    if letter_ids is None:
        letter_ids = letter_token_ids(tok)

    user = mcq_prompt(record)
    prompt = chat_prompt(tok, user)
    trace = generate(model, tok, prompt)
    prefix, forced = cut_at_box(trace)
    logits = next_token_logits(model, tok, prompt + prefix)
    probs = probs_over_letters(logits, letter_ids)

    idx = int(probs.argmax())
    pred = LETTERS[idx]
    conf = float(probs[idx])
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": "logit",
        "system_prompt": None, "user_prompt": user,
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct, threshold=LOGIT_CONF_THRESHOLD),
        "probs": [round(p, 4) for p in probs.tolist()],
        "trace_answer": parse_boxed_letter(trace), "forced_box": forced,
        "generations": [{"role": "answer", "prompt": prompt, "text": trace}],
    }


def _selftest():
    logits = torch.tensor([0.0, 3.0, 0.0, 0.0, 9.9])
    probs = probs_over_letters(logits, [0, 1, 2, 3])
    assert int(probs.argmax()) == 1
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    prefix, forced = cut_at_box("<think>try \\boxed{A}? no.</think>\n\\boxed{")
    assert prefix.endswith("</think>\n\\boxed{") and not forced
    prefix, forced = cut_at_box("<think>trailed off")
    assert forced and prefix.endswith(BOX)
    print("confidence_logit self-test passed")
