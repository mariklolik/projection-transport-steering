# M5 — YES/NO reformulation answered by sampling (M4's structure x M1's sampling).
#
# HYPOTHESIS: same reasoning YES/NO reformulation as M4, but instead of reading
# P(YES) from the logits we SAMPLE `n_seeds` full reasoning rollouts per option
# and use the YES-fraction of their final answers as the score.
#
#   yes_frac_i = (# rollouts answering YES) / n_seeds
#   predicted  = argmax_i yes_frac_i ; confidence = normalized share (as in M4)
#
# The vote is parsed from the AFTER-</think> region only. Most expensive method:
# 4 options x n_seeds full traces per question (one batched call per option, so
# no padding — which also dodges the MPS padded-batch bug).
#
# `python -m behaviour_specific.overconfidence.confidence_yesno_sampled`

from __future__ import annotations

import re

from behaviour_specific.overconfidence.confidence_yesno import YESNO_CONF_THRESHOLD, YESNO_TEMPLATE, predict
from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence.mmlu.data import LETTERS, correct_letter, load_mmlu, split_think
from general.inference import generate_batch
from models_specific.active import chat_prompt

N_SEEDS = 5
MAX_NEW_TOKENS = 192


def parse_yes_no(text: str) -> bool | None:
    """True/False for the final YES/NO after </think>; None if absent/unclosed."""
    _, after = split_think(text)
    m = re.search(r"\b(yes|no)\b", after, re.IGNORECASE)
    return None if not m else m.group(1).lower() == "yes"


def yes_fraction(votes: list[bool | None], n_seeds: int) -> float:
    """Fraction of seeds voting YES (unparsed seeds count as not-YES, as in M1)."""
    return sum(v is True for v in votes) / n_seeds


def measure(model, tok, record: dict, n_seeds: int = N_SEEDS, seed: int = 0) -> dict:
    """Score one MCQ via 4 YES/NO reformulations x n_seeds sampled reasoning rollouts."""
    fracs, gens = [], []
    for opt in record["options"]:
        prompt = chat_prompt(tok, YESNO_TEMPLATE.format(question=record["question"], option=opt))
        rollouts = generate_batch(model, tok, [prompt] * n_seeds, do_sample=True, seed=seed,
                                  max_new_tokens=MAX_NEW_TOKENS)
        votes = [parse_yes_no(t) for t in rollouts]
        fracs.append(yes_fraction(votes, n_seeds))
        gens += [{"role": "yesno_sampled", "option": opt, "seed": j, "prompt": prompt,
                  "text": t, "vote": votes[j]} for j, t in enumerate(rollouts)]

    pred_idx, conf = predict(fracs)
    pred = LETTERS[pred_idx]
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": "yesno_sampled",
        "system_prompt": None, "user_prompt": [g["prompt"] for g in gens[::n_seeds]],
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct, threshold=YESNO_CONF_THRESHOLD),
        "yes_frac": [round(f, 3) for f in fracs], "generations": gens,
    }


def _selftest():
    assert parse_yes_no("<think>hmm</think>\nYES") is True
    assert parse_yes_no("<think>no no no</think>\nno, that's wrong") is False
    assert parse_yes_no("<think>yes maybe? but") is None
    assert parse_yes_no("<think>x</think>\nmaybe") is None
    assert abs(yes_fraction([True, True, True, True, False], 5) - 0.8) < 1e-9
    assert yes_fraction([None] * 5, 5) == 0.0
    idx, conf = predict([0.2, 0.2, 1.0, 0.4])
    assert idx == 2 and conf > 0.5
    print("confidence_yesno_sampled self-test passed")


if __name__ == "__main__":
    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    for r in load_mmlu(n=1, seed=7):
        res = measure(model, tok, r)
        print(f"{res['id']}: pred={res['final_answer']} gold={res['gold']} "
              f"conf={res['confidence']:.2f} yes_frac={res['yes_frac']}")
