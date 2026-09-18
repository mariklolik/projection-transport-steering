# M1 — answer-distribution confidence (self-agreement across sampled reasonings).
#
# HYPOTHESIS: "confidence" = how stable the model's reasoning is. Sample `n_seeds`
# full reasoning rollouts (one batched call — each row samples independently);
# the fraction landing on the modal final letter is the confidence, the modal
# letter is the answer.
#
#   confidence = count(modal letter) / n_seeds        in {1/n, ..., 1}
#
# Different sampled thoughts can end in different answers, so this measures the
# spread of the model's own reasoning. Records keep every rollout for hand-check.
#
# `python -m behaviour_specific.overconfidence.confidence_answer_distribution`

from __future__ import annotations

from collections import Counter

from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence.mmlu.data import (
    correct_letter, load_mmlu, mcq_prompt, parse_boxed_letter,
)
from general.inference import generate_batch
from models_specific.active import chat_prompt

N_SEEDS = 5
MAX_NEW_TOKENS = 256  # traces are ~50-100 tokens; a cap keeps one rambler from stalling the batch


def confidence_from_letters(letters: list[str | None]) -> tuple[str | None, float]:
    """Given parsed letters from n rollouts, return (modal_letter, confidence)."""
    valid = [x for x in letters if x is not None]
    if not valid:
        return None, 0.0
    modal, count = Counter(valid).most_common(1)[0]
    return modal, count / len(letters)  # denominator = n_seeds (unparsed = disagreement)


def measure(model, tok, record: dict, n_seeds: int = N_SEEDS, seed: int = 0) -> dict:
    """Sample n_seeds reasoning rollouts (one batched call) and score agreement."""
    user = mcq_prompt(record)
    prompt = chat_prompt(tok, user)
    rollouts = generate_batch(model, tok, [prompt] * n_seeds, do_sample=True, seed=seed,
                              max_new_tokens=MAX_NEW_TOKENS)
    letters = [parse_boxed_letter(t) for t in rollouts]

    pred, conf = confidence_from_letters(letters)
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": "answer_distribution",
        "system_prompt": None, "user_prompt": user,
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct),
        "letters": letters,
        "generations": [{"role": f"seed{i}", "prompt": prompt, "text": t, "answer": letters[i]}
                        for i, t in enumerate(rollouts)],
    }


def _selftest():
    modal, conf = confidence_from_letters(["B", "B", "A", "B", "B"])
    assert modal == "B" and abs(conf - 0.8) < 1e-9
    modal, conf = confidence_from_letters(["A", None, None, None, None])
    assert modal == "A" and abs(conf - 0.2) < 1e-9
    assert confidence_from_letters([None, None]) == (None, 0.0)
    print("confidence_answer_distribution self-test passed")


if __name__ == "__main__":
    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    for r in load_mmlu(n=2, seed=7):
        res = measure(model, tok, r)
        print(f"{res['id']}: pred={res['final_answer']} gold={res['gold']} "
              f"conf={res['confidence']:.2f} -> {res['state']}  letters={res['letters']}")
