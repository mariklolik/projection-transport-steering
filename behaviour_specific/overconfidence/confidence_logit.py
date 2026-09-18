# M2 — logit confidence at the trace's own answer position.
#
# HYPOTHESIS: "confidence" = the softmax probability the model puts on its chosen
# answer letter, read AFTER its own reasoning. Flow:
#   1. greedy-generate the full <think>...</think> trace + \boxed{X} answer;
#   2. cut the generation at the LAST "\boxed{" (the final-answer scaffold);
#   3. one forward over prompt + reasoning-up-to-the-cut; softmax the next-token
#      logits restricted to the four letter tokens.
#
#   confidence = max softmax prob over {A,B,C,D}     in [0.25, 1]
#
# One greedy trace + one readout forward per question. Every record carries the
# system/user prompt, the full trace, and the parsed answer, for hand-checking.
#
# `python -m behaviour_specific.overconfidence.confidence_logit` -> self-test + live demo.

from __future__ import annotations

import torch

from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence.mmlu.data import (
    LETTERS, correct_letter, load_mmlu, mcq_prompt, parse_boxed_letter,
)
from general.inference import generate, next_token_logits
from models_specific.active import chat_prompt

LOGIT_CONF_THRESHOLD = 0.5
BOX = "\\boxed{"


def letter_token_ids(tok) -> list[int]:
    """Token id emitted for each option letter right after `\\boxed{`.

    Tokenization is context-sensitive, so we encode "{A", "{B", ... and take the
    final token — verified to be exactly the tokens the model ranks highest there.
    """
    return [tok.encode("{" + L, add_special_tokens=False)[-1] for L in LETTERS]


def probs_over_letters(logits: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
    """Softmax the full-vocab logits restricted to the 4 letter tokens -> [4] probs."""
    return torch.softmax(logits[letter_ids], dim=-1)


def cut_at_box(generation: str) -> tuple[str, bool]:
    """Truncate a trace right AFTER its last `\\boxed{` -> (prefix, forced).

    The next token after the prefix is the answer letter. If the model never
    wrote \\boxed{, append it (forced=True) so the readout still works.
    """
    idx = generation.rfind(BOX)
    if idx == -1:
        return generation.rstrip() + "\n" + BOX, True
    return generation[: idx + len(BOX)], False


def measure(model, tok, record: dict, letter_ids: list[int] | None = None) -> dict:
    """Generate the reasoning trace, then read letter logits at its \\boxed{ cut."""
    if letter_ids is None:
        letter_ids = letter_token_ids(tok)

    user = mcq_prompt(record)
    prompt = chat_prompt(tok, user)
    trace = generate(model, tok, prompt)  # greedy -> deterministic reasoning
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
    logits = torch.tensor([0.0, 3.0, 0.0, 0.0, 9.9])  # index 1 dominates among first 4
    probs = probs_over_letters(logits, [0, 1, 2, 3])
    assert int(probs.argmax()) == 1
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    prefix, forced = cut_at_box("<think>try \\boxed{A}? no.</think>\n\\boxed{")
    assert prefix.endswith("</think>\n\\boxed{") and not forced
    prefix, forced = cut_at_box("<think>trailed off")
    assert forced and prefix.endswith(BOX)
    print("confidence_logit self-test passed")


if __name__ == "__main__":
    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    lids = letter_token_ids(tok)
    for r in load_mmlu(n=3, seed=7):
        res = measure(model, tok, r, letter_ids=lids)
        print(f"{res['id']}: pred={res['final_answer']} (trace {res['trace_answer']}) "
              f"gold={res['gold']} conf={res['confidence']:.2f} -> {res['state']} "
              f"forced={res['forced_box']} probs={res['probs']}")
