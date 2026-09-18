# M4 — YES/NO reformulation, logit readout.
#
# HYPOTHESIS: break one MCQ into FOUR independent "is <option i> the right
# answer? YES/NO" questions. Each REASONS inside <think>...</think> first; P(YES)
# is read from the logits at the point where the trace writes its final YES/NO.
#
#   p_yes_i    = P(yes-tokens) / (P(yes-tokens) + P(no-tokens))  at the answer word
#   predicted  = argmax_i p_yes_i
#   confidence = p_yes[predicted] / sum_i p_yes_i               (normalized)
#
# Per question: 4 greedy traces + 4 cut forwards. All four option traces saved.
#
# `python -m behaviour_specific.overconfidence.confidence_yesno`

from __future__ import annotations

import re

import torch

from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence.mmlu.data import LETTERS, correct_letter, load_mmlu
from general.inference import generate_batch, next_token_logits
from general.reasoning import THINK_CLOSE
from models_specific.active import chat_prompt

YESNO_TEMPLATE = (
    "Answer whether the given answer is the right answer for the following question.\n"
    "Please think step-by-step and put your thinking inside <think>...</think> tokens.\n"
    'After </think>, answer ONLY with "YES" or "NO".\n\n'
    "Question:\n{question}\nAnswer:\n{option}"
)
YESNO_CONF_THRESHOLD = 0.5
MAX_NEW_TOKENS = 192  # yes/no traces are short; cap keeps one rambler from stalling the batch


def yes_no_token_ids(tok) -> tuple[list[int], list[int]]:
    """Single-token ids for YES-like and NO-like answers.

    Covers case + leading-space variants (the model sometimes emits `▁YES`/`▁NO`).
    Only surface forms that encode to exactly ONE token are kept.
    """
    def single(words):
        return sorted({t[0] for w in words if len(t := tok.encode(w, add_special_tokens=False)) == 1})

    return (single(["YES", "Yes", "yes", " YES", " Yes", " yes"]),
            single(["NO", "No", "no", " NO", " No", " no"]))


def p_yes(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    """P(YES) vs P(NO) at the answer position, from a full-vocab logit vector."""
    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.softmax(torch.stack([yes, no]), dim=0)[0])


def yesno_cut(generation: str) -> tuple[str, bool]:
    """Truncate a trace right BEFORE its final YES/NO answer -> (prefix, forced).

    The next token after the prefix is the answer word, so its logits give P(YES)
    conditioned on the reasoning. If the trace never closes or no answer word
    follows </think>, append a scaffold (forced=True).
    """
    close = generation.find(THINK_CLOSE)
    if close == -1:
        return generation.rstrip() + "\n" + THINK_CLOSE + "\nAnswer: ", True
    after_start = close + len(THINK_CLOSE)
    m = re.search(r"\b(yes|no)\b", generation[after_start:], re.IGNORECASE)
    if m is None:
        return generation[:after_start] + "\nAnswer: ", True
    return generation[: after_start + m.start()], False


def predict(p_yes_scores: list[float]) -> tuple[int, float]:
    """Pick the option with the highest P(YES); confidence = its normalized share."""
    total = sum(p_yes_scores)
    idx = int(max(range(len(p_yes_scores)), key=lambda i: p_yes_scores[i]))
    return idx, (p_yes_scores[idx] / total if total > 0 else 0.0)


def measure(model, tok, record: dict, ids: tuple[list[int], list[int]] | None = None) -> dict:
    """Score one MCQ via 4 YES/NO reformulations, each with its own reasoning trace."""
    yes_ids, no_ids = ids if ids is not None else yes_no_token_ids(tok)
    prompts = [chat_prompt(tok, YESNO_TEMPLATE.format(question=record["question"], option=opt))
               for opt in record["options"]]
    traces = generate_batch(model, tok, prompts, max_new_tokens=MAX_NEW_TOKENS)  # greedy

    scores, gens, forced_cuts = [], [], 0
    for opt, prompt, trace in zip(record["options"], prompts, traces):
        prefix, forced = yesno_cut(trace)
        forced_cuts += forced
        score = p_yes(next_token_logits(model, tok, prompt + prefix), yes_ids, no_ids)
        scores.append(score)
        gens.append({"role": "yesno", "option": opt, "prompt": prompt, "text": trace,
                     "p_yes": round(score, 4), "forced": forced})

    pred_idx, conf = predict(scores)
    pred = LETTERS[pred_idx]
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": "yesno",
        "system_prompt": None, "user_prompt": prompts,
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct, threshold=YESNO_CONF_THRESHOLD),
        "p_yes": [round(s, 4) for s in scores], "forced_cuts": forced_cuts,
        "generations": gens,
    }


def _selftest():
    logits = torch.full((10,), -5.0)
    logits[3] = 5.0
    assert p_yes(logits, [3], [4]) > 0.99 and p_yes(logits, [4], [3]) < 0.01
    idx, conf = predict([0.1, 0.9, 0.1, 0.1])
    assert idx == 1 and conf > 0.5
    prefix, forced = yesno_cut("<think>hmm, no wait, it fits.</think>\n\nYES")
    assert prefix.endswith("</think>\n\n") and not forced
    prefix, forced = yesno_cut("<think>never closes")
    assert forced and prefix.endswith("Answer: ")
    print("confidence_yesno self-test passed")


if __name__ == "__main__":
    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    ids = yes_no_token_ids(tok)
    for r in load_mmlu(n=2, seed=7):
        res = measure(model, tok, r, ids=ids)
        print(f"{res['id']}: pred={res['final_answer']} gold={res['gold']} "
              f"conf={res['confidence']:.2f} p_yes={res['p_yes']} forced={res['forced_cuts']}")
