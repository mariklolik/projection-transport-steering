# Steer the overconfidence direction DURING REASONING and measure the effect on
# BEHAVIOR (the 4-state label) and CAPABILITY (accuracy + ECE).
#
#   RQ2 — does steering the reasoning change behavior?  -> positive vs negative
#         4-state changes between unsteered and steered runs.
#   RQ3 — does steering hurt accuracy / calibration?     -> accuracy + ECE.
#   RQ4 — does the ANSWER depend on the REASONING trace? -> phase-split steering:
#         steer ONLY the <think> tokens; if the (unsteered) answer moves, the
#         answer causally depends on the trace.
#
# Scoring: M2 (logit at the trace's \boxed{ cut) AND M4 (yes/no) — a direction
# that only moves the M2 reading is probably an artifact of M2's scale; a real
# behavioral feature should move BOTH readouts. Steering modes:
#   always-on   add / ablate / conditional (hook active for prompt+trace+readout)
#   phase-split think-only / answer-only (generation split at </think>; M2 only,
#               run for the FIRST direction in --directions)
#
# This module is a data-parallel WORKER: --shard i --num-shards k takes records
# [i::k] and streams every condition's rollouts to
# results/steering/rollouts/<condition>__shard<i>.jsonl. The alpha scale (mean
# activation norm) is computed on the FULL record set so all shards steer
# identically. Merge + tables: analyze_steering. Launcher: run_steer.sh.
#
# `python -m ...steer_overconfidence [--n N] [--layer L] [--directions a,b,c]`

from __future__ import annotations

import argparse
from collections import Counter

import torch

from behaviour_specific.overconfidence import confidence_logit as logit_method
from behaviour_specific.overconfidence.confidence_logit import cut_at_box, letter_token_ids, probs_over_letters
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.labeling import behavior_change, label_from_score, rank
from behaviour_specific.overconfidence.mmlu.data import LETTERS, correct_letter, load_mmlu, mcq_prompt, parse_boxed_letter
from general.inference import (
    generate, generate_batch_chunked, next_token_logits, next_token_logits_batch,
)
from general.metrics import ece
from general.paths import RESULTS_DIR
from general.reasoning import THINK_CLOSE
from general.steering import steer_add, steer_ablate, steer_conditional, steered, steering_hook
from general.storage import write_json, write_jsonl
from models_specific.active import chat_prompt


def score_set(model, tok, records, method=logit_method, **kw):
    """Measure every record with `method` (whatever hooks are active apply)."""
    return [method.measure(model, tok, r, **kw) for r in records]


def _reasoning_len(tok, trace: str) -> dict:
    """Chars + tokens of a reasoning trace, logged on every record."""
    return {"reasoning_chars": len(trace), "reasoning_tokens": len(tok(trace, add_special_tokens=False).input_ids)}


def score_m2_batch(model, tok, records, letter_ids, batch_size, system=None):
    """Batched M2: generate all MCQ traces at once, read letter logits at the \\boxed{ cut.

    Identical to logit_method.measure per record, but generation and readout run
    over the whole batch (one hook, one forward) — the speed lever on CUDA.
    Whatever steering hook is installed applies to the whole batch. `system`
    prepends a system instruction (the prompting-baseline lever).
    """
    from behaviour_specific.overconfidence.confidence_logit import cut_at_box, probs_over_letters

    prompts = [chat_prompt(tok, mcq_prompt(r), system=system) for r in records]
    traces = generate_batch_chunked(model, tok, prompts, batch_size=batch_size)  # greedy
    cut_texts, forced = [], []
    for prompt, trace in zip(prompts, traces):
        prefix, f = cut_at_box(trace)
        cut_texts.append(prompt + prefix)
        forced.append(f)
    logits = next_token_logits_batch(model, tok, cut_texts, batch_size=batch_size)

    out = []
    for r, prompt, trace, lg, f in zip(records, prompts, traces, logits, forced):
        probs = probs_over_letters(lg, letter_ids)
        idx = int(probs.argmax())
        pred, conf = LETTERS[idx], float(probs[idx])
        is_correct = pred == correct_letter(r)
        out.append({
            "id": r["id"], "subject": r["subject"], "method": "logit",
            "system_prompt": None, "user_prompt": mcq_prompt(r),
            "final_answer": pred, "gold": correct_letter(r),
            "confidence": conf, "is_correct": is_correct,
            "state": label_from_score(conf, is_correct, threshold=logit_method.LOGIT_CONF_THRESHOLD),
            "probs": [round(p, 4) for p in probs.tolist()],
            "trace_answer": parse_boxed_letter(trace), "forced_box": f,
            **_reasoning_len(tok, trace),
            "generations": [{"role": "answer", "prompt": prompt, "text": trace}],
        })
    return out


def score_m4_batch(model, tok, records, ids, batch_size, system=None):
    """Batched M4: all 4 yes/no option traces per question, generated + read in batches."""
    from behaviour_specific.overconfidence.confidence_yesno import (
        MAX_NEW_TOKENS, YESNO_CONF_THRESHOLD, YESNO_TEMPLATE, p_yes, predict, yesno_cut,
    )

    yes_ids, no_ids = ids
    opts_per = [len(r["options"]) for r in records]
    prompts, owner = [], []
    for k, r in enumerate(records):
        for opt in r["options"]:
            prompts.append(chat_prompt(tok, YESNO_TEMPLATE.format(question=r["question"], option=opt),
                                       system=system))
            owner.append(k)
    traces = generate_batch_chunked(model, tok, prompts, batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS)
    cut_texts = [p + yesno_cut(t)[0] for p, t in zip(prompts, traces)]
    forced = [yesno_cut(t)[1] for t in traces]
    logits = next_token_logits_batch(model, tok, cut_texts, batch_size=batch_size)

    grp: dict[int, list] = {k: [] for k in range(len(records))}
    for j, k in enumerate(owner):
        grp[k].append(j)
    out = []
    for k, r in enumerate(records):
        js = grp[k]
        scores = [p_yes(logits[j], yes_ids, no_ids) for j in js]
        gens = [{"role": "yesno", "option": r["options"][t], "prompt": prompts[js[t]],
                 "text": traces[js[t]], "p_yes": round(scores[t], 4), "forced": forced[js[t]]}
                for t in range(len(js))]
        pred_idx, conf = predict(scores)
        pred = LETTERS[pred_idx]
        is_correct = pred == correct_letter(r)
        trace_join = " ".join(traces[j] for j in js)
        out.append({
            "id": r["id"], "subject": r["subject"], "method": "yesno",
            "system_prompt": None, "user_prompt": [prompts[j] for j in js],
            "final_answer": pred, "gold": correct_letter(r),
            "confidence": conf, "is_correct": is_correct,
            "state": label_from_score(conf, is_correct, threshold=YESNO_CONF_THRESHOLD),
            "p_yes": [round(s, 4) for s in scores], "forced_cuts": sum(forced[j] for j in js),
            **_reasoning_len(tok, trace_join),
            "generations": gens,
        })
    return out


VERIFY_TEMPLATE = "You answered {letter}. Is that answer correct? Reply with only YES or NO."


def verification_text(tok, r: dict, system: str | None) -> str:
    content = f"{system}\n\n{r['user_prompt']}" if system else r["user_prompt"]
    return tok.apply_chat_template(
        [{"role": "user", "content": content},
         {"role": "assistant", "content": r["generations"][0]["text"]},
         {"role": "user", "content": VERIFY_TEMPLATE.format(letter=r["final_answer"])}],
        tokenize=False, add_generation_prompt=True)


def score_m5_batch(model, tok, records, letter_ids, ids, batch_size, system=None, m2_rows=None):
    from behaviour_specific.overconfidence.confidence_yesno import YESNO_CONF_THRESHOLD, p_yes

    m2_rows = m2_rows or score_m2_batch(model, tok, records, letter_ids, batch_size, system=system)
    logits = next_token_logits_batch(model, tok, [verification_text(tok, r, system) for r in m2_rows],
                                     batch_size=batch_size)
    out = []
    for r, lg in zip(m2_rows, logits):
        conf = p_yes(lg, *ids)
        out.append({**r, "method": "ptrue", "m2_confidence": r["confidence"], "m2_state": r["state"],
                    "confidence": conf,
                    "state": label_from_score(conf, r["is_correct"], threshold=YESNO_CONF_THRESHOLD)})
    return out


def _mean_reasoning(results: list[dict], key: str) -> float | None:
    """Mean reasoning length over records that carry it (None if none do)."""
    vals = [r[key] for r in results if key in r]
    return sum(vals) / len(vals) if vals else None


def summary(results: list[dict]) -> dict:
    """Accuracy, ECE, mean confidence, mean rank, reasoning length, state histogram."""
    states = [r["state"] for r in results]
    return {
        "accuracy": sum(r["is_correct"] for r in results) / len(results),
        "ece": ece([r["confidence"] for r in results], [r["is_correct"] for r in results]),
        "mean_confidence": sum(r["confidence"] for r in results) / len(results),
        "mean_rank": sum(rank(s) for s in states) / len(states),
        "mean_reasoning_chars": _mean_reasoning(results, "reasoning_chars"),
        "mean_reasoning_tokens": _mean_reasoning(results, "reasoning_tokens"),
        "states": dict(Counter(states)),
    }


def compare(before: list[dict], after: list[dict]) -> dict:
    """Diff two scored sets (matched by id): behavior changes + capability."""
    after_by_id = {r["id"]: r for r in after}
    changes = Counter(behavior_change(b["state"], after_by_id[b["id"]]["state"]) for b in before)
    sb, sa = summary(before), summary(after)
    return {"positive_changes": changes["positive"], "negative_changes": changes["negative"],
            "before": sb, "after": sa}


def make_operator(v, operator, alpha, tau):
    """Build the residual-stream transform h -> h' for an always-on operator."""
    if operator == "add":
        return lambda h: steer_add(h, v, alpha)
    if operator == "ablate":
        return lambda h: steer_ablate(h, v)
    if operator == "conditional":
        return lambda h: steer_conditional(h, v, alpha, tau)
    raise ValueError(f"unknown operator: {operator}")


def run_steered(model, tok, records, v, layer, scorer, operator="add", alpha=0.0, tau=0.0):
    """Score `records` with an always-on steering operator installed at `layer`.

    `scorer(records) -> list[dict]` runs while the hook is live (so its generation
    and readout are both steered).
    """
    v = v.to(model.device, torch.float32)
    handle = steering_hook(model.model.layers[layer], make_operator(v, operator, alpha, tau))
    try:
        return scorer(records)
    finally:
        handle.remove()


def measure_phase_steered(model, tok, record, v, layer, alpha, phase, letter_ids):
    """M2's measurement with additive steering applied to ONE phase only.

    phase="think":  steer while generating up to </think>; answer + readout unsteered.
    phase="answer": thought unsteered; steer the continuation after </think> + readout.
    """
    if phase not in ("think", "answer"):
        raise ValueError(f"unknown phase: {phase}")
    v = v.to(model.device, torch.float32)
    fn = lambda h: steer_add(h, v, alpha)  # noqa: E731
    user = mcq_prompt(record)
    prompt = chat_prompt(tok, user)

    if phase == "think":
        with steered(model, layer, fn):
            think_part = generate(model, tok, prompt, stop_strings=[THINK_CLOSE])
        rest = generate(model, tok, prompt + think_part)
        trace = think_part + rest
        prefix, forced = cut_at_box(trace)
        logits = next_token_logits(model, tok, prompt + prefix)
    else:
        think_part = generate(model, tok, prompt, stop_strings=[THINK_CLOSE])
        with steered(model, layer, fn):
            rest = generate(model, tok, prompt + think_part)
            trace = think_part + rest
            prefix, forced = cut_at_box(trace)
            logits = next_token_logits(model, tok, prompt + prefix)

    probs = probs_over_letters(logits, letter_ids)
    idx = int(probs.argmax())
    pred, conf = LETTERS[idx], float(probs[idx])
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": f"logit_steer_{phase}",
        "system_prompt": None, "user_prompt": user,
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct, threshold=logit_method.LOGIT_CONF_THRESHOLD),
        "probs": [round(p, 4) for p in probs.tolist()],
        "trace_answer": parse_boxed_letter(trace), "forced_box": forced,
        **_reasoning_len(tok, trace),
        "generations": [{"role": f"steer_{phase}", "prompt": prompt, "text": trace}],
    }


def run_phase_steered(model, tok, records, v, layer, alpha, phase, letter_ids):
    """measure_phase_steered over a record set."""
    return [measure_phase_steered(model, tok, r, v, layer, alpha, phase, letter_ids) for r in records]


def mean_activation_norm(model, tok, records, layer, prompt_fn) -> float:
    """Mean L2 norm of the last-token residual at `layer` over `records`."""
    from general.inference import get_activations
    return sum(get_activations(model, tok, prompt_fn(r), layer).norm().item() for r in records) / len(records)


def _selftest():
    before = [{"id": "a", "state": "overconfident_wrong", "is_correct": False, "confidence": 0.9},
              {"id": "b", "state": "confident_right", "is_correct": True, "confidence": 0.9}]
    after = [{"id": "a", "state": "nonconfident_wrong", "is_correct": False, "confidence": 0.4},
             {"id": "b", "state": "confident_right", "is_correct": True, "confidence": 0.9}]
    c = compare(before, after)
    assert c["positive_changes"] == 1 and c["negative_changes"] == 1
    v, h = torch.randn(8), torch.randn(2, 3, 8)
    assert make_operator(v, "add", -1.0, 0.0)(h).shape == h.shape
    try:
        make_operator(v, "nope", 0.0, 0.0)
        raise AssertionError
    except ValueError:
        pass
    print("steer_overconfidence self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="eval questions (default 30)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--directions", default="caa",
                    help="comma-separated stems in directions/; the FIRST also gets phase-split conditions")
    ap.add_argument("--alphas", default="-1.5,-0.75,0.75,1.5", help="fractions of the mean activation norm")
    ap.add_argument("--methods", default="m2,m4", help="readout methods for always-on conditions")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--outdir", default="steering", help="results/<outdir>/ (isolate exp1 from §4)")
    ap.add_argument("--batch-size", type=int, default=64, help="batched generation/readout (CUDA speed lever)")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
    from models_specific.active import load_model

    model, tok = load_model()
    all_records = load_mmlu(n=args.n, seed=args.seed)
    records = all_records[args.shard::args.num_shards]
    dir_names = args.directions.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]
    bs = args.batch_size
    scorers = {}
    for m in args.methods.split(","):
        if m == "m2":
            lids_ = letter_token_ids(tok)
            scorers["m2"] = lambda recs, lids_=lids_: score_m2_batch(model, tok, recs, lids_, bs)
        elif m == "m4":
            ids_ = yes_no_token_ids(tok)
            scorers["m4"] = lambda recs, ids_=ids_: score_m4_batch(model, tok, recs, ids_, bs)
        else:
            raise ValueError(f"unknown method: {m}")

    # alpha scale from the FULL eval set so every shard steers identically
    norm = mean_activation_norm(model, tok, all_records, args.layer, lambda r: chat_prompt(tok, mcq_prompt(r)))
    n_cond = len(scorers) * (1 + len(dir_names) * (len(alphas) + 1)) + 2
    print(f"shard {args.shard}/{args.num_shards}: {len(records)} records, layer {args.layer}, "
          f"||h||={norm:.1f}, batch={bs}, {n_cond} conditions", flush=True)

    out_dir = RESULTS_DIR / args.outdir / "rollouts"

    def dump(tag: str, rows: list[dict]) -> None:
        write_jsonl(out_dir / f"{tag}__shard{args.shard}.jsonl", rows)
        print(f"  {tag}: {len(rows)} records done", flush=True)

    for mname, scorer in scorers.items():
        dump(f"baseline_{mname}", scorer(records))

    for dname in dir_names:
        v = torch.load(DIRECTIONS_DIR / f"{dname}.pt")["directions"][args.layer]
        for frac in alphas:
            for mname, scorer in scorers.items():
                dump(f"{dname}_a{frac:+.2f}_{mname}",
                     run_steered(model, tok, records, v, args.layer, scorer, operator="add", alpha=frac * norm))
        for mname, scorer in scorers.items():
            dump(f"{dname}_ablate_{mname}",
                 run_steered(model, tok, records, v, args.layer, scorer, operator="ablate"))

    v0 = torch.load(DIRECTIONS_DIR / f"{dir_names[0]}.pt")["directions"][args.layer]
    lids = letter_token_ids(tok)
    for phase in ("think", "answer"):
        dump(f"{dir_names[0]}_{phase}_a-1.50_m2",
             run_phase_steered(model, tok, records, v0, args.layer,
                               alpha=-1.5 * norm, phase=phase, letter_ids=lids))

    write_json(RESULTS_DIR / args.outdir / f"meta_shard{args.shard}.json",
               {"config": vars(args), "norm": norm, "n_records": len(records)})
    print(f"shard {args.shard} done -> {out_dir}")
