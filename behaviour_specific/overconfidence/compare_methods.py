# Run all five confidence-measurement hypotheses on the SAME questions and
# compare them. Every method is reasoning-mode.
#
# This is the experiment behind the headline finding: the methods can disagree on
# whether the model is even overconfident. "Overconfidence" is a property of the
# model x the elicitation method, not the model alone.
#
# Outputs (under results/methods_n{N}_seed{SEED}/):
#   summary.json          per-method aggregate metrics (accuracy, ECE, gap,
#                         mean confidence, parse rate, state histogram, seconds)
#   rollouts/<m>.jsonl    one rich record per question (system/user prompt, full
#                         generations, parsed answer) — for hand-checking
# Methods run fast->slow and stream to disk, so an interrupted run keeps
# everything computed so far.
#
# `python -m behaviour_specific.overconfidence.compare_methods [--n N] [--seed S]`
# --n omitted -> the WHOLE MMLU test split.

from __future__ import annotations

import argparse
import time
from collections import Counter

from behaviour_specific.overconfidence import (
    confidence_answer_distribution as m1,
    confidence_logit as m2,
    confidence_self_reported as m3,
    confidence_yesno as m4,
    confidence_yesno_sampled as m5,
)
from behaviour_specific.overconfidence.labeling import rank
from behaviour_specific.overconfidence.mmlu.data import load_mmlu
from general.metrics import ece, overconfidence_gap
from general.paths import RESULTS_DIR
from general.storage import JsonlWriter, read_json, write_json


def summarize(results: list[dict]) -> dict:
    """Aggregate metrics for one method's rollout records."""
    n = len(results)
    conf = [r["confidence"] for r in results]
    corr = [r["is_correct"] for r in results]
    return {
        "n": n,
        "accuracy": sum(corr) / n,
        "mean_confidence": sum(conf) / n,
        "ece": ece(conf, corr),
        "overconfidence_gap": overconfidence_gap(conf, corr),
        "parsed": sum(r["final_answer"] is not None for r in results),
        "mean_rank": sum(rank(r["state"]) for r in results) / n,
        "states": dict(Counter(r["state"] for r in results)),
    }


ALL_METHODS = ["m2_logit", "m3_self_report", "m4_yesno", "m1_answer_dist", "m5_yesno_sampled"]


def run_all(model, tok, records: list[dict], out_dir, only: set[str] | None = None) -> dict[str, dict]:
    """Score `records` with the selected methods (fast->slow), streaming rich rollouts.

    only: a subset of ALL_METHODS to run (default all). Lets the deterministic
    methods (m2/m3/m4) run on the full split while the sampling methods (m1/m5)
    run on a subsample.
    """
    lids = m2.letter_token_ids(tok)
    ynids = m4.yes_no_token_ids(tok)
    methods = [  # ordered cheap -> expensive
        ("m2_logit", lambda r: m2.measure(model, tok, r, letter_ids=lids)),
        ("m3_self_report", lambda r: m3.measure(model, tok, r)),
        ("m4_yesno", lambda r: m4.measure(model, tok, r, ids=ynids)),
        ("m1_answer_dist", lambda r: m1.measure(model, tok, r)),
        ("m5_yesno_sampled", lambda r: m5.measure(model, tok, r)),
    ]

    # merge into any earlier run in this dir (so cheap-full + sample-subset accumulate)
    sfile = out_dir / "summary.json"
    summaries = read_json(sfile) if sfile.exists() else {}
    for name, fn in methods:
        if only is not None and name not in only:
            continue
        t0 = time.time()
        results = []
        with JsonlWriter(out_dir / "rollouts" / f"{name}.jsonl") as w:
            for r in records:
                res = fn(r)
                w.write(res)          # full rich record incl. all generations
                results.append(res)
        s = summarize(results)
        s["seconds"] = round(time.time() - t0, 1)
        summaries[name] = s
        write_json(out_dir / "summary.json", summaries)  # rewrite after each method (partial-safe)
        print(f"{name:18s} acc={s['accuracy']:.3f} conf={s['mean_confidence']:.3f} "
              f"ECE={s['ece']:.3f} gap={s['overconfidence_gap']:+.3f} parsed={s['parsed']}/{s['n']} "
              f"[{s['seconds'] / 60:.1f} min]  {s['states']}", flush=True)
    return summaries


def _selftest():
    fake = [
        {"is_correct": True, "confidence": 0.9, "final_answer": "A", "state": "confident_right"},
        {"is_correct": False, "confidence": 0.8, "final_answer": "B", "state": "overconfident_wrong"},
    ]
    s = summarize(fake)
    assert s["accuracy"] == 0.5 and s["parsed"] == 2
    assert abs(s["mean_confidence"] - 0.85) < 1e-9
    print("compare_methods self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="questions (default: WHOLE MMLU test split)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--methods", default="", help=f"comma subset of {ALL_METHODS} (default: all)")
    args = ap.parse_args()

    _selftest()

    only = set(args.methods.split(",")) if args.methods else None
    if only and not only <= set(ALL_METHODS):
        raise SystemExit(f"unknown methods {only - set(ALL_METHODS)}; valid: {ALL_METHODS}")

    from models_specific.active import load_model

    model, tok = load_model()
    records = load_mmlu(n=args.n, seed=args.seed)
    # dir keyed by seed only, so cheap-full + sample-subset share one summary.json
    out_dir = RESULTS_DIR / f"methods_seed{args.seed}"
    shown = sorted(only) if only else "all 5"
    print(f"methods {shown} on {len(records)} MMLU questions (seed={args.seed}) -> {out_dir}\n")

    summaries = run_all(model, tok, records, out_dir, only=only)
    write_json(out_dir / "summary.json", summaries)
    print(f"\nsaved summary + rollouts under {out_dir}")
