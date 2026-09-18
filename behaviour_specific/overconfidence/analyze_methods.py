# Deeper analysis of the method-comparison run: bootstrap CIs + per-question
# agreement between the five confidence-measurement methods.
#
# The §2 aggregates already show the methods DISAGREE in bulk (M2 looks
# over-confident, M4 calibrated). This sharpens that: on the SAME question, how
# often do they pick the same answer, and how often do they even agree on
# whether the model was "confident"? And it puts bootstrap 95% CIs on the
# headline metrics so the ranking is defensible.
#
# Reads results/methods_seed*/rollouts/*.jsonl (streams; keeps only the compact
# fields, not the full traces). Writes results/analysis/summary.json + prints
# tables. `python -m behaviour_specific.overconfidence.analyze_methods
# [--results-dir results]`.

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from behaviour_specific.overconfidence.compare_methods import ALL_METHODS
from general.metrics import bootstrap_ci, ece, overconfidence_gap
from general.storage import write_json

CONFIDENT_STATES = {"confident_right", "overconfident_wrong"}


def is_confident(rec: dict) -> bool:
    """Did this method call the answer 'confident' (either of the two confident states)?"""
    return rec["state"] in CONFIDENT_STATES


def load_seed(seed_dir: Path) -> dict[str, dict[str, dict]]:
    """{method: {id: compact_record}} for one seed, streaming the rollouts jsonl."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted((seed_dir / "rollouts").glob("*.jsonl")):
        by_id = {}
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_id[r["id"]] = {"final_answer": r["final_answer"], "gold": r["gold"],
                                  "is_correct": r["is_correct"], "confidence": r["confidence"],
                                  "state": r["state"]}
        out[f.stem] = by_id
    return out


# ---- metric functions over a list of compact records (for bootstrap) ----
def _acc(rs): return sum(r["is_correct"] for r in rs) / len(rs)                              # noqa: E704
def _conf(rs): return sum(r["confidence"] for r in rs) / len(rs)                             # noqa: E704
def _ece(rs): return ece([r["confidence"] for r in rs], [r["is_correct"] for r in rs])      # noqa: E704
def _gap(rs): return overconfidence_gap([r["confidence"] for r in rs], [r["is_correct"] for r in rs])  # noqa: E704


def bootstrap_table(pooled: dict[str, list[dict]]) -> dict:
    """Per-method bootstrap 95% CIs for accuracy / mean-conf / ECE / gap."""
    metrics = {"accuracy": _acc, "mean_confidence": _conf, "ece": _ece, "overconfidence_gap": _gap}
    table = {}
    for m, recs in pooled.items():
        table[m] = {name: dict(zip(("point", "lo", "hi"), bootstrap_ci(recs, fn)))
                    for name, fn in metrics.items()}
    return table


def agreement(seed_data: dict[str, dict[str, dict]], methods: list[str]) -> dict:
    """Pairwise + all-method agreement on the same questions, averaged over seeds.

    seed_data here is ONE seed's {method: {id: rec}}. Returns per-pair fractions
    for answer / 4-state / confident-call agreement, plus the all-5 fractions and
    the per-question 'how many methods call it confident' distribution.
    """
    ids = set.intersection(*(set(seed_data[m]) for m in methods))
    pair = {}
    for a, b in combinations(methods, 2):
        n = len(ids)
        same_ans = sum(seed_data[a][i]["final_answer"] == seed_data[b][i]["final_answer"] for i in ids) / n
        same_state = sum(seed_data[a][i]["state"] == seed_data[b][i]["state"] for i in ids) / n
        same_conf = sum(is_confident(seed_data[a][i]) == is_confident(seed_data[b][i]) for i in ids) / n
        pair[f"{a}|{b}"] = {"answer": same_ans, "state": same_state, "confident_call": same_conf}

    all_same_ans = sum(len({seed_data[m][i]["final_answer"] for m in methods}) == 1 for i in ids) / len(ids)
    conf_count = Counter(sum(is_confident(seed_data[m][i]) for m in methods) for i in ids)
    return {"n": len(ids), "pairwise": pair, "all_answer_agree": all_same_ans,
            "confident_count_hist": {k: conf_count.get(k, 0) for k in range(len(methods) + 1)}}


def _selftest():
    a = {"final_answer": "A", "gold": "A", "is_correct": True, "confidence": 0.9, "state": "confident_right"}
    b = {"final_answer": "B", "gold": "A", "is_correct": False, "confidence": 0.2, "state": "nonconfident_wrong"}
    assert is_confident(a) and not is_confident(b)
    seed = {"mX": {"q1": a, "q2": b}, "mY": {"q1": a, "q2": a}}
    ag = agreement(seed, ["mX", "mY"])
    assert ag["n"] == 2
    assert ag["pairwise"]["mX|mY"]["answer"] == 0.5          # q1 same (A), q2 differ (B vs A)
    assert ag["pairwise"]["mX|mY"]["confident_call"] == 0.5  # q1 both confident, q2 differ
    print("analyze_methods self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    _selftest()

    seed_dirs = sorted(Path(args.results_dir).glob("methods_seed*"))
    if not seed_dirs:
        raise SystemExit(f"no methods_seed* under {args.results_dir}")
    print(f"analyzing {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")

    per_seed = {d.name: load_seed(d) for d in seed_dirs}
    methods = [m for m in ALL_METHODS if all(m in sd for sd in per_seed.values())]

    # 1) bootstrap CIs, pooling all seeds per method
    pooled = {m: [rec for sd in per_seed.values() for rec in sd[m].values()] for m in methods}
    boot = bootstrap_table(pooled)
    print(f"\n=== bootstrap 95% CI (pooled n={sum(len(v) for v in pooled.values()) // len(methods)}/method) ===")
    print(f"{'method':18s} {'accuracy':>20s} {'ECE':>20s} {'over-conf gap':>20s}")
    for m in methods:
        b = boot[m]
        f = lambda k: f"{b[k]['point']:.3f}[{b[k]['lo']:.3f},{b[k]['hi']:.3f}]"  # noqa: E731
        print(f"{m:18s} {f('accuracy'):>20s} {f('ece'):>20s} {f('overconfidence_gap'):>20s}")

    # 2) agreement, averaged over seeds
    ags = [agreement(sd, methods) for sd in per_seed.values()]
    print("\n=== pairwise agreement on the SAME question (avg over seeds) ===")
    print(f"{'pair':40s} {'answer':>8s} {'4-state':>8s} {'conf-call':>10s}")
    for pair in ags[0]["pairwise"]:
        a = sum(x["pairwise"][pair]["answer"] for x in ags) / len(ags)
        s = sum(x["pairwise"][pair]["state"] for x in ags) / len(ags)
        c = sum(x["pairwise"][pair]["confident_call"] for x in ags) / len(ags)
        print(f"{pair:40s} {a:>8.3f} {s:>8.3f} {c:>10.3f}")
    all_ans = sum(x["all_answer_agree"] for x in ags) / len(ags)
    hist = {k: round(sum(x["confident_count_hist"][k] for x in ags) / len(ags), 1)
            for k in range(len(methods) + 1)}
    print(f"\nall {len(methods)} methods pick the SAME answer: {all_ans:.3f} of questions")
    print(f"per-question '# methods calling it confident' (0..{len(methods)}), avg count/seed: {hist}")

    out = {"methods": methods, "seeds": list(per_seed), "bootstrap": boot,
           "agreement_per_seed": ags, "all_answer_agree": all_ans, "confident_count_hist": hist}
    write_json(Path(args.results_dir) / "analysis" / "summary.json", out)
    print(f"\nsaved {args.results_dir}/analysis/summary.json")
