# Merge + report for the sharded steering run (CPU-only, no model).
#
# Reads results/steering/rollouts/<condition>__shard<k>.jsonl (written by the
# steer_overconfidence workers), pools the shards, matches every condition
# against its readout method's pooled baseline by question id, and reports per
# condition: accuracy / ECE / mean confidence / OCW count / 4-state changes,
# with PAIRED bootstrap 95% CIs on the deltas (resampling questions).
#
# The cross-method table then answers the key question of the steering phase:
# does a direction move the M2 reading AND the M4 reading, or only one of them?
# (A direction that only moves M2 is likely an artifact of M2's scale.)
#
# Also sanity-checks the M2 baseline against the unsteered eval run
# (results/methods_seed7) — greedy decoding is deterministic, so the shared
# questions must get the same answers.
#
# `python -m behaviour_specific.overconfidence.analyze_steering`

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from behaviour_specific.overconfidence.labeling import rank
from behaviour_specific.overconfidence.steer_overconfidence import summary
from general.metrics import bootstrap_ci
from general.paths import RESULTS_DIR
from general.storage import read_jsonl, write_json

STEER_DIR = RESULTS_DIR / "steering"      # overridden by --steer-dir at the CLI


def collect(rollouts_dir: Path) -> dict[str, list[dict]]:
    """{condition_tag: pooled records} — merges <tag>__shard<k>.jsonl files."""
    pooled: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(rollouts_dir.glob("*__shard*.jsonl")):
        tag = f.stem.rsplit("__shard", 1)[0]
        pooled[tag].extend(read_jsonl(f))
    return dict(pooled)


def parse_tag(tag: str) -> tuple[str, str]:
    """'caa_a-1.50_m2' -> ('caa_a-1.50', 'm2'); 'baseline_m4' -> ('baseline', 'm4')."""
    rest, method = tag.rsplit("_", 1)
    return rest, method


def pair_up(before: list[dict], after: list[dict]) -> list[tuple[dict, dict]]:
    """Match records by question id (only ids present in both)."""
    b_by_id = {r["id"]: r for r in before}
    return [(b_by_id[r["id"]], r) for r in after if r["id"] in b_by_id]


# ---- paired delta metrics over [(before, after)] pairs, for bootstrap ----
def _d_conf(ps): return sum(a["confidence"] - b["confidence"] for b, a in ps) / len(ps)          # noqa: E704
def _d_acc(ps): return sum(a["is_correct"] - b["is_correct"] for b, a in ps) / len(ps)           # noqa: E704
def _d_rank(ps): return sum(rank(a["state"]) - rank(b["state"]) for b, a in ps) / len(ps)        # noqa: E704
def _d_ocw(ps): return sum((a["state"] == "overconfident_wrong") - (b["state"] == "overconfident_wrong")
                           for b, a in ps) / len(ps)                                              # noqa: E704


def deltas(before: list[dict], after: list[dict]) -> dict:
    """Paired bootstrap CIs for the confidence/accuracy/rank/OCW deltas."""
    pairs = pair_up(before, after)
    out = {"n_pairs": len(pairs)}
    for name, fn in [("d_conf", _d_conf), ("d_acc", _d_acc), ("d_rank", _d_rank), ("d_ocw_rate", _d_ocw)]:
        point, lo, hi = bootstrap_ci(pairs, fn)
        out[name] = {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    return out


def ci_str(d: dict) -> str:
    return f"{d['point']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]"


def baseline_sanity(baseline_m2: list[dict], eval_m2_path: Path) -> float | None:
    """Answer agreement between the steering M2 baseline and the eval-run M2 records."""
    if not eval_m2_path.exists():
        return None
    eval_by_id = {r["id"]: r["final_answer"] for r in read_jsonl(eval_m2_path)}
    shared = [r for r in baseline_m2 if r["id"] in eval_by_id]
    if not shared:
        return None
    return sum(eval_by_id[r["id"]] == r["final_answer"] for r in shared) / len(shared)


def _selftest():
    assert parse_tag("caa_a-1.50_m2") == ("caa_a-1.50", "m2")
    assert parse_tag("baseline_m4") == ("baseline", "m4")
    assert parse_tag("probe_behavioral_m4_ablate_m4") == ("probe_behavioral_m4_ablate", "m4")
    b = [{"id": "a", "confidence": 0.9, "is_correct": False, "state": "overconfident_wrong"},
         {"id": "b", "confidence": 0.9, "is_correct": True, "state": "confident_right"}]
    a = [{"id": "a", "confidence": 0.4, "is_correct": False, "state": "nonconfident_wrong"},
         {"id": "b", "confidence": 0.8, "is_correct": True, "state": "confident_right"}]
    d = deltas(b, a)
    assert d["n_pairs"] == 2
    assert abs(d["d_conf"]["point"] + 0.3) < 1e-6
    assert abs(d["d_ocw_rate"]["point"] + 0.5) < 1e-6 and d["d_rank"]["point"] == 0.5
    print("analyze_steering self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-dir", default="steering", help="results/<steer-dir>/ to analyze")
    args = ap.parse_args()

    _selftest()

    STEER_DIR = RESULTS_DIR / args.steer_dir
    pooled = collect(STEER_DIR / "rollouts")
    if not pooled:
        raise SystemExit(f"no rollouts under {STEER_DIR / 'rollouts'}")
    baselines = {m: pooled[f"baseline_{m}"] for m in ("m2", "m4") if f"baseline_{m}" in pooled}
    print(f"{len(pooled)} conditions, baselines: "
          f"{ {m: len(v) for m, v in baselines.items()} }")

    agree = baseline_sanity(baselines.get("m2", []),
                            RESULTS_DIR / "methods_seed7" / "rollouts" / "m2_logit.jsonl")
    if agree is not None:
        print(f"sanity: steering M2 baseline vs eval-run answers agree {agree:.3f} (greedy => ~1.0 expected)")

    out = {"baseline_sanity_answer_agreement": agree, "baselines": {}, "conditions": {}}
    for m, recs in baselines.items():
        out["baselines"][m] = summary(recs)
        s = out["baselines"][m]
        print(f"baseline_{m}: acc {s['accuracy']:.3f}  ece {s['ece']:.3f}  conf {s['mean_confidence']:.3f}  "
              f"states {s['states']}")

    print(f"\n{'condition':38s} {'n':>4} {'acc':>6} {'conf':>6} {'OCW':>4} {'rtok':>5}   "
          f"{'d_conf (95% CI)':>22} {'d_rank (95% CI)':>22} {'d_ocw (95% CI)':>22}")
    rows = {}
    for tag in sorted(pooled):
        rest, method = parse_tag(tag)
        if rest == "baseline" or method not in baselines:
            continue
        s = summary(pooled[tag])
        d = deltas(baselines[method], pooled[tag])
        rows[tag] = {"summary": s, "deltas": d}
        rtok = f"{s['mean_reasoning_tokens']:.0f}" if s["mean_reasoning_tokens"] else "-"
        print(f"{tag:38s} {d['n_pairs']:>4} {s['accuracy']:>6.3f} {s['mean_confidence']:>6.3f} "
              f"{s['states'].get('overconfident_wrong', 0):>4} {rtok:>5}   "
              f"{ci_str(d['d_conf']):>22} {ci_str(d['d_rank']):>22} {ci_str(d['d_ocw_rate']):>22}")
    out["conditions"] = rows

    # cross-method pivot: same steering condition, two readouts
    by_rest = defaultdict(dict)
    for tag, r in rows.items():
        rest, method = parse_tag(tag)
        by_rest[rest][method] = r
    print(f"\n{'steering condition':34s} {'d_conf m2':>22} {'d_conf m4':>22}   both moved?")
    pivot = {}
    for rest in sorted(by_rest):
        r = by_rest[rest]
        if "m2" not in r or "m4" not in r:
            continue
        c2, c4 = r["m2"]["deltas"]["d_conf"], r["m4"]["deltas"]["d_conf"]
        both = (c2["hi"] < 0 and c4["hi"] < 0) or (c2["lo"] > 0 and c4["lo"] > 0)
        pivot[rest] = {"d_conf_m2": c2, "d_conf_m4": c4, "both_moved_same_sign": both}
        print(f"{rest:34s} {ci_str(c2):>22} {ci_str(c4):>22}   {'YES' if both else 'no'}")
    out["cross_method"] = pivot

    write_json(STEER_DIR / "analysis.json", out)
    print(f"\nsaved {STEER_DIR / 'analysis.json'}")
