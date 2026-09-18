# Analysis: per-condition summaries, paired bootstrap deltas, 4-state
# transition matrices, surgical metrics (ocw_removal / cr_retention /
# selectivity with CI), selective prediction (AURC/E-AURC). Writes
# analysis_v2.json + report_v2.md.
# Run: python -m behaviour_specific.overconfidence.analyze_steering_v2 --steer-dir steering_v2

from __future__ import annotations

import random
from collections import defaultdict

from behaviour_specific.overconfidence.analyze_steering import ci_str, collect, deltas, pair_up, parse_tag
from behaviour_specific.overconfidence.labeling import STATES
from behaviour_specific.overconfidence.steer_overconfidence import summary
from general.paths import RESULTS_DIR
from general.storage import write_json

OCW, CR = "overconfident_wrong", "confident_right"


def transition_matrix(pairs: list[tuple[dict, dict]]) -> dict[str, dict[str, int]]:
    """{before_state: {after_state: count}} over paired records."""
    tm: dict[str, dict[str, int]] = {s: {t: 0 for t in STATES} for s in STATES}
    for b, a in pairs:
        tm[b["state"]][a["state"]] += 1
    return tm


def surgical(pairs: list[tuple[dict, dict]]) -> dict | None:
    """ocw_removal / cr_retention / selectivity (None if a pole is empty)."""
    ocw_pairs = [(b, a) for b, a in pairs if b["state"] == OCW]
    cr_pairs = [(b, a) for b, a in pairs if b["state"] == CR]
    if not ocw_pairs or not cr_pairs:
        return None
    removal = sum(a["state"] != OCW for _, a in ocw_pairs) / len(ocw_pairs)
    retention = sum(a["state"] == CR for _, a in cr_pairs) / len(cr_pairs)
    return {"n_ocw": len(ocw_pairs), "n_cr": len(cr_pairs),
            "ocw_removal": round(removal, 4), "cr_retention": round(retention, 4),
            "selectivity": round(removal - (1 - retention), 4)}


def aurc(records: list[dict]) -> dict:
    """Selective-prediction metrics from (confidence, is_correct) pairs.

    Risk-coverage curve: sort by confidence DESC; risk(c) = error rate among the
    top-c fraction. AURC = mean risk over all coverages (lower better);
    E-AURC = AURC − AURC of the oracle ordering (errors last). Post-hoc monotone
    recalibration (e.g. temperature scaling) leaves these INVARIANT — only an
    intervention that changes answers or their ranking can move them.
    """
    n = len(records)
    order = sorted(records, key=lambda r: -r["confidence"])
    errs = [0.0] * n
    run = 0
    for i, r in enumerate(order):
        run += (not r["is_correct"])
        errs[i] = run / (i + 1)
    a = sum(errs) / n
    k_err = sum(not r["is_correct"] for r in records)
    oracle = sum((max(0, (i + 1) - (n - k_err))) / (i + 1) for i in range(n)) / n
    cov_at_10 = max((i + 1) / n for i in range(n) if errs[i] <= 0.10) if errs[0] <= 0.10 else 0.0
    return {"aurc": round(a, 4), "e_aurc": round(a - oracle, 4),
            "coverage_at_risk10": round(cov_at_10, 4)}


def selectivity_ci(pairs: list[tuple[dict, dict]], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI on selectivity, resampling questions."""
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        s = surgical([pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))])
        if s is not None:
            vals.append(s["selectivity"])
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def _selftest():
    mk = lambda i, s: {"id": f"q{i}", "state": s, "confidence": 0.5, "is_correct": s.endswith("right")}  # noqa: E731
    before = [mk(0, OCW), mk(1, OCW), mk(2, CR), mk(3, CR), mk(4, "nonconfident_wrong")]
    after_surgical = [mk(0, "nonconfident_wrong"), mk(1, "nonconfident_wrong"), mk(2, CR), mk(3, CR),
                      mk(4, "nonconfident_wrong")]
    after_knob = [mk(0, "nonconfident_wrong"), mk(1, "nonconfident_wrong"),
                  mk(2, "nonconfident_right"), mk(3, "nonconfident_right"), mk(4, "nonconfident_wrong")]
    ps = pair_up(before, after_surgical)
    s = surgical(ps)
    assert s["selectivity"] == 1.0 and s["ocw_removal"] == 1.0 and s["cr_retention"] == 1.0
    s2 = surgical(pair_up(before, after_knob))
    assert s2["selectivity"] == 0.0
    tm = transition_matrix(ps)
    assert tm[OCW]["nonconfident_wrong"] == 2 and tm[CR][CR] == 2
    lo, hi = selectivity_ci(ps, iters=200)
    assert lo <= 1.0 <= hi or hi <= 1.0

    mkr = lambda c, ok: {"confidence": c, "is_correct": ok}  # noqa: E731
    assert aurc([mkr(0.9, True), mkr(0.8, True)])["aurc"] == 0.0
    perfect = [mkr(0.9, True), mkr(0.8, True), mkr(0.2, False)]
    a = aurc(perfect)
    assert a["e_aurc"] == 0.0 and a["coverage_at_risk10"] >= 2 / 3
    inverted = [mkr(0.9, False), mkr(0.2, True), mkr(0.1, True)]
    assert aurc(inverted)["aurc"] > a["aurc"]
    print("analyze_steering_v2 self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-dir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    sdir = RESULTS_DIR / args.steer_dir
    pooled = collect(sdir / "rollouts")
    if not pooled:
        raise SystemExit(f"no rollouts under {sdir / 'rollouts'}")
    baselines = {m: pooled[f"baseline_{m}"] for m in ("m2", "m4") if f"baseline_{m}" in pooled}

    out = {"baselines": {m: {**summary(r), "selective": aurc(r)} for m, r in baselines.items()},
           "conditions": {}}
    md = ["# PTS steering grid — surgical report", "",
          "| condition | readout | acc | conf | ECE | OCW | ocw_removal | cr_retention | selectivity [95% CI] | Δconf | Δacc |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, recs in baselines.items():
        s = out["baselines"][m]
        md.append(f"| baseline | {m} | {s['accuracy']:.3f} | {s['mean_confidence']:.3f} | {s['ece']:.3f} | "
                  f"{s['states'].get(OCW, 0)} | — | — | — | — | — |")
        print(f"baseline_{m}: acc {s['accuracy']:.3f} ece {s['ece']:.3f} conf {s['mean_confidence']:.3f} "
              f"states {s['states']}")

    for tag in sorted(pooled):
        rest, method = parse_tag(tag)
        if rest == "baseline" or method not in baselines:
            continue
        pairs = pair_up(baselines[method], pooled[tag])
        s = summary(pooled[tag])
        d = deltas(baselines[method], pooled[tag])
        surg = surgical(pairs)
        tm = transition_matrix(pairs)
        row = {"summary": s, "deltas": d, "surgical": surg, "transitions": tm,
               "selective": aurc(pooled[tag])}
        if surg:
            lo, hi = selectivity_ci(pairs)
            row["selectivity_ci"] = [round(lo, 4), round(hi, 4)]
            sel_str = f"{surg['selectivity']:+.3f} [{lo:+.3f},{hi:+.3f}]"
        else:
            sel_str = "—"
        out["conditions"][tag] = row
        md.append(f"| {rest} | {method} | {s['accuracy']:.3f} | {s['mean_confidence']:.3f} | {s['ece']:.3f} | "
                  f"{s['states'].get(OCW, 0)} | {surg['ocw_removal'] if surg else '—'} | "
                  f"{surg['cr_retention'] if surg else '—'} | {sel_str} | "
                  f"{ci_str(d['d_conf'])} | {ci_str(d['d_acc'])} |")
        print(f"{tag:40s} sel={sel_str:24s} ocw_rm={surg['ocw_removal'] if surg else '—'} "
              f"cr_keep={surg['cr_retention'] if surg else '—'} dconf={ci_str(d['d_conf'])}")

    # cross-readout: same condition under m2 and m4
    by_rest = defaultdict(dict)
    for tag in out["conditions"]:
        rest, method = parse_tag(tag)
        by_rest[rest][method] = out["conditions"][tag]
    md += ["", "## Cross-readout selectivity", "",
           "| condition | selectivity m2 | selectivity m4 |", "|---|---|---|"]
    for rest in sorted(by_rest):
        r = by_rest[rest]
        f = lambda m: (f"{r[m]['surgical']['selectivity']:+.3f}" if m in r and r[m]["surgical"] else "—")  # noqa: E731
        md.append(f"| {rest} | {f('m2')} | {f('m4')} |")

    write_json(sdir / "analysis_v2.json", out)
    (sdir / "report_v2.md").write_text("\n".join(md) + "\n")
    print(f"\nsaved {sdir / 'analysis_v2.json'} and report_v2.md")
