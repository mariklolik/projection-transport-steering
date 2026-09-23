from __future__ import annotations

import argparse

import numpy as np

from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.label_pool import extraction_records
from general.paths import RESULTS_DIR
from general.storage import write_json

OCW, CR = "overconfident_wrong", "confident_right"


def pooled_rows(dirs: list[str], readout: str, conds: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    banned = {r["id"] for r in extraction_records()}
    base, after, seen = [], {c: [] for c in conds}, set()
    source, field = ("m5", "m2_state") if readout == "m5m2" else (readout, "state")
    for d in dirs:
        runs = {k: [{**r, "state": r[field]} for r in v] for k, v in collect(RESULTS_DIR / d / "rollouts").items()
                if k.endswith(f"_{source}")}
        readout = source
        b = runs.get(f"baseline_{readout}")
        if not b or any(f"{c}_{readout}" not in runs for c in conds):
            print(f"skip {d}: missing {[c for c in conds if f'{c}_{readout}' not in runs]}", flush=True)
            continue
        by = {c: {r["id"]: r for r in runs[f"{c}_{readout}"]} for c in conds}
        for r in b:
            if r["id"] in seen or r["id"] in banned or any(r["id"] not in by[c] for c in conds):
                continue
            seen.add(r["id"])
            base.append(r)
            for c in conds:
                after[c].append(by[c][r["id"]])
    return base, after


def stats(w: np.ndarray, ocw: np.ndarray, cr: np.ndarray, left: np.ndarray, kept: np.ndarray,
          dacc: np.ndarray) -> dict[str, np.ndarray]:
    rm = (w * ocw * left).sum(-1) / (w * ocw).sum(-1)
    keep = (w * cr * kept).sum(-1) / (w * cr).sum(-1)
    return {"ocw_rm": rm, "cr_keep": keep, "sel": rm - (1 - keep), "dacc": (w * dacc).sum(-1) / w.sum(-1)}


def ci(x: np.ndarray) -> list[float]:
    return [round(float(np.quantile(x, 0.025)), 4), round(float(np.quantile(x, 0.975)), 4)]


def analyse(base: list[dict], after: dict[str, list[dict]], ref: str, margin: float, iters: int, seed: int) -> dict:
    n = len(base)
    ocw = np.array([r["state"] == OCW for r in base], float)
    cr = np.array([r["state"] == CR for r in base], float)
    acc0 = np.array([r["is_correct"] for r in base], float)
    w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=iters).astype(float)
    ones = np.ones(n)
    cols = {}
    for c, rows in after.items():
        left = np.array([r["state"] != OCW for r in rows], float)
        kept = np.array([r["state"] == CR for r in rows], float)
        dacc = np.array([r["is_correct"] for r in rows], float) - acc0
        cols[c] = (stats(ones, ocw, cr, left, kept, dacc), stats(w, ocw, cr, left, kept, dacc))
    out = {"n": n, "n_ocw": int(ocw.sum()), "n_cr": int(cr.sum()), "baseline_acc": round(float(acc0.mean()), 4),
           "ref": ref, "margin": margin, "methods": {}}
    for c, (pt, bs) in cols.items():
        row = {k: {"point": round(float(pt[k]), 4), "ci": ci(bs[k])} for k in pt}
        row["dacc"]["tost_equivalent"] = bool(-margin < ci(bs["dacc"])[0] and ci(bs["dacc"])[1] < margin)
        if ref in cols and c != ref:
            diff = {k: bs[k] - cols[ref][1][k] for k in pt}
            row["vs_ref"] = {k: {"point": round(float(pt[k] - cols[ref][0][k]), 4), "ci": ci(diff[k]),
                                 "p_two_sided": round(float(2 * min((diff[k] <= 0).mean(), (diff[k] >= 0).mean())), 4)}
                             for k in pt}
        out["methods"][c] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--ref", default="tuned_ours")
    ap.add_argument("--readouts", default="m4,m2")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conds = args.conds.split(",")
    res = {}
    for ro in args.readouts.split(","):
        base, after = pooled_rows(args.dirs.split(","), ro, conds)
        res[ro] = analyse(base, after, args.ref, args.margin, args.iters, seed=0)
        for c, m in res[ro]["methods"].items():
            vs = m.get("vs_ref", {}).get("sel", {})
            print(f"{ro} {c:45s} sel {m['sel']['point']:+.3f} {m['sel']['ci']} keep {m['cr_keep']['point']:.3f} "
                  f"dacc {m['dacc']['point']:+.3f} {m['dacc']['ci']} Δref {vs.get('point', 0):+.3f} {vs.get('ci', '')} "
                  f"p={vs.get('p_two_sided', '')}", flush=True)
    write_json(RESULTS_DIR / args.out, res)
