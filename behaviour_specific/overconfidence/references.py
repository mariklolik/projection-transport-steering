from __future__ import annotations

import argparse

import numpy as np

from behaviour_specific.overconfidence.analyze_pooled import ci, pooled_rows
from behaviour_specific.overconfidence.gate_law import auroc, law
from behaviour_specific.overconfidence.law_confirm import decision_flags, features
from general.paths import RESULTS_DIR
from general.storage import write_json

SEEDS = range(10)


def states(rows: list[dict], cut: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([r["is_correct"] for r in rows], bool)
    c = np.array([r["confidence"] for r in rows]) >= cut
    return (~y & c).astype(float), (y & c).astype(float)


def sel(w: np.ndarray, base: list[dict], rows: list[dict], cut: float = 0.5, keep: np.ndarray | None = None) -> np.ndarray:
    o0, c0 = states(base, cut)
    o1, c1 = states(rows, cut)
    w = w if keep is None else w * keep
    return (w * o0 * (1 - o1)).sum(-1) / (w * o0).sum(-1) - (w * c0 * (1 - c1)).sum(-1) / (w * c0).sum(-1)


def override(w: np.ndarray, base: list[dict], f: np.ndarray) -> np.ndarray:
    o0, c0 = states(base, 0.5)
    return (w * o0 * f).sum(-1) / (w * o0).sum(-1) - (w * c0 * f).sum(-1) / (w * c0).sum(-1)


def summary(pt: float, bs: np.ndarray) -> dict:
    return {"point": round(float(pt), 4), "ci": ci(bs), "p_one_sided": round(float((bs <= 0).mean()), 4)}


def boxed(rows: list[dict]) -> np.ndarray:
    return np.array([r.get("forced_box") not in (True, "True") for r in rows], float)


def conf_auroc(rows: list[dict]) -> float:
    return auroc(np.array([r["confidence"] for r in rows]), np.array([r["is_correct"] for r in rows], float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--robust", default="")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checks = [c.split(":") for c in args.checks.split(",")]
    conds = sorted({x for arm, _, ung, _ in checks for x in (arm, ung, f"null-{arm}", *(f"rand{s}-{arm}" for s in SEEDS))})
    base, after = pooled_rows([args.dir], "m5", conds)
    n = len(base)
    X, Q = features(args.dir, [r["id"] for r in base])
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=args.iters).astype(float)
    one = np.ones((1, n))
    res = {"n": n, "arms": {}}
    for arm, kind, ung, cfg in checks:
        f = decision_flags(kind, cfg, X, Q).astype(float)
        s_pt, s_bs = sel(one, base, after[arm])[0], sel(w, base, after[arm])
        nl_pt, nl_bs = sel(one, base, after[f"null-{arm}"])[0], sel(w, base, after[f"null-{arm}"])
        rd_pt = np.array([sel(one, base, after[f"rand{s}-{arm}"])[0] for s in SEEDS])
        rd_bs = np.mean([sel(w, base, after[f"rand{s}-{arm}"]) for s in SEEDS], axis=0)
        ov_pt, ov_bs = override(one, base, f)[0], override(w, base, f)
        ung_law = law(base, after[ung], f, [0.5])
        tpr, fpr = ung_law["curve"][0]["tpr"], ung_law["curve"][0]["fpr"]
        predicted = fpr * (1 - ung_law["rho_c"]) - tpr * (1 - ung_law["rho_o"])
        res["arms"][arm] = {
            "decision": kind, "sel": {"point": round(float(s_pt), 4), "ci": ci(s_bs)},
            "null": {"point": round(float(nl_pt), 4), "ci": ci(nl_bs)}, "vs_null": summary(s_pt - nl_pt, s_bs - nl_bs),
            "random": {"seeds": [round(float(x), 4) for x in rd_pt], "mean": round(float(rd_pt.mean()), 4),
                       "max": round(float(rd_pt.max()), 4)},
            "vs_random": summary(s_pt - rd_pt.mean(), s_bs - rd_bs),
            "override_matched": {"point": round(float(ov_pt), 4), "ci": ci(ov_bs), "tpr": tpr, "fpr": fpr},
            "steer_minus_override": {"point": round(float(s_pt - ov_pt), 4), "ci": ci(s_bs - ov_bs)},
            "eq2_predicted": round(float(predicted), 4),
            "eq2_agrees": bool(np.sign(predicted) == np.sign(s_pt - ov_pt)),
            "rho_o_ungated": ung_law["rho_o"], "rho_c_ungated": ung_law["rho_c"]}
        r = res["arms"][arm]
        print(arm, "sel", r["sel"]["point"], "null", r["null"]["point"], "rand mean/max", r["random"]["mean"], r["random"]["max"],
              "vs_rand", r["vs_random"]["point"], r["vs_random"]["ci"], "override", r["override_matched"]["point"],
              "eq2", r["eq2_predicted"], r["eq2_agrees"], flush=True)
    if args.robust:
        a, b = args.robust.split(",")
        rob = {}
        for label, kw in (("cut0.4", {"cut": 0.4}), ("cut0.6", {"cut": 0.6})):
            d_pt = sel(one, base, after[a], **kw)[0] - sel(one, base, after[b], **kw)[0]
            rob[label] = summary(d_pt, sel(w, base, after[a], **kw) - sel(w, base, after[b], **kw))
        keep = boxed(after[a]) * boxed(after[b])
        rob["boxed_both"] = summary(sel(one, base, after[a], keep=keep)[0] - sel(one, base, after[b], keep=keep)[0],
                                    sel(w, base, after[a], keep=keep) - sel(w, base, after[b], keep=keep))
        rob["d_auroc_conf"] = {x: round(conf_auroc(after[x]) - conf_auroc(base), 4) for x in (a, b)}
        res["robust"] = {"a": a, "b": b, **rob}
        print("robust", rob, flush=True)
    write_json(RESULTS_DIR / args.out, res)
