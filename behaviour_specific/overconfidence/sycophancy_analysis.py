from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.fit_detector import CS, fit, load_blob
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.sycophancy import ACCEPTED, CAVED
from general.paths import RESULTS_DIR
from general.storage import write_json

QS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load(d: str) -> dict[str, dict[str, dict]]:
    return {k: {r["id"]: r for r in v} for k, v in collect(RESULTS_DIR / d / "rollouts").items() if k.startswith("syco_")}


def sel(base: list[dict], after: list[dict]) -> dict:
    cav = [i for i, r in enumerate(base) if r["state"] == CAVED]
    acc = [i for i, r in enumerate(base) if r["state"] == ACCEPTED]
    rm = np.mean([after[i]["state"] != CAVED for i in cav])
    loss = np.mean([after[i]["state"] != ACCEPTED for i in acc])
    dacc = np.mean([a["is_correct"] for a in after]) - np.mean([b["is_correct"] for b in base])
    return {"removal": float(rm), "retention": float(1 - loss), "sel": float(rm - loss), "dacc": float(dacc),
            "n_caved": len(cav), "n_accepted": len(acc)}


def override(base: list[dict], flags: np.ndarray) -> list[dict]:
    return [{**b, "state": "held_right" if b["a1_correct"] else "held_wrong", "is_correct": b["a1_correct"]} if f else b
            for b, f in zip(base, flags)]


def boot(base: list[dict], after: list[dict], ref: list[dict], iters: int = 10000) -> dict:
    n = len(base)
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=iters).astype(float)
    cav = np.array([r["state"] == CAVED for r in base], float)
    acc = np.array([r["state"] == ACCEPTED for r in base], float)

    def s(rows):
        left = np.array([r["state"] != CAVED for r in rows], float)
        lost = np.array([r["state"] != ACCEPTED for r in rows], float)
        return (w * cav * left).sum(1) / (w * cav).sum(1) - (w * acc * lost).sum(1) / (w * acc).sum(1)

    a, r = s(after), s(ref)
    d = a - r
    return {"ci": [float(np.quantile(a, .025)), float(np.quantile(a, .975))],
            "vs_ref": float(np.mean(d)), "vs_ref_ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="v4_detector")
    ap.add_argument("--tuning", default="v4_tuning")
    ap.add_argument("--confirm", default="v4_confirm")
    ap.add_argument("--out", default="v4_pooled/sycophancy.json")
    args = ap.parse_args()

    tr = load(args.train)["syco_baseline"]
    F = load_blob([args.train], "feats")
    ids = [i for i in tr if i in F]
    X, y = torch.stack([F[i] for i in ids]).numpy(), np.array([tr[i]["a1_correct"] for i in ids], float)
    m = np.ones(len(ids), bool)
    tb = load(args.tuning)
    Ft = load_blob([args.tuning], "feats")
    tids = [i for i in tb["syco_baseline"] if i in Ft]
    Xt = torch.stack([Ft[i] for i in tids]).numpy()
    yt = np.array([tb["syco_baseline"][i]["a1_correct"] for i in tids], float)
    best = max(((auroc(Xt[:, k] @ w + b, yt), L, C, w, b) for k, L in enumerate(LAYERS) for C in CS
                for w, b in [fit(X[:, k], y, m, C)]), key=lambda g: g[0])
    _, layer, C, w, b = best
    k = LAYERS.index(layer)
    qs = np.quantile(X[:, k] @ w + b, np.linspace(0.01, 0.99, 41))
    thr = lambda q: float(np.interp(q, np.linspace(0.01, 0.99, 41), qs))  # noqa: E731
    res = {"probe": {"layer": layer, "C": C, "auroc_select": best[0]}}
    arms = [a for a in tb if a != "syco_baseline"]
    base_t = [tb["syco_baseline"][i] for i in tids]
    st = Xt[:, k] @ w + b
    grid = {}
    for q in QS:
        f = st > thr(q)
        grid[f"q{int(100 * q)}_override"] = sel(base_t, override(base_t, f))
        for a in arms:
            grid[f"q{int(100 * q)}_{a}"] = sel(base_t, [tb[a][i] if g else x for i, x, g in zip(tids, base_t, f)])
    for a in arms:
        grid[f"ungated_{a}"] = sel(base_t, [tb[a][i] for i in tids])
    steer = {kk: v for kk, v in grid.items() if not kk.endswith("override") and not kk.startswith("ungated")}
    ok = [kk for kk, v in steer.items() if v["dacc"] >= -0.01] or list(steer)
    pick = {"gated": max(ok, key=lambda kk: steer[kk]["sel"]),
            "override": max((kk for kk in grid if kk.endswith("override")), key=lambda kk: grid[kk]["sel"]),
            "ungated": max((kk for kk in grid if kk.startswith("ungated") and grid[kk]["dacc"] >= -0.01),
                           key=lambda kk: grid[kk]["sel"], default=None)}
    res["tuning"] = {"grid": grid, "selected": pick}
    cb = load(args.confirm)
    Fc = load_blob([args.confirm], "feats")
    cids = [i for i in cb["syco_baseline"] if i in Fc and all(i in cb[a] for a in arms)]
    base_c = [cb["syco_baseline"][i] for i in cids]
    sc = torch.stack([Fc[i] for i in cids]).numpy()[:, k] @ w + b
    yc = np.array([r["a1_correct"] for r in base_c], float)
    res["confirm"] = {"n": len(cids), "auroc_probe": auroc(sc, yc)}
    out_arms = {}
    for label, key in pick.items():
        if key is None:
            continue
        if label == "ungated":
            out_arms[label] = [cb[key[len("ungated_"):]][i] for i in cids]
            continue
        q, act = int(key.split("_")[0][1:]) / 100, key.split("_", 1)[1]
        f = sc > thr(q)
        out_arms[label] = override(base_c, f) if act == "override" else [cb[act][i] if g else x for i, x, g in zip(cids, base_c, f)]
        res["confirm"][f"{label}_fire"] = float(f.mean())
    for label, rows in out_arms.items():
        res["confirm"][label] = {**sel(base_c, rows), **boot(base_c, rows, out_arms["gated"]), "config": pick[label]}
        print(label, pick[label], {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res["confirm"][label].items()})
    q, act = int(pick["gated"].split("_")[0][1:]) / 100, pick["gated"].split("_", 1)[1]
    f = sc > thr(q)
    rand = [k for k in cb if k.startswith("syco_rand")]
    rs = {k: sel(base_c, [cb[k][i] if g else x for i, x, g in zip(cids, base_c, f)])["sel"] for k in rand if all(i in cb[k] for i in cids)}
    ung = sel(base_c, [cb[act][i] for i in cids])
    cav = np.array([b["state"] == CAVED for b in base_c])
    acc = np.array([b["state"] == ACCEPTED for b in base_c])
    tpr, fpr = f[cav].mean(), f[acc].mean()
    gated = res["confirm"]["gated"]["sel"]
    res["confirm"]["references"] = {
        "random_seeds": rs, "random_mean": float(np.mean(list(rs.values()))) if rs else None,
        "override_matched": float(sel(base_c, override(base_c, f))["sel"]),
        "eq2_predicted": float(fpr * ung["retention"] - tpr * (1 - ung["removal"])), "tpr": float(tpr), "fpr": float(fpr)}
    res["confirm"]["references"]["eq2_agrees"] = bool(
        np.sign(res["confirm"]["references"]["eq2_predicted"]) == np.sign(gated - res["confirm"]["references"]["override_matched"]))
    print("references", res["confirm"]["references"])
    print("probe", res["probe"], "confirm AUROC", round(res["confirm"]["auroc_probe"], 3), "n", len(cids))
    write_json(RESULTS_DIR / args.out, res)
