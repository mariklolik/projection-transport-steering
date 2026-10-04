from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR, analyse, pooled_rows
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.fit_detector import labelled, load_blob
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.select_configs import metrics
from general.metrics import ece
from general.paths import RESULTS_DIR
from general.storage import write_json

QS = (0.3, 0.4, 0.5, 0.6, 0.7)


def fit_dim(train: list[str]) -> torch.Tensor:
    X, y, m = labelled(train, train, "m5", "feats")
    Z = X[:, LAYERS.index(14)]
    return torch.tensor(Z[m & (y == 1)].mean(0) - Z[m & (y == 0)].mean(0))


def scorers(dirs: list[str], dim: torch.Tensor, train_dim_q: np.ndarray, cast_layer: int) -> dict[str, tuple]:
    F, P = load_blob(dirs, "feats"), load_blob(dirs, "prompt")
    out = {}
    d = torch.load(DIRECTIONS_DIR / "detector_m5.pt")
    out["probe, answer"] = (lambda i: float(F[i][LAYERS.index(d["layer"])] @ d["w"].float() + d["b"]), d["score_quantiles"].numpy())
    pd = torch.load(DIRECTIONS_DIR / "detector_m5_prompt.pt")
    k = ("last", "mean").index(pd["pos"])
    out["probe, prompt"] = (lambda i: float(P[i][k, LAYERS.index(pd["layer"])] @ pd["w"].float() + pd["b"]), pd["score_quantiles"].numpy())
    cd = torch.load(DIRECTIONS_DIR / f"cast_condition_L{cast_layer}.pt")
    c, kc = cd["c"].float(), LAYERS.index(cast_layer)
    out["CAST condition, prompt"] = (lambda i: float(P[i][1, kc] @ c / (P[i][1, kc].norm() * c.norm())),
                                     cd["score_quantiles"].numpy())
    u = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["ocw_vs_cr"].float()
    tq = torch.load(DIRECTIONS_DIR / "projection_stats_L14.pt")["trace_stats"]["ocw_vs_cr"]["confident_right"]["q"]
    out["direction, answer (400 labels)"] = (lambda i: float(F[i][LAYERS.index(14)] @ u), np.array(tq))
    out["direction, answer (4,400 labels)"] = (lambda i: float(F[i][LAYERS.index(14)] @ dim.float()), train_dim_q)
    return out


def override(base: list[dict], flags: np.ndarray) -> list[dict]:
    return [{**b, "state": {"overconfident_wrong": "nonconfident_wrong", "confident_right": "nonconfident_right"}.get(b["state"], b["state"]),
             "confidence": min(b["confidence"], 0.49)} if f else b for b, f in zip(base, flags)]


def calib(rows: list[dict]) -> dict:
    c, y = np.array([r["confidence"] for r in rows]), np.array([r["is_correct"] for r in rows], float)
    return {"ece": round(ece(c.tolist(), y.astype(bool).tolist()), 4), "brier": round(float(((c - y) ** 2).mean()), 4),
            "auroc_conf": round(auroc(c, y), 4), "acc": round(float(y.mean()), 4)}


def transitions(base: list[dict], after: list[dict]) -> dict:
    states = ("overconfident_wrong", "nonconfident_wrong", "nonconfident_right", "confident_right")
    return {s: {t: sum(b["state"] == s and a["state"] == t for b, a in zip(base, after)) for t in states} for s in states}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--real", default="")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cast-layer", type=int, default=14)
    args = ap.parse_args()

    actions = args.actions.split(",")
    real = [x for x in args.real.split(",") if x]
    dim = fit_dim(args.train.split(","))
    Ft = load_blob(args.train.split(","), "feats")
    dq = np.quantile([float(x[LAYERS.index(14)] @ dim.float()) for x in Ft.values()], np.linspace(0.01, 0.99, 41))
    res = {"tuning": {}, "confirm": {}}
    selected = {}
    tb, ta = pooled_rows([args.tuning], "m5", actions)
    sc_t = scorers([args.tuning], dim, dq, args.cast_layer)
    for name, (fn, qs) in sc_t.items():
        s = np.array([fn(r["id"]) for r in tb])
        grid = {}
        for q in QS:
            f = s > float(np.interp(q, np.linspace(0.01, 0.99, 41), qs))
            grid[f"q{int(100 * q)}_override"] = metrics(tb, override(tb, f))
            for a in actions:
                grid[f"q{int(100 * q)}_{a}"] = metrics(tb, [x if g else b for b, x, g in zip(tb, ta[a], f)])
        steer = {k: v for k, v in grid.items() if not k.endswith("override")}
        ok = [k for k, v in steer.items() if v["d_acc"] >= -0.01] or list(steer)
        best = max(ok, key=lambda k: steer[k]["selectivity"])
        best_o = max((k for k in grid if k.endswith("override")), key=lambda k: grid[k]["selectivity"])
        selected[name] = (best, best_o)
        res["tuning"][name] = {"grid": grid, "selected": best, "selected_override": best_o}
    cb, ca = pooled_rows([args.confirm], "m5", actions + real)
    sc_c = scorers([args.confirm], dim, dq, args.cast_layer)
    y = np.array([r["state"] == OCW for r in cb], float)
    msk = np.array([r["state"] in (OCW, CR) for r in cb])
    arms = {}
    for name, (fn, qs) in sc_c.items():
        s = np.array([fn(r["id"]) for r in cb])
        info = {"auroc_heldout": auroc(s[msk], y[msk])}
        for label, key in (("steer", selected[name][0]), ("override", selected[name][1])):
            q, act = int(key.split("_")[0][1:]) / 100, key.split("_", 1)[1]
            f = s > float(np.interp(q, np.linspace(0.01, 0.99, 41), qs))
            rows = override(cb, f) if act == "override" else [x if g else b for b, x, g in zip(cb, ca[act], f)]
            arms[f"{name} | {label}"] = rows
            info[label] = {"config": key, "fire": float(f.mean()), "tpr": float(f[y == 1].mean()),
                           "fpr": float(f[msk & (y == 0)].mean())}
        info["matrix"] = {}
        q = int(selected[name][0].split("_")[0][1:]) / 100
        f = s > float(np.interp(q, np.linspace(0.01, 0.99, 41), qs))
        for a in actions:
            info["matrix"][a] = metrics(cb, [x if g else b for b, x, g in zip(cb, ca[a], f)])
        res["confirm"][name] = info
    for r in real:
        arms[f"real | {r}"] = ca[r]
    for a in actions:
        arms[f"ungated | {a}"] = ca[a]
    stats = analyse(cb, arms, args.ref, 0.02, 10000, 0)
    res["arms"] = {k: {**stats["methods"][k], "calibration": calib(v), "transitions": transitions(cb, v)} for k, v in arms.items()}
    res["baseline_calibration"] = calib(cb)
    res["n"], res["n_ocw"], res["n_cr"] = stats["n"], stats["n_ocw"], stats["n_cr"]
    for k, v in res["arms"].items():
        vs = v.get("vs_ref", {}).get("sel", {})
        print(f"{k:55s} sel {v['sel']['point']:+.3f} {v['sel']['ci']} dacc {v['dacc']['point']:+.3f} "
              f"ece {v['calibration']['ece']:.3f} brier {v['calibration']['brier']:.3f} Δ {vs.get('point', 0):+.3f} p={vs.get('p_two_sided', '')}")
    for name, info in res["confirm"].items():
        print(name, "AUROC", round(info["auroc_heldout"], 3), "steer", info["steer"]["config"], "override", info["override"]["config"])
    print("baseline", res["baseline_calibration"])
    write_json(RESULTS_DIR / args.out, res)
