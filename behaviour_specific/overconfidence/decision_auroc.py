from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.fit_detector import labelled
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.overconfidence.label_pool import LAYERS, PREFIX_LAYERS, PREFIX_T
from general.paths import RESULTS_DIR
from general.storage import write_json


def boot_ci(s: np.ndarray, y: np.ndarray, iters: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    vals = [auroc(s[i], y[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(iters))]
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="v4_confirm")
    ap.add_argument("--detector", default="detector_m5")
    ap.add_argument("--out", default="v4_pooled/decision_auroc_gemma.json")
    args = ap.parse_args()

    dirs = args.dirs.split(",")
    res = {}
    X, y, m = labelled(dirs, dirs, "m5", "prompt")
    d = torch.load(DIRECTIONS_DIR / f"{args.detector}_prompt.pt")
    s = X[:, ("last", "mean").index(d["pos"]), LAYERS.index(d["layer"])] @ d["w"].numpy() + d["b"]
    res["prompt"] = {"auroc": auroc(s[m], y[m]), "ci": boot_ci(s[m], y[m]), "n": int(m.sum())}
    X, y, m = labelled(dirs, dirs, "m5", "prefix")
    for ti, t in enumerate(PREFIX_T):
        d = torch.load(DIRECTIONS_DIR / f"{args.detector}_prefix_t{t}.pt")
        s = X[:, PREFIX_LAYERS.index(d["layer"]), ti] @ d["w"].numpy() + d["b"]
        res[f"prefix{t}"] = {"auroc": auroc(s[m], y[m]), "ci": boot_ci(s[m], y[m]), "n": int(m.sum())}
    X, y, m = labelled(dirs, dirs, "m5", "feats")
    d = torch.load(DIRECTIONS_DIR / f"{args.detector}.pt")
    s = X[:, LAYERS.index(d["layer"])] @ d["w"].numpy() + d["b"]
    res["trace"] = {"auroc": auroc(s[m], y[m]), "ci": boot_ci(s[m], y[m]), "n": int(m.sum())}
    u = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["ocw_vs_cr"].float().numpy()
    s = X[:, LAYERS.index(14)] @ u
    res["trace_u"] = {"auroc": auroc(s[m], y[m]), "ci": boot_ci(s[m], y[m]), "n": int(m.sum())}
    print(res)
    write_json(RESULTS_DIR / args.out, res)
