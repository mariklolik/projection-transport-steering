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


def score(name: str, kind: str, X: np.ndarray) -> np.ndarray:
    d = torch.load(DIRECTIONS_DIR / f"{name}.pt")
    w, b = d["w"].numpy(), d["b"]
    if kind == "feats":
        return X[:, LAYERS.index(d["layer"])] @ w + b
    if kind == "prompt":
        return X[:, ("last", "mean").index(d["pos"]), LAYERS.index(d["layer"])] @ w + b
    return X[:, PREFIX_LAYERS.index(d["layer"]), PREFIX_T.index(d["t"])] @ w + b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = {"detector_m5": "feats", "detector_m5_prompt": "prompt",
             **{f"detector_m5_prefix_t{t}": "prefix" for t in PREFIX_T}}
    report = {}
    for name, kind in names.items():
        d = torch.load(DIRECTIONS_DIR / f"{name}.pt")
        row = {"layer": d["layer"], "pos": d.get("pos"), "t": d.get("t")}
        for split, dirs in (("train", args.train), ("select", args.select), ("heldout", args.heldout)):
            X, y, m = labelled(dirs.split(","), dirs.split(","), "m5", kind)
            row[f"auroc_{split}"] = round(auroc(score(name, kind, X)[m], y[m]), 4)
            row[f"n_{split}"] = int(m.sum())
        report[name] = row
        print(name, row, flush=True)
    write_json(RESULTS_DIR / args.out, report)
