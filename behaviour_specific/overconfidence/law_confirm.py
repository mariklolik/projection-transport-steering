from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_pooled import pooled_rows
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.fit_detector import load_blob
from behaviour_specific.overconfidence.gate_law import law
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.steer_v2 import q_at
from behaviour_specific.overconfidence.steer_v7_tuned import parse
from general.paths import RESULTS_DIR
from general.storage import write_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="v4_confirm")
    ap.add_argument("--checks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checks = [c.split(":") for c in args.checks.split(",")]
    base, after = pooled_rows([args.dir], "m5", sorted({x for c in checks for x in (c[0], c[2])}))
    ids = [r["id"] for r in base]
    F, P = load_blob([args.dir], "feats"), load_blob([args.dir], "prompt")
    X, Q = torch.stack([F[i] for i in ids]), torch.stack([P[i] for i in ids])
    tstats = torch.load(DIRECTIONS_DIR / "projection_stats_L14.pt")["trace_stats"]["ocw_vs_cr"]
    u = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["ocw_vs_cr"].float()

    def flags(kind: str, cfg: str) -> np.ndarray:
        w = parse(cfg)
        if kind == "trace":
            d = torch.load(DIRECTIONS_DIR / "detector_m5.pt")
            s = X[:, LAYERS.index(d["layer"])] @ d["w"].float() + d["b"]
            return (s > q_at({"q": d["score_quantiles"].tolist()}, w["q"])).numpy()
        if kind == "prompt":
            d = torch.load(DIRECTIONS_DIR / "detector_m5_prompt.pt")
            s = Q[:, ("last", "mean").index(d["pos"]), LAYERS.index(d["layer"])] @ d["w"].float() + d["b"]
            return (s > q_at({"q": d["score_quantiles"].tolist()}, w["q"])).numpy()
        if kind == "castdim":
            d = torch.load(DIRECTIONS_DIR / f"cast_condition_L{w['layer']}.pt")
            z = Q[:, 1, LAYERS.index(w["layer"])]
            s = (z @ d["c"].float()) / (z.norm(dim=1) * d["c"].float().norm())
            return (s > q_at({"q": d["score_quantiles"].tolist()}, w["q"])).numpy()
        return (X[:, LAYERS.index(14)] @ u > q_at(tstats["confident_right"], w["cr_q"])).numpy()

    out = []
    for real, kind, ungated, cfg in checks:
        f = flags(kind, cfg)
        sim = law(base, after[ungated], f.astype(float), [0.5])
        rl = law(base, after[real], np.ones(len(base)), [0.0])
        row = {"arm": real, "decision": kind, "sel_real": rl["curve"][0]["sel_sim"], "sel_sim": sim["curve"][0]["sel_sim"],
               "sel_law": sim["curve"][0]["sel_law"], "tpr": sim["curve"][0]["tpr"], "fpr": sim["curve"][0]["fpr"],
               "rho_o": sim["rho_o"], "rho_c": sim["rho_c"]}
        out.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    write_json(RESULTS_DIR / args.out, {"checks": out})
