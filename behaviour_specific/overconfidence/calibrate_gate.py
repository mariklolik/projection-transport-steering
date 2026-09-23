from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.extract_joint_stats import lda_fit
from behaviour_specific.overconfidence.extract_projection_stats import auroc, token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from general.online_gate import boundary, platt_fit
from general.storage import read_jsonl, write_json

GRID = (8, 16, 32, 64, 128, 256)
OCW = "overconfident_wrong"
QUANTILES = torch.linspace(0.01, 0.99, 41)


def fit_sigma(dev: dict[int, torch.Tensor], delta: float) -> float:
    need = []
    for t, d in dev.items():
        q = float(d.abs().quantile(1 - delta))
        unit = float(boundary(torch.tensor([float(t)]), 1.0, delta))
        need.append(q / max(unit, 1e-9))
    return max(need)


def _selftest():
    dev = {16: torch.zeros(10), 64: torch.zeros(10)}
    assert fit_sigma(dev, 0.05) == 0.0
    dev = {16: torch.full((10,), 1.0)}
    s = fit_sigma(dev, 0.05)
    assert abs(float(boundary(torch.tensor([16.0]), s, 0.05)) - 1.0) < 1e-3
    print("calibrate_gate self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-layer", type=int, default=16)
    ap.add_argument("--ref-layer", type=int, default=14, help="layer whose saved traces we reuse")
    ap.add_argument("--deltas", default="0.05,0.1,0.2")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
    V, w, b = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
    rows = read_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.ref_layer}.jsonl")
    model, tok = load_model()

    dev = {t: [] for t in GRID}
    prefix = {t: [] for t in GRID}
    lens, states, trace_scores = [], [], []
    for i, r in enumerate(rows):
        _, ht = token_projections(model, tok, r["prompt"], r["trace"], args.gate_layer)
        q = ht @ V.T
        s_tok = q @ w + b
        run = s_tok.cumsum(0) / torch.arange(1, len(s_tok) + 1)
        lens.append(len(s_tok))
        states.append(r["m4_state"])
        trace_scores.append(float(s_tok.mean()))
        for t in GRID:
            if len(s_tok) > t:
                dev[t].append(float(run[t - 1] - run[-1]))
            prefix[t].append(q[:t].mean(0))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    is_ocw = torch.tensor([st == OCW for st in states])
    is_cr = torch.tensor([st == "confident_right" for st in states])
    heads = {}
    for t in GRID:
        X = torch.stack(prefix[t])
        wt, bt = lda_fit(X[is_ocw], X[~is_ocw])
        sc = X @ wt + bt
        both = is_ocw | is_cr
        heads[t] = {"w": wt, "b": bt,
                    "auroc_ocw_vs_cr": round(auroc(sc[both].tolist(), is_ocw[both].int().tolist()), 4),
                    "score_quantiles": {"all": sc.quantile(QUANTILES)}}
        print(f"  prefix head t={t:3d}: AUROC(OCW|CR) = {heads[t]['auroc_ocw_vs_cr']:.3f}", flush=True)

    s_trace = torch.tensor(trace_scores)
    a_cal, c_cal = platt_fit(s_trace, is_ocw.int())
    print(f"  posterior calibration: P(OCW|s) = sigmoid({a_cal:.3f} s {c_cal:+.3f})", flush=True)

    dev = {t: torch.tensor(v) for t, v in dev.items() if len(v) >= 30}
    out = {"gate_layer": args.gate_layer, "n_traces": len(rows),
           "median_trace_len": int(sorted(lens)[len(lens) // 2]),
           "deviation_p95": {t: round(float(v.abs().quantile(0.95)), 4) for t, v in dev.items()},
           "sigma_eff": {},
           "prefix_auroc": {t: h["auroc_ocw_vs_cr"] for t, h in heads.items()},
           "platt": {"a": round(a_cal, 4), "c": round(c_cal, 4)}}
    for delta in [float(x) for x in args.deltas.split(",")]:
        out["sigma_eff"][str(delta)] = round(fit_sigma(dev, delta), 4)
    torch.save({**out, "prefix_heads": heads}, DIRECTIONS_DIR / f"gatecal_L{args.gate_layer}.pt")
    write_json(DIRECTIONS_DIR / f"gatecal_L{args.gate_layer}.json", out)
    print(out)
