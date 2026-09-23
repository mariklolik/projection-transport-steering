# Theorem 3 asks for the transport map of the law conditioned on the gate
# event, not of the marginal law. This dumps the projected token moments of
# the flagged traces and of the calibrated traces among them, one pair per
# gate quantile, so the gated operator can use the map it is entitled to.
# Run: python -m behaviour_specific.overconfidence.extract_gated_moments

from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.diag_layers import all_layer_trace
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import q_at
from general.storage import read_jsonl, write_json

CALIBRATED = {"confident_right", "nonconfident_wrong"}
QUANTILES = (0.10, 0.20, 0.30, 0.50)


def moments(x: torch.Tensor) -> dict:
    return {"m": x.mean(0), "S": torch.cov(x.T), "n": len(x)}


def _selftest():
    x = torch.randn(500, 2) * 3.0
    m = moments(x)
    assert m["m"].shape == (2,) and m["S"].shape == (2, 2) and m["n"] == 500
    print("extract_gated_moments self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14, help="action layer")
    ap.add_argument("--gate-layer", type=int, default=16)
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    rows = read_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.layer}.jsonl")
    fit_g = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
    Vg = fit_g["V"].float()
    wg, bg = fit_g["gate_lda"]["w"].float(), float(fit_g["gate_lda"]["b"])
    qs = fit_g["gate_lda"]["score_quantiles"]["all"].tolist()
    V = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")["V"].float()

    model, tok = load_model()
    scores, toks, states = [], [], []
    for i, r in enumerate(rows):
        per = all_layer_trace(model, tok, r["prompt"], r["trace"],
                              [args.layer, args.gate_layer])
        hg = per[args.gate_layer].float()
        scores.append(float((hg @ Vg.T).mean(0) @ wg + bg))
        toks.append((per[args.layer].float() @ V.T).half())
        states.append(r["m4_state"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    out, report = {}, {}
    for tq in QUANTILES:
        tau = q_at({"q": qs}, tq)
        flag = [i for i, s in enumerate(scores) if s > tau]
        cal = [i for i in flag if states[i] in CALIBRATED]
        if len(cal) < 2:
            print(f"  skip q{int(tq * 100)}: {len(cal)} calibrated in gate", flush=True)
            continue
        src = torch.cat([toks[i] for i in flag]).float()
        tgt = torch.cat([toks[i] for i in cal]).float()
        key = f"q{int(tq * 100)}"
        out[key] = {"tau": float(tau), "src": moments(src), "tgt": moments(tgt)}
        report[key] = {"tau": round(float(tau), 4), "n_flagged": len(flag),
                       "n_calibrated_in_gate": len(cal),
                       "src_tokens": len(src), "tgt_tokens": len(tgt)}
        print(f"  {key}: {len(flag)} traces flagged, {len(cal)} calibrated, "
              f"{len(src)} source tokens", flush=True)

    torch.save({"layer": args.layer, "gate_layer": args.gate_layer, "V": V, "cond": out},
               DIRECTIONS_DIR / f"gatedmoments_L{args.layer}.pt")
    write_json(DIRECTIONS_DIR / f"gatedmoments_L{args.layer}.json", report)
    print("saved ->", DIRECTIONS_DIR / f"gatedmoments_L{args.layer}.pt")
