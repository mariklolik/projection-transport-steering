# Dump behavioral-plane coordinates: per-benchmark baseline clouds and
# per-method measured before/after movements (figure data).
# Run: python -m behaviour_specific.overconfidence.proj_dump

from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from general.paths import RESULTS_DIR
from general.storage import read_jsonl

MOVES = {
    "ours_gatedabl": "sweep_ocwcr-crq50_ablate",
    "additive": "m4_conf_add_a-0.75",
    "cast": "cast_gate-cr_q50_add-0.75",
    "mimic": "mimic_full",
    "clamp": "m4_conf_clamp_q50",
}
BENCH_DIRS = {"mmlu": "steering_v2", "arc": "steering_v2_arc",
              "gsm8k": "steering_v2_gsm8k", "gpqa": "steering_v2_gpqa"}


def coords(model, tok, rec, layer, v, u):
    g = rec["generations"][0]
    _, ht = token_projections(model, tok, g["prompt"], g["text"], layer)
    return float((ht @ v).mean()), float((ht @ u).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    v, u = pts["dirs"]["m4_conf"].float(), pts["dirs"]["ocw_vs_cr"].float()
    model, tok = load_model()
    out = RESULTS_DIR / "viz"
    out.mkdir(parents=True, exist_ok=True)

    for bench, d in BENCH_DIRS.items():
        rd = RESULTS_DIR / d / "rollouts"
        if not (rd / "baseline_m2__shard0.jsonl").exists():
            print("skip", bench)
            continue
        base = read_jsonl(rd / "baseline_m2__shard0.jsonl")
        st4 = {r["id"]: r["state"] for r in read_jsonl(rd / "baseline_m4__shard0.jsonl")}
        with (out / f"plane_{bench}.csv").open("w") as f:
            f.write("x,y,state\n")
            for i, r in enumerate(base):
                x, y = coords(model, tok, r, args.layer, v, u)
                f.write(f"{x:.2f},{y:.2f},{st4.get(r['id'], 'na')}\n")
        print(f"plane_{bench}: {len(base)}", flush=True)

    rd = RESULTS_DIR / "steering_v2" / "rollouts"
    base = {r["id"]: r for r in read_jsonl(rd / "baseline_m2__shard0.jsonl")}
    st4 = {r["id"]: r["state"] for r in read_jsonl(rd / "baseline_m4__shard0.jsonl")}
    bcoords = {}
    for rid, r in base.items():
        bcoords[rid] = coords(model, tok, r, args.layer, v, u)
    print("mmlu baseline coords done", flush=True)
    for name, tag in MOVES.items():
        p = rd / f"{tag}_m2__shard0.jsonl"
        if not p.exists():
            print("skip", name)
            continue
        with (out / f"moves_{name}.csv").open("w") as f:
            f.write("x,y,u,v,state,changed\n")
            for r in read_jsonl(p):
                rid = r["id"]
                if rid not in bcoords:
                    continue
                x0, y0 = bcoords[rid]
                changed = r["generations"][0]["text"] != base[rid]["generations"][0]["text"]
                if changed:
                    x1, y1 = coords(model, tok, r, args.layer, v, u)
                else:
                    x1, y1 = x0, y0
                f.write(f"{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{st4.get(rid,'na')},{int(changed)}\n")
        print(f"moves_{name} done", flush=True)
    print("all done ->", out)
