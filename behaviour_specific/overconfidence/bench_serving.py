# End-to-end serving cost, not just operator FLOPs: wall-clock for a whole
# 300-question workload under each intervention, including the extra forward
# pass and the regeneration that a post-hoc trace gate needs. The single-pass
# online gate is measured on the same workload and the same hardware.
# Run: python -m behaviour_specific.overconfidence.bench_serving

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_overconfidence import mean_activation_norm, score_m2_batch
from behaviour_specific.overconfidence.steer_v6_online import load_gate
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from general.online_gate import SequentialGate
from general.paths import RESULTS_DIR
from general.steering import (
    bw_map, steer_ablate, steer_add, steer_fullspace_affine, steer_perneuron_affine, steering_hook,
)
from general.storage import write_json
from models_specific.active import chat_prompt


def timed(fn) -> tuple[float, object]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _selftest():
    dt, out = timed(lambda: sum(range(1000)))
    assert dt >= 0 and out == 499500
    print("bench_serving self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--gate-layer", type=int, default=16)
    ap.add_argument("--decide-at", type=int, default=16, help="prefix budget of the sequential gate")
    ap.add_argument("--tau-q", type=float, default=0.30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="v3_serving")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    joint = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")
    pn = torch.load(DIRECTIONS_DIR / f"perneuron_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]

    model, tok = load_model()
    dev, d = model.device, fm["m_s"].shape[0]
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    u = pts["dirs"]["ocw_vs_cr"].float()
    V_c, w_c, b, tau_lda, sigma = load_gate(args.gate_layer, args.tau_q, 0.05, args.decide_at)
    V, w = V_c.to(dev, torch.float32), w_c.to(dev, torch.float32)

    records = load_records("mmlu", n=args.n, seed=args.seed)
    lids, bs = letter_token_ids(tok), args.batch_size
    run = lambda recs: score_m2_batch(model, tok, recs, lids, bs)
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)
    A = bw_map(fm["S_s"].double() + 1e-4 * torch.eye(d, dtype=torch.float64),
               fm["S_t"].double() + 1e-4 * torch.eye(d, dtype=torch.float64)).float().to(dev)

    def tokens(rows):
        return sum(r.get("reasoning_tokens", 0) or 0 for r in rows)

    print("warmup ...", flush=True)
    run(records[:bs])

    out = {"n": len(records), "batch_size": bs, "layer": args.layer, "methods": {}}
    t_base, rows_base = timed(lambda: run(records))
    out["methods"]["baseline"] = {"seconds": round(t_base, 2), "passes": 1, "tokens": tokens(rows_base)}
    print(f"baseline {t_base:.1f}s", flush=True)

    hooked = {
        "additive": lambda h: steer_add(h, v, -0.75 * norm),
        "ablation": lambda h: steer_ablate(h, v),
        "clamp_q50": conds["clamp_q50"](v),
        "quantile-OT": conds["otq_cal"](v),
        "MiMiC": lambda h: steer_fullspace_affine(h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)),
        "Linear-AcT": lambda h: steer_perneuron_affine(h, pn["mu_s"].to(dev), pn["sig_s"].to(dev),
                                                       pn["mu_t"].to(dev), pn["sig_t"].to(dev), 1.0),
    }
    for name, fn in hooked.items():
        def go(fn=fn):
            handle = steering_hook(model.model.layers[args.layer], fn)
            try:
                return run(records)
            finally:
                handle.remove()
        dt, rows = timed(go)
        out["methods"][name] = {"seconds": round(dt, 2), "passes": 1, "tokens": tokens(rows)}
        print(f"{name} {dt:.1f}s", flush=True)

    # two-pass post-hoc gate: generate, score the finished traces, regenerate the flagged ones
    def two_pass():
        t_gen, rows = timed(lambda: run(records))
        def gate_pass():
            sc = {}
            for r in rows:
                g = r["generations"][0]
                _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
                sc[r["id"]] = float((ht @ u).mean())
            return sc
        t_gate, scores = timed(gate_pass)
        tau = q_at(pstats["trace_stats"]["ocw_vs_cr"]["confident_right"], 0.50)
        flagged = [r for r in records if scores.get(r["id"], -1e9) > tau]
        handle = steering_hook(model.model.layers[args.layer], lambda h: steer_ablate(h, v))
        try:
            t_re, _ = timed(lambda: run(flagged) if flagged else [])
        finally:
            handle.remove()
        return t_gen, t_gate, t_re, len(flagged)

    tg, tgate, tre, nflag = two_pass()
    out["methods"]["gated ablation (two-pass)"] = {
        "seconds": round(tg + tgate + tre, 2), "passes": 2, "n_regenerated": nflag,
        "breakdown": {"generate": round(tg, 2), "gate_scoring": round(tgate, 2),
                      "regenerate": round(tre, 2)}}
    print(f"two-pass {tg + tgate + tre:.1f}s (regen {nflag})", flush=True)

    # single-pass online gate
    def online():
        gate = SequentialGate(V, w, b, tau_lda, sigma, delta=0.05, warmup=8,
                              decide_at=args.decide_at)
        h_act = steering_hook(model.model.layers[args.layer],
                              gate.actor(lambda h: steer_ablate(h, v)))
        h_read = steering_hook(model.model.layers[args.gate_layer], gate.reader)
        try:
            rows = run(records)
        finally:
            h_act.remove()
            h_read.remove()
        return rows, gate
    dt, (rows_on, gate) = timed(online)
    steps = [x for x in gate.fire_steps() if x is not None]
    out["methods"]["gated ablation (online, ours)"] = {
        "seconds": round(dt, 2), "passes": 1, "tokens": tokens(rows_on),
        "fire_rate": round(len(steps) / max(len(gate.fire_steps()), 1), 4),
        "fire_step_median": int(sorted(steps)[len(steps) // 2]) if steps else None}
    print(f"online {dt:.1f}s", flush=True)

    # a second unsteered pass at the end detects drift from anything else on the node
    t_base2, _ = timed(lambda: run(records))
    out["baseline_repeat_seconds"] = round(t_base2, 2)
    out["drift"] = round(abs(t_base2 - t_base) / t_base, 3)
    ref = min(t_base, t_base2)
    for m in out["methods"].values():
        m["latency_x_baseline"] = round(m["seconds"] / ref, 3)
    out["methods"]["baseline"]["seconds"] = round(ref, 2)
    write_json(RESULTS_DIR / args.outdir / "serving.json", out)
    print("done ->", RESULTS_DIR / args.outdir / "serving.json")
