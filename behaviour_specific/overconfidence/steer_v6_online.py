from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_overconfidence import (
    mean_activation_norm, score_m2_batch, score_m4_batch,
)
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from general.online_gate import SequentialGate
from general.paths import RESULTS_DIR
from general.steering import steer_ablate, steering_hook
from general.storage import write_json, write_jsonl
from models_specific.active import chat_prompt


def gate_sigma(cov: torch.Tensor, w: torch.Tensor) -> float:
    return float((w @ cov.float() @ w).clamp(min=1e-12).sqrt())


def load_gate(layer: int, tau_q: float, delta: float = 0.05, decide_at: int | None = None):
    fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{layer}.pt")
    w, b = fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
    tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, tau_q)
    cal_path = DIRECTIONS_DIR / f"gatecal_L{layer}.pt"
    sigma = gate_sigma(fit["S_tok"], w)
    if cal_path.exists():
        cal = torch.load(cal_path)
        key = min(cal["sigma_eff"], key=lambda k: abs(float(k) - delta))
        sigma = float(cal["sigma_eff"][key])
        if decide_at is not None:
            heads = cal["prefix_heads"]
            t = min(heads, key=lambda k: abs(int(k) - decide_at))
            w, b = heads[t]["w"].float(), float(heads[t]["b"])
            tau = q_at({"q": heads[t]["score_quantiles"]["all"].tolist()}, tau_q)
    elif decide_at is not None:
        raise SystemExit(f"no gatecal_L{layer}.pt — run calibrate_gate first")
    return fit["V"].float(), w, b, float(tau), sigma


def _selftest():
    S = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    assert abs(gate_sigma(S, torch.tensor([1.0, 0.0])) - 2.0) < 1e-5
    print("steer_v6_online self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14, help="action layer")
    ap.add_argument("--gate-layer", type=int, default=16, help="detection layer (>= action layer)")
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--tau-q", type=float, default=0.50, help="gate-score quantile on the extraction split")
    ap.add_argument("--deltas", default="0.05", help="comma-list of anytime-valid risk levels")
    ap.add_argument("--kappas", default="0", help="comma-list of soft-dose scales (0 = hard switch)")
    ap.add_argument("--actions", default="ablate,clamp_q50")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--decide-at", type=int, default=0,
                    help="commit once at this generated-token budget (0 = anytime boundary)")
    ap.add_argument("--read-prompt", action="store_true",
                    help="also accumulate the prompt positions during prefill")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="v3_mmlu_s7")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    assert args.gate_layer >= args.layer, "the reader must sit at or above the action layer"
    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    deltas = [float(x) for x in args.deltas.split(",")]
    at = args.decide_at or None
    V_c, w_c, b, tau, sigma = load_gate(args.gate_layer, args.tau_q, deltas[0], at)

    model, tok = load_model()
    dev = model.device
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    V, w = V_c.to(dev, torch.float32), w_c.to(dev, torch.float32)

    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    lids, ids4, bs = letter_token_ids(tok), yes_no_token_ids(tok), args.batch_size
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)
    actions = {"ablate": lambda h: steer_ablate(h, v)}
    for nm in ("clamp_q30", "clamp_q50"):
        if nm in conds:
            actions[nm] = conds[nm](v)

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    meta = {"config": vars(args), "tau": float(tau), "sigma": sigma, "runs": {}}
    print(f"online gate: read L{args.gate_layer}, act L{args.layer}, "
          f"tau={tau:.3f} sigma={sigma:.3f} warmup={args.warmup}", flush=True)

    for delta in deltas:
      _, w_d, b, tau, sigma = load_gate(args.gate_layer, args.tau_q, delta, at)
      w = w_d.to(dev, torch.float32)
      for kappa in [float(x) for x in args.kappas.split(",")]:
        for aname in args.actions.split(","):
            if aname not in actions:
                continue
            for mname, scorer in scorers.items():
                t0 = time.time()
                gate = SequentialGate(V, w, b, tau, sigma, delta=delta, warmup=args.warmup,
                                      kappa=kappa * sigma, read_prompt=args.read_prompt,
                                      decide_at=at)
                h_act = steering_hook(model.model.layers[args.layer], gate.actor(actions[aname]))
                h_read = steering_hook(model.model.layers[args.gate_layer], gate.reader)
                try:
                    rows = scorer(records)
                finally:
                    h_act.remove()
                    h_read.remove()
                steps = gate.fire_steps()
                fired = [x for x in steps if x is not None]
                ktag = "" if kappa == 0 else f"_k{str(kappa).replace('.', 'p')}"
                ktag += "_pp" if args.read_prompt else ""
                ktag += f"_t{at}" if at else ""
                tag = (f"online_d{str(delta).replace('.', 'p')}_q{int(args.tau_q * 100)}"
                       f"_L{args.gate_layer}{ktag}_{aname}_{mname}")
                write_jsonl(out_dir / f"{tag}__shard0.jsonl", rows)
                meta["runs"][tag] = {
                    "seconds": round(time.time() - t0, 1), "n": len(rows),
                    "n_sequences": len(steps), "fire_rate": round(len(fired) / max(len(steps), 1), 4),
                    "fire_step_median": int(sorted(fired)[len(fired) // 2]) if fired else None,
                    "fire_step_mean": round(sum(fired) / len(fired), 1) if fired else None}
                print(f"  {tag}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    write_json(RESULTS_DIR / args.outdir / f"meta_v6_{args.benchmark}_s{args.seed}.json", meta)
    print("done ->", out_dir)
