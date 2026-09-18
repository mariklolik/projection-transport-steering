# Two-pass conditional steering: gate on the model's own unsteered trace
# (projection onto a detection axis vs a calibration-split threshold), steer
# only flagged questions. No gold labels at inference.
# Run: python -m behaviour_specific.overconfidence.steer_v2_gated --taus cr_q50 \
#      --actions clamp_q50 [--benchmark ...]

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import load_mmlu
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_overconfidence import (
    mean_activation_norm, score_m2_batch, score_m4_batch,
)
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from general.paths import RESULTS_DIR
from general.steering import steering_hook
from general.storage import read_jsonl, write_json, write_jsonl
from models_specific.active import chat_prompt

OCW = "overconfident_wrong"


def gate_threshold(trace_stats: dict, spec: str) -> float:
    """'cr_q50' -> quantile 0.50 of confident_right trace-mean projections, etc."""
    state_key, q = spec.split("_q")
    state = {"cr": "confident_right", "ocw": OCW, "ncw": "nonconfident_wrong"}[state_key]
    return q_at(trace_stats[state], float(q) / 100)


def gate_quality(flags: dict[str, bool], baseline: list[dict]) -> dict:
    """How well the gate targets OCW on the eval set (diagnostic, not used to steer)."""
    st = {r["id"]: r["state"] for r in baseline}
    flagged = [i for i in flags if flags[i]]
    n_ocw = sum(st[i] == OCW for i in st)
    return {
        "n_flagged": len(flagged), "flag_rate": round(len(flagged) / len(flags), 4),
        "p_ocw_given_flagged": round(sum(st[i] == OCW for i in flagged) / len(flagged), 4) if flagged else None,
        "ocw_recall": round(sum(flags[i] for i in st if st[i] == OCW) / n_ocw, 4) if n_ocw else None,
        "cr_false_flag_rate": round(sum(flags[i] for i in st if st[i] == "confident_right")
                                    / max(1, sum(st[i] == "confident_right" for i in st)), 4),
    }


def _selftest():
    ts = {"confident_right": {"q": torch.linspace(0, 1, 41).tolist()}}
    assert abs(gate_threshold(ts, "cr_q50") - 0.5) < 1e-6
    flags = {"a": True, "b": False, "c": True}
    base = [{"id": "a", "state": OCW}, {"id": "b", "state": "confident_right"},
            {"id": "c", "state": "confident_right"}]
    gq = gate_quality(flags, base)
    assert gq["n_flagged"] == 2 and gq["ocw_recall"] == 1.0 and gq["p_ocw_given_flagged"] == 0.5
    assert gq["cr_false_flag_rate"] == 0.5
    print("steer_v2_gated self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--steer-direction", default="m4_conf")
    ap.add_argument("--gate", default="ocw_vs_cr", help="gate direction name in pts_L<layer>.pt")
    ap.add_argument("--taus", default="cr_q50,cr_q70")
    ap.add_argument("--actions", default="otq_cal,clamp_q50")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    if args.gate not in pts["dirs"]:
        raise SystemExit(f"gate direction {args.gate!r} not in pts_L{args.layer}.pt — rerun extraction")
    if "trace_stats" not in pstats:
        raise SystemExit("projection_stats has no trace_stats — rerun extraction")
    u_cpu = pts["dirs"][args.gate]
    v_cpu = pts["dirs"][args.steer_direction]
    stats = pstats["stats"][args.steer_direction]
    tstats = pstats["trace_stats"][args.gate]

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    baselines = {m: list(read_jsonl(out_dir / f"baseline_{m}__shard0.jsonl"))
                 for m in args.methods.split(",")}
    base_m2 = {r["id"]: r for r in read_jsonl(out_dir / "baseline_m2__shard0.jsonl")}

    from behaviour_specific.overconfidence.benchmarks import load_records

    model, tok = load_model()
    u = u_cpu.to(torch.float32)
    v = v_cpu.to(model.device, torch.float32)
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    lids = letter_token_ids(tok)
    ids4 = yes_no_token_ids(tok)
    bs = args.batch_size

    # pass 1: gate scores from the unsteered M2 traces (one forward per record)
    print("gate pass: projecting baseline traces onto", args.gate, flush=True)
    gate_score = {}
    for i, (rid, r) in enumerate(base_m2.items()):
        g = r["generations"][0]
        _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        gate_score[rid] = float((ht @ u).mean())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(base_m2)}", flush=True)

    norm = mean_activation_norm(model, tok, records, args.layer, lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    meta = {"config": vars(args), "gates": {}}
    for tau_spec in args.taus.split(","):
        tau = gate_threshold(tstats, tau_spec)
        flags = {rid: s > tau for rid, s in gate_score.items()}
        flagged_ids = [rid for rid, f in flags.items() if f]
        gq = gate_quality(flags, baselines[list(scorers)[0]])
        meta["gates"][tau_spec] = {"tau": tau, **gq}
        print(f"gate {args.gate}>{tau_spec} (τ={tau:.2f}): {gq}", flush=True)

        for action in args.actions.split(","):
            fn = conds[action](v)
            for mname, scorer in scorers.items():
                t0 = time.time()
                flagged_records = [by_id[rid] for rid in flagged_ids if rid in by_id]
                if flagged_records:
                    handle = steering_hook(model.model.layers[args.layer], fn)
                    try:
                        steered_rows = scorer(flagged_records)
                    finally:
                        handle.remove()
                else:
                    steered_rows = []  # gate never fired -> condition == baseline
                steered_by_id = {r["id"]: r for r in steered_rows}
                merged = [steered_by_id.get(r["id"], r) for r in baselines[mname]]
                for row in merged:
                    row_flag = flags.get(row["id"], False)
                    row["gated"] = bool(row_flag)
                tag = f"{args.steer_direction}_gate-{args.gate}-{tau_spec}_{action}_{mname}"
                write_jsonl(out_dir / f"{tag}__shard0.jsonl", merged)
                print(f"  {tag}: steered {len(flagged_records)}/{len(merged)} ({time.time() - t0:.0f}s)", flush=True)

    write_json(RESULTS_DIR / args.outdir / "meta_gated.json", meta)
    print("done ->", out_dir)
