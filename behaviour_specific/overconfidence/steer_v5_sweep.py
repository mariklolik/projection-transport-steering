# Strong-gate x strong-action sweep: gates {ocw_vs_cr@cr_q50, LDA@all_q50} x
# actions {ablate, clamp_q30, clamp_q50}. Produces the headline gated-ablation
# condition.
# Run: python -m behaviour_specific.overconfidence.steer_v5_sweep [--benchmark ...]

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_v2_gated import gate_quality
from behaviour_specific.overconfidence.steer_overconfidence import (
    mean_activation_norm, score_m2_batch, score_m4_batch,
)
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from general.paths import RESULTS_DIR
from general.steering import steer_ablate, steering_hook
from general.storage import read_jsonl, write_json, write_jsonl
from models_specific.active import chat_prompt


def _selftest():
    v = torch.zeros(6); v[0] = 1.0
    h = torch.randn(4, 6)
    assert steer_ablate(h, v)[:, 0].abs().max() < 1e-5
    print("steer_v5_sweep self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--gates", default="ocwcr_crq50,lda_allq50")
    ap.add_argument("--actions", default="ablate,clamp_q30,clamp_q50")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    js = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    tstats = pstats["trace_stats"]["ocw_vs_cr"]

    model, tok = load_model()
    v = pts["dirs"]["m4_conf"].to(model.device, torch.float32)
    u1 = pts["dirs"]["ocw_vs_cr"].float()
    V2, wl, bl = js["V"].float(), js["gate_lda"]["w"].float(), float(js["gate_lda"]["b"])
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    lids, ids4, bs = letter_token_ids(tok), yes_no_token_ids(tok), args.batch_size
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    baselines = {m: list(read_jsonl(out_dir / f"baseline_{m}__shard0.jsonl")) for m in scorers}
    base_m2 = {r["id"]: r for r in read_jsonl(out_dir / "baseline_m2__shard0.jsonl")}
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))

    # one activation pass -> BOTH gate scores per question
    print("gate pass (shared) ...", flush=True)
    gs1, gs2 = {}, {}
    for i, (rid, r) in enumerate(base_m2.items()):
        g = r["generations"][0]
        _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        gs1[rid] = float((ht @ u1).mean())
        gs2[rid] = float((ht @ V2.T).mean(0) @ wl + bl)
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(base_m2)}", flush=True)

    gates = {}
    if "ocwcr_crq50" in args.gates:
        gates["ocwcr-crq50"] = (gs1, q_at(tstats["confident_right"], 0.50))
    if "lda_allq50" in args.gates:
        gates["lda-allq50"] = (gs2, q_at({"q": js["gate_lda"]["score_quantiles"]["all"].tolist()}, 0.50))

    conds = build_conditions(stats, norm)
    actions = {"ablate": lambda h: steer_ablate(h, v)}
    for nm in ("clamp_q30", "clamp_q50"):
        if nm in args.actions.split(",") and nm in conds:
            actions[nm] = conds[nm](v)

    meta = {"config": vars(args), "gates": {}}
    for gname, (scores, tau) in gates.items():
        flags = {rid: s > tau for rid, s in scores.items()}
        flagged = [by_id[rid] for rid, f in flags.items() if f and rid in by_id]
        gq = gate_quality(flags, baselines[list(scorers)[0]])
        meta["gates"][gname] = {"tau": float(tau), **gq}
        print(f"gate {gname} (τ={tau:.2f}): {gq}", flush=True)
        for aname in args.actions.split(","):
            if aname not in actions:
                continue
            fn = actions[aname]
            for mname, scorer in scorers.items():
                t0 = time.time()
                if flagged:
                    handle = steering_hook(model.model.layers[args.layer], fn)
                    try:
                        srows = scorer(flagged)
                    finally:
                        handle.remove()
                else:
                    srows = []
                sb = {r["id"]: r for r in srows}
                merged = [sb.get(r["id"], r) for r in baselines[mname]]
                tag = f"sweep_{gname}_{aname}_{mname}"
                write_jsonl(out_dir / f"{tag}__shard0.jsonl", merged)
                print(f"  {tag}: steered {len(flagged)}/{len(merged)} ({time.time() - t0:.0f}s)", flush=True)

    write_json(RESULTS_DIR / args.outdir / f"meta_v5_{args.benchmark}_s{args.seed}.json", meta)
    print("done ->", out_dir)
