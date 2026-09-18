# NP-gate (LDA) x Bures-Wasserstein subspace transport (npbw conditions).
# self_qXX taus re-anchor the gate threshold to the eval domain's own score
# quantiles (label-free transfer).
# Run: python -m behaviour_specific.overconfidence.steer_v3_optimal [--benchmark ...]

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import q_at
from behaviour_specific.overconfidence.steer_v2_gated import gate_quality
from behaviour_specific.overconfidence.steer_overconfidence import score_m2_batch, score_m4_batch
from general.paths import RESULTS_DIR
from general.steering import bw_map, steer_subspace_affine, steering_hook
from general.storage import read_jsonl, write_json, write_jsonl

OCW = "overconfident_wrong"


def make_subspace_fn(V: torch.Tensor, m_s: torch.Tensor, S_s: torch.Tensor,
                     m_t: torch.Tensor, S_t: torch.Tensor):
    """h -> h with q = hVᵀ transported by the closed-form BW map N(m_s,S_s)→N(m_t,S_t)."""
    A = bw_map(S_s.double(), S_t.double()).float()
    def fn(h):
        return steer_subspace_affine(h, V.to(h.device, h.dtype), m_s.to(h.device, h.dtype),
                                     A.to(h.device, h.dtype), m_t.to(h.device, h.dtype))
    return fn


def _selftest():
    torch.manual_seed(0)
    V = torch.eye(2, 8)
    m_s, S_s = torch.tensor([2.0, -1.0]), torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    m_t, S_t = torch.zeros(2), torch.eye(2) * 0.25
    fn = make_subspace_fn(V, m_s, S_s, m_t, S_t)
    L = torch.linalg.cholesky(S_s)
    h = torch.zeros(4096, 8)
    h[:, :2] = torch.randn(4096, 2) @ L.T + m_s
    q = fn(h)[:, :2]
    assert q.mean(0).abs().max() < 0.05 and (torch.cov(q.T) - S_t).abs().max() < 0.05
    print("steer_v3_optimal self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--taus", default="all_q50,all_q70",
                    help="LDA-score thresholds: <group>_q<pct> over extraction-split scores")
    ap.add_argument("--targets", default="tgt_calibrated")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    js = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    V, tokstats, gate = js["V"].float(), js["token"], js["gate_lda"]
    w, b = gate["w"].float(), float(gate["b"])

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    baselines = {m: list(read_jsonl(out_dir / f"baseline_{m}__shard0.jsonl"))
                 for m in args.methods.split(",")}
    base_m2 = {r["id"]: r for r in read_jsonl(out_dir / "baseline_m2__shard0.jsonl")}

    model, tok = load_model()
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    lids = letter_token_ids(tok)
    ids4 = yes_no_token_ids(tok)
    bs = args.batch_size
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    print("gate pass: LDA scores from baseline traces", flush=True)
    gate_score = {}
    for i, (rid, r) in enumerate(base_m2.items()):
        g = r["generations"][0]
        _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        q = (ht @ V.T).mean(0)
        gate_score[rid] = float(q @ w + b)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(base_m2)}", flush=True)

    src = tokstats["src_all"]
    meta = {"config": vars(args), "gate_auroc_extraction": gate["auroc_ocw_vs_rest"], "gates": {}}

    for tgt_name in args.targets.split(","):
        tgt = tokstats[tgt_name]
        fn = make_subspace_fn(V, src["m"], src["S"], tgt["m"], tgt["S"])

        # ungated subspace transport (gate ablation)
        for mname, scorer in scorers.items():
            t0 = time.time()
            handle = steering_hook(model.model.layers[args.layer], fn)
            try:
                rows = scorer(records)
            finally:
                handle.remove()
            tag = f"subbw_{tgt_name}_{mname}"
            write_jsonl(out_dir / f"{tag}__shard0.jsonl", rows)
            print(f"  {tag}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

        for tau_spec in args.taus.split(","):
            grp, qq = tau_spec.rsplit("_q", 1)
            if grp == "self":
                # label-free recalibration: quantile of THIS benchmark's own gate
                # scores (no labels used) — the deployment-legal transfer variant
                # when the extraction-split threshold is off-distribution.
                sc = torch.tensor(sorted(gate_score.values()))
                tau = float(sc.quantile(float(qq) / 100))
            else:
                grp = {"all": "all", "cr": "cr", "nonocw": "non_ocw"}[grp]
                tau = q_at({"q": gate["score_quantiles"][grp].tolist()}, float(qq) / 100)
            flags = {rid: s > tau for rid, s in gate_score.items()}
            flagged = [by_id[rid] for rid, f in flags.items() if f and rid in by_id]
            gq = gate_quality(flags, baselines[list(scorers)[0]])
            meta["gates"][f"{tgt_name}|{tau_spec}"] = {"tau": tau, **gq}
            print(f"LDA gate>{tau_spec} (τ={tau:.2f}): {gq}", flush=True)

            for mname, scorer in scorers.items():
                t0 = time.time()
                if flagged:
                    handle = steering_hook(model.model.layers[args.layer], fn)
                    try:
                        steered_rows = scorer(flagged)
                    finally:
                        handle.remove()
                else:
                    steered_rows = []  # gate never fired (e.g. OOD threshold) -> condition == baseline
                sb = {r["id"]: r for r in steered_rows}
                merged = [sb.get(r["id"], r) for r in baselines[mname]]
                for row in merged:
                    row["gated"] = bool(flags.get(row["id"], False))
                tag = f"npbw_{tgt_name}-{tau_spec}_{mname}"
                write_jsonl(out_dir / f"{tag}__shard0.jsonl", merged)
                print(f"  {tag}: steered {len(flagged)}/{len(merged)} ({time.time() - t0:.0f}s)", flush=True)

    write_json(RESULTS_DIR / args.outdir / "meta_v3.json", meta)
    print("done ->", out_dir)
