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
from general.steering import (
    bw_map, steer_fullspace_affine, steer_perneuron_affine, steering_hook,
)
from general.storage import read_jsonl, write_json, write_jsonl
from models_specific.active import chat_prompt

OCW = "overconfident_wrong"
QUANTILES = torch.linspace(0.01, 0.99, 41)


HEDGE_SYSTEM = ("Be careful and well-calibrated. Only answer confidently when you are "
                "sure; if you are uncertain, explicitly acknowledge the uncertainty in "
                "your reasoning and confidence.")


def _selftest():
    torch.manual_seed(0)
    h = torch.randn(64, 8) * 2 + 1
    mu_s, sig_s = h.mean(0), h.std(0)
    hp = steer_perneuron_affine(h, mu_s, sig_s, torch.zeros(8), torch.ones(8), lam=1.0)
    assert hp.mean(0).abs().max() < 0.3 and (hp.std(0) - 1).abs().max() < 0.3
    hh = steer_perneuron_affine(h, mu_s, sig_s, torch.zeros(8), torch.ones(8), lam=0.0)
    assert torch.allclose(hh, h)
    S = torch.eye(8) * 4.0
    A = bw_map(S, torch.eye(8))
    hf = steer_fullspace_affine(h, h.mean(0), A, torch.zeros(8))
    assert hf.mean(0).abs().max() < 0.3
    print("steer_v4_sota self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--conditions", default="all",
                    help="all | comma-list of {cast_add,mimic,act_l05,act_l10,probegate}")
    ap.add_argument("--probe-taus", default="all_q60,all_q75")
    ap.add_argument("--probe-actions", default="clamp_q30,ablate,otq_cal")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    wanted = ["prompt_hedge", "cast_add", "mimic", "act_l05", "act_l10", "probegate"] \
        if args.conditions == "all" else args.conditions.split(",")

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    tstats = pstats["trace_stats"]["ocw_vs_cr"]

    model, tok = load_model()
    dev, dt = model.device, torch.float32
    v = pts["dirs"]["m4_conf"].to(dev, dt)
    u_gate = pts["dirs"]["ocw_vs_cr"].float()
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    lids, ids4, bs = letter_token_ids(tok), yes_no_token_ids(tok), args.batch_size
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    baselines = {m: list(read_jsonl(out_dir / f"baseline_{m}__shard0.jsonl"))
                 for m in scorers}
    base_m2 = {r["id"]: r for r in read_jsonl(out_dir / "baseline_m2__shard0.jsonl")}
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    meta = {"config": vars(args), "gates": {}}

    def run_hooked(fn, tag_stem):
        for mname, scorer in scorers.items():
            t0 = time.time()
            handle = steering_hook(model.model.layers[args.layer], fn)
            try:
                rows = scorer(records)
            finally:
                handle.remove()
            write_jsonl(out_dir / f"{tag_stem}_{mname}__shard0.jsonl", rows)
            print(f"  {tag_stem}_{mname}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    def run_gated(score_by_id, tau, fn, tag_stem):
        flags = {rid: s > tau for rid, s in score_by_id.items()}
        flagged = [by_id[rid] for rid, f in flags.items() if f and rid in by_id]
        gq = gate_quality(flags, baselines[list(scorers)[0]])
        meta["gates"][tag_stem] = {"tau": float(tau), **gq}
        print(f"  gate {tag_stem}: {gq}", flush=True)
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
            write_jsonl(out_dir / f"{tag_stem}_{mname}__shard0.jsonl", merged)
            print(f"  {tag_stem}_{mname}: steered {len(flagged)}/{len(merged)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if "prompt_hedge" in wanted:
        for mname in scorers:
            t0 = time.time()
            rows = (score_m2_batch(model, tok, records, lids, bs, system=HEDGE_SYSTEM)
                    if mname == "m2"
                    else score_m4_batch(model, tok, records, ids4, bs, system=HEDGE_SYSTEM))
            write_jsonl(out_dir / f"prompt_hedge_{mname}__shard0.jsonl", rows)
            print(f"  prompt_hedge_{mname}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    need_1d_gate = "cast_add" in wanted
    if need_1d_gate:
        print("gate pass (1-D ocw_vs_cr) ...", flush=True)
        gs_1d = {}
        for i, (rid, r) in enumerate(base_m2.items()):
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
            gs_1d[rid] = float((ht @ u_gate).mean())
            if (i + 1) % 150 == 0:
                print(f"  {i + 1}/{len(base_m2)}", flush=True)

    if "cast_add" in wanted:
        from general.steering import steer_add
        tau = q_at(tstats["confident_right"], 0.50)
        run_gated(gs_1d, tau, lambda h: steer_add(h, v, -0.75 * norm),
                  "cast_gate-cr_q50_add-0.75")

    if "mimic" in wanted:
        fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")
        A = bw_map(fm["S_s"].double() + 1e-4 * torch.eye(fm["S_s"].shape[0]),
                   fm["S_t"].double() + 1e-4 * torch.eye(fm["S_t"].shape[0])).float()
        m_s, m_t, A = fm["m_s"].to(dev, dt), fm["m_t"].to(dev, dt), A.to(dev, dt)
        run_hooked(lambda h: steer_fullspace_affine(h, m_s, A, m_t), "mimic_full")

    if "act_l05" in wanted or "act_l10" in wanted:
        pn = torch.load(DIRECTIONS_DIR / f"perneuron_stats_L{args.layer}.pt")
        mus, sgs = pn["mu_s"].to(dev, dt), pn["sig_s"].to(dev, dt)
        mut, sgt = pn["mu_t"].to(dev, dt), pn["sig_t"].to(dev, dt)
        for lam, name in ((0.5, "act_l05"), (1.0, "act_l10")):
            if name in wanted:
                run_hooked(lambda h, lam=lam: steer_perneuron_affine(h, mus, sgs, mut, sgt, lam),
                           name)

    if "probegate" in wanted:
        pg = torch.load(DIRECTIONS_DIR / f"probe_gate_L{args.layer}.pt")
        w, b = pg["w"].float(), float(pg["b"])
        print("gate pass (full-d probe) ...", flush=True)
        gs_probe = {}
        for i, (rid, r) in enumerate(base_m2.items()):
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
            gs_probe[rid] = float(ht.mean(0) @ w + b)
            if (i + 1) % 150 == 0:
                print(f"  {i + 1}/{len(base_m2)}", flush=True)
        conds = build_conditions(stats, norm)
        from general.steering import steer_ablate
        actions = {"ablate": lambda h: steer_ablate(h, v)}
        for name in ("clamp_q30", "clamp_q50", "otq_cal", "otq_nonocw"):
            if name in conds:
                actions[name] = conds[name](v)
        for tau_spec in args.probe_taus.split(","):
            grp, qq = tau_spec.rsplit("_q", 1)
            key = {"all": "all", "cr": "cr"}[grp]
            tau = q_at({"q": pg["score_quantiles"][key].tolist()}, float(qq) / 100)
            for act_name in args.probe_actions.split(","):
                run_gated(gs_probe, tau, actions[act_name],
                          f"probegate-{tau_spec}_{act_name}")

    write_json(RESULTS_DIR / args.outdir / f"meta_v4_{args.benchmark}_s{args.seed}.json", meta)
    print("done ->", out_dir)
