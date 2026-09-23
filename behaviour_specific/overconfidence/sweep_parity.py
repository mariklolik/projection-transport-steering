# Equal-budget hyperparameter parity. Every method — ours included — gets the
# same number of configurations, searched on the same held-out tuning split
# (disjoint from the extraction split and from every evaluation subset) under
# one pre-declared objective: maximise selectivity subject to no accuracy loss.
# The winners are written out and then run, unchanged, on the evaluation seeds.
# Run: python -m behaviour_specific.overconfidence.sweep_parity --method cast

from __future__ import annotations

import argparse
import json
import time

import torch

from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import load_mmlu, mcq_prompt
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_overconfidence import mean_activation_norm, score_m4_batch
from general.paths import RESULTS_DIR
from behaviour_specific.overconfidence.steer_v3_optimal import make_subspace_fn
from general.behavioral_subspace import displacement_subspace, projected_moments
from general.steering import (
    bw_map, steer_ablate, steer_add, steer_fullspace_affine,
    steer_perneuron_affine, steering_hook,
)
from behaviour_specific.overconfidence.steer_v6_online import load_gate
from general.online_gate import SequentialGate, posterior_dose
from general.storage import write_json
from models_specific.active import chat_prompt

OCW = "overconfident_wrong"
CR = "confident_right"
EVAL_SEEDS = (7, 11, 23, 31, 47)


def tuning_records(n: int, seed: int, eval_n: int = 300, extract_n: int = 400,
                   extract_seed: int = 2) -> list[dict]:
    """`n` MMLU records disjoint from the extraction split and every eval subset."""
    used = {r["id"] for s in EVAL_SEEDS for r in load_mmlu(n=eval_n, seed=s)}
    used |= {r["id"] for r in load_mmlu(n=extract_n, seed=extract_seed)}
    return [r for r in load_mmlu(seed=seed) if r["id"] not in used][:n]


def surgical(before: list[dict], after: list[dict]) -> dict:
    """ocw-removal, cr-retention and their difference, paired by question id."""
    a = {r["id"]: r for r in after}
    ocw = [r for r in before if r["state"] == OCW]
    cr = [r for r in before if r["state"] == CR]
    rm = sum(a[r["id"]]["state"] != OCW for r in ocw) / max(len(ocw), 1)
    keep = sum(a[r["id"]]["state"] == CR for r in cr) / max(len(cr), 1)
    acc_b = sum(r["is_correct"] for r in before) / len(before)
    acc_a = sum(a[r["id"]]["is_correct"] for r in before) / len(before)
    return {"ocw_rm": round(rm, 4), "cr_keep": round(keep, 4),
            "selectivity": round(rm - (1 - keep), 4), "d_acc": round(acc_a - acc_b, 4),
            "n_ocw": len(ocw), "n_cr": len(cr)}


def pick(results: dict, acc_floor: float = -0.01) -> str:
    """Pre-declared rule: best selectivity among configs that keep accuracy."""
    ok = [k for k, v in results.items() if v["d_acc"] >= acc_floor] or list(results)
    return max(ok, key=lambda k: (results[k]["selectivity"], results[k]["d_acc"]))


def _selftest():
    b = [{"id": 1, "state": OCW, "is_correct": False}, {"id": 2, "state": CR, "is_correct": True}]
    a = [{"id": 1, "state": CR, "is_correct": True}, {"id": 2, "state": CR, "is_correct": True}]
    s = surgical(b, a)
    assert s["ocw_rm"] == 1.0 and s["cr_keep"] == 1.0 and s["selectivity"] == 1.0
    assert pick({"x": {"selectivity": .1, "d_acc": 0.0}, "y": {"selectivity": .3, "d_acc": -0.5}}) == "x"
    print("sweep_parity self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cast", help="additive | cast | mimic | act | ours")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--layers", default="10,14,18", help="layer grid for mimic/act")
    ap.add_argument("--gate-layer", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="v3_parity")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    records = tuning_records(args.n, args.seed)
    model, tok = load_model()
    dev = model.device
    ids4, bs = yes_no_token_ids(tok), args.batch_size
    run = lambda: score_m4_batch(model, tok, records, ids4, bs)

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    joint = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    tstats = pstats["trace_stats"]["ocw_vs_cr"]
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    u = pts["dirs"]["ocw_vs_cr"].float()
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)

    print(f"tuning split: {len(records)} records, ||h||={norm:.1f}", flush=True)
    base = run()
    print(f"baseline acc={sum(r['is_correct'] for r in base) / len(base):.3f} "
          f"ocw={sum(r['state'] == OCW for r in base)} cr={sum(r['state'] == CR for r in base)}",
          flush=True)

    def eval_hook_rows(fn, recs):
        """The steered rows for `recs`, keyed by id (empty dict for an empty subset)."""
        if not recs:
            return {}
        handle = steering_hook(model.model.layers[args.layer], fn)
        try:
            rows = score_m4_batch(model, tok, recs, ids4, bs)
        finally:
            handle.remove()
        return {r["id"]: r for r in rows}

    def eval_hook(fn, subset=None):
        recs = subset if subset is not None else records
        handle = steering_hook(model.model.layers[args.layer], fn)
        try:
            rows = score_m4_batch(model, tok, recs, ids4, bs) if recs else []
        finally:
            handle.remove()
        merged = {r["id"]: r for r in rows}
        return surgical(base, [merged.get(r["id"], r) for r in base])

    gate_scores = None

    def trace_gate(u_vec):
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
            sc[r["id"]] = float((ht @ u_vec).mean())
        return sc

    out, extra, t0 = {}, {}, time.time()
    if args.method == "additive":
        for a in (-0.125, -0.25, -0.375, -0.5, -0.75, -1.0, -1.25, -1.5):
            out[f"alpha{a}"] = eval_hook(lambda h, a=a: steer_add(h, v, a * norm))
    elif args.method == "cast":
        gate_scores = trace_gate(u)
        by_id = {r["id"]: r for r in records}
        for tq in (0.30, 0.50, 0.70, 0.90):
            tau = q_at(tstats["confident_right"], tq)
            flagged = [by_id[i] for i, s in gate_scores.items() if s > tau and i in by_id]
            for a in (-0.375, -0.75):
                out[f"tau_cr_q{int(tq * 100)}_alpha{a}"] = eval_hook(
                    lambda h, a=a: steer_add(h, v, a * norm), flagged)
    elif args.method in ("mimic", "act"):
        for L in [int(x) for x in args.layers.split(",")]:
            fmp = DIRECTIONS_DIR / f"fullspace_moments_L{L}.pt"
            pnp = DIRECTIONS_DIR / f"perneuron_stats_L{L}.pt"
            if not fmp.exists():
                print(f"  skip L{L}: no moments", flush=True)
                continue
            args.layer = L
            if args.method == "mimic":
                fm = torch.load(fmp)
                d = fm["m_s"].shape[0]
                for reg in (1e-4, 1e-2, 1e-1):
                    A = bw_map(fm["S_s"].double() + reg * torch.eye(d, dtype=torch.float64),
                               fm["S_t"].double() + reg * torch.eye(d, dtype=torch.float64)).float().to(dev)
                    out[f"L{L}_reg{reg}"] = eval_hook(
                        lambda h, A=A, fm=fm: steer_fullspace_affine(h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)))
            else:
                pn = torch.load(pnp)
                for lam in (0.25, 0.5, 1.0):
                    out[f"L{L}_lam{lam}"] = eval_hook(
                        lambda h, pn=pn, lam=lam: steer_perneuron_affine(
                            h, pn["mu_s"].to(dev), pn["sig_s"].to(dev),
                            pn["mu_t"].to(dev), pn["sig_t"].to(dev), lam))
        args.layer = 14
    elif args.method == "ours":
        # the analogue of the layer search MiMiC and Linear-AcT get: where the
        # gate reads, how often it fires, and which local action it composes with
        by_id = {r["id"]: r for r in records}
        for gl in (14, 16):
            fitp = DIRECTIONS_DIR / f"layerfit_L{gl}.pt"
            fit = torch.load(fitp) if fitp.exists() else None
            if fit is None:
                print(f"  skip gate layer {gl}: no layerfit", flush=True)
                continue
            Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
            sc = {}
            for r in base:
                g = r["generations"][0]
                _, ht = token_projections(model, tok, g["prompt"], g["text"], gl)
                sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
            qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
            jl = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
            tokstats = jl["token"]
            bw = make_subspace_fn(jl["V"].float().to(dev), tokstats["src_all"]["m"].to(dev),
                                  tokstats["src_all"]["S"].to(dev),
                                  tokstats["tgt_calibrated"]["m"].to(dev),
                                  tokstats["tgt_calibrated"]["S"].to(dev))
            for tq in (0.30, 0.50):
                tau = q_at({"q": qs}, tq)
                flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
                for aname, fn in (("ablate", lambda h: steer_ablate(h, v)), ("bw", bw)):
                    out[f"gl{gl}_q{int(tq * 100)}_{aname}"] = {
                        **eval_hook(fn, flagged), "n_flagged": len(flagged)}
    elif args.method == "ourscond":
        # Theorem 3 prescribes the map of the law conditioned on the gate
        # event; every gated transport we ran until now used the marginal
        # one. This measures the difference at four operating points
        by_id = {r["id"]: r for r in records}
        gm = torch.load(DIRECTIONS_DIR / f"gatedmoments_L{args.layer}.pt")
        Vc = gm["V"].float().to(dev)
        jl = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")["token"]
        fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.gate_layer)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
        marg = make_subspace_fn(Vc, jl["src_all"]["m"].to(dev), jl["src_all"]["S"].to(dev),
                                jl["tgt_calibrated"]["m"].to(dev), jl["tgt_calibrated"]["S"].to(dev))
        for key, cm in gm["cond"].items():
            tq = int(key[1:]) / 100
            tau = q_at({"q": qs}, tq)
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            cond = make_subspace_fn(Vc, cm["src"]["m"].to(dev), cm["src"]["S"].to(dev),
                                    cm["tgt"]["m"].to(dev), cm["tgt"]["S"].to(dev))
            for aname, fn in (("cond", cond), ("marg", marg)):
                out[f"gl{args.gate_layer}_{key}_{aname}"] = {
                    **eval_hook(fn, flagged), "n_flagged": len(flagged)}
    elif args.method == "ourstok":
        # the gate decides exposure at the trace level, the action stays
        # token-local: a quantile clamp only moves the positions actually in
        # the abnormal tail, so a permissive threshold costs little
        by_id = {r["id"]: r for r in records}
        fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.gate_layer)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
        for tq in (0.10, 0.20, 0.30, 0.50):
            tau = q_at({"q": qs}, tq)
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            for aname in ("clamp_q30", "clamp_q50"):
                out[f"gl{args.gate_layer}_q{int(tq * 100)}_{aname}"] = {
                    **eval_hook(conds[aname](v), flagged), "n_flagged": len(flagged)}
    elif args.method == "oursk":
        # Theorem 1 for a k-dimensional behavioural subspace: the tuned
        # dimension is k, the analogue of the layer a full-space map selects
        by_id = {r["id"]: r for r in records}
        fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.gate_layer)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
        fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")
        jl = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
        for k in (1, 2, 8, 32):
            Vk = displacement_subspace(fm["m_s"], fm["S_s"], fm["m_t"], fm["S_t"], k,
                                       seed=jl["V"].float()[:min(k, 2)])
            ms, Ss, mt, St = projected_moments(Vk, fm["m_s"], fm["S_s"], fm["m_t"], fm["S_t"])
            fn = make_subspace_fn(Vk.to(dev), ms.to(dev), Ss.to(dev), mt.to(dev), St.to(dev))
            for tq in (0.30, 0.50):
                tau = q_at({"q": qs}, tq)
                flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
                out[f"k{k}_q{int(tq * 100)}"] = {**eval_hook(fn, flagged), "n_flagged": len(flagged)}
    elif args.method == "ourssoft":
        # the gate decides exposure, but severity need not be all-or-nothing:
        # scaling the action by how far the score sits above tau spends the
        # edit where the evidence is, and spares the marginal firings
        by_id = {r["id"]: r for r in records}
        fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.gate_layer)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
        spread = float(torch.tensor(qs).std())
        for tq in (0.10, 0.30):
            tau = q_at({"q": qs}, tq)
            for kappa in (0.0, 0.5, 1.0, 2.0):
                lam = {i: 1.0 if kappa == 0 else min(max((x - tau) / (kappa * spread), 0.0), 1.0)
                       for i, x in sc.items() if x > tau}
                buckets: dict[float, list] = {}
                for i, L in lam.items():
                    if i in by_id and L > 0:
                        buckets.setdefault(round(min(4, max(1, round(L * 4))) / 4, 2), []).append(by_id[i])
                merged = {}
                for L, recs in buckets.items():
                    got = eval_hook_rows(lambda h, L=L: h + L * (steer_ablate(h, v) - h), recs)
                    merged.update(got)
                st = surgical(base, [merged.get(r["id"], r) for r in base])
                st["n_flagged"] = sum(len(x) for x in buckets.values())
                out[f"gl{args.gate_layer}_q{int(tq * 100)}_kappa{kappa}"] = st
    elif args.method == "ourspost":
        # Prop. 5: the dose that minimises displacement plus behavioural risk is
        # affine in the gate posterior, clipped to [0,1]. tau is measured, not
        # tuned: it is the collateral rate of the full dose over its removal
        # rate plus itself, so the only searched quantity is the slope rho.
        by_id = {r["id"]: r for r in records}
        cal = torch.load(DIRECTIONS_DIR / f"gatecal_L{args.gate_layer}.pt")
        fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{args.gate_layer}.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], args.gate_layer)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        full = eval_hook(lambda h: steer_ablate(h, v))
        b1, b0 = full["ocw_rm"], 1.0 - full["cr_keep"]
        tau = b0 / max(b1 + b0, 1e-9)
        extra["full_dose"] = full
        print(f"  full dose: removal {b1:.3f} collateral {b0:.3f} -> tau {tau:.3f}", flush=True)
        a_cal, c_cal = cal["platt"]["a"], cal["platt"]["c"]
        ids = list(sc)
        pi = torch.sigmoid(a_cal * torch.tensor([sc[i] for i in ids]) + c_cal)
        for rho in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 1e6):
            lam = posterior_dose(torch.tensor([sc[i] for i in ids]), a_cal, c_cal, tau, rho)
            buckets: dict[float, list] = {}
            for i, L in zip(ids, lam.tolist()):
                if i in by_id and L > 0:
                    buckets.setdefault(round(min(4, max(1, round(L * 4))) / 4, 2), []).append(by_id[i])
            merged = {}
            for L, recs in buckets.items():
                merged.update(eval_hook_rows(lambda h, L=L: h + L * (steer_ablate(h, v) - h), recs))
            st = surgical(base, [merged.get(r["id"], r) for r in base])
            st["n_flagged"] = sum(len(x) for x in buckets.values())
            st["mean_dose"] = round(float(lam.mean()), 4)
            out[f"rho{rho:g}"] = st
        extra["posterior"] = {"a": a_cal, "c": c_cal, "tau": round(tau, 4),
                              "mean_pi": round(float(pi.mean()), 4)}

    elif args.method == "gatedmimic":
        # Theorem 3 says the gate composes with ANY transport; this checks it on
        # the strongest unconditional baseline rather than on our local action
        by_id = {r["id"]: r for r in records}
        fit = torch.load(DIRECTIONS_DIR / "layerfit_L16.pt")
        Vg, wg, bg = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        sc = {}
        for r in base:
            g = r["generations"][0]
            _, ht = token_projections(model, tok, g["prompt"], g["text"], 16)
            sc[r["id"]] = float((ht @ Vg.T).mean(0) @ wg + bg)
        qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
        for L in (14, 18):
            fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{L}.pt")
            d = fm["m_s"].shape[0]
            A = bw_map(fm["S_s"].double() + 1e-2 * torch.eye(d, dtype=torch.float64),
                       fm["S_t"].double() + 1e-2 * torch.eye(d, dtype=torch.float64)).float().to(dev)
            args.layer = L
            for tq in (0.30, 0.50):
                tau = q_at({"q": qs}, tq)
                flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
                out[f"L{L}_q{int(tq * 100)}_mimic"] = {
                    **eval_hook(lambda h, A=A, fm=fm: steer_fullspace_affine(
                        h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)), flagged),
                    "n_flagged": len(flagged)}
        args.layer = 14
    elif args.method == "online":
        Vg = load_gate(args.gate_layer, 0.50)[0].to(dev, torch.float32)
        for tq in (0.30, 0.50):
            for at in (16, 32, 64, 128):
                _, w_g, b_g, tau_g, sigma = load_gate(args.gate_layer, tq, 0.05, at)
                wg = w_g.to(dev, torch.float32)
                gate = SequentialGate(Vg, wg, b_g, tau_g, sigma, warmup=8, decide_at=at)
                h_act = steering_hook(model.model.layers[args.layer],
                                      gate.actor(lambda h: steer_ablate(h, v)))
                h_read = steering_hook(model.model.layers[args.gate_layer], gate.reader)
                try:
                    rows = score_m4_batch(model, tok, records, ids4, bs)
                finally:
                    h_act.remove()
                    h_read.remove()
                merged = {r["id"]: r for r in rows}
                st = surgical(base, [merged.get(r["id"], r) for r in base])
                fired = [x for x in gate.fire_steps() if x is not None]
                st["fire_rate"] = round(len(fired) / max(len(gate.fire_steps()), 1), 4)
                out[f"q{int(tq * 100)}_at{at}"] = st
    else:
        raise SystemExit(f"unknown method {args.method}")

    for k, val in out.items():
        print(f"  {k:26s} {val}", flush=True)
    best = pick(out)
    write_json(RESULTS_DIR / args.outdir / f"parity_{args.method}.json",
               {"method": args.method, "n": len(records), "seed": args.seed,
                "batch_size": bs,
                "budget": len(out), "configs": out, "winner": best, **extra,
                "winner_stats": out[best], "seconds": round(time.time() - t0, 1)})
    print(f"WINNER {args.method}: {best} -> {out[best]}")
