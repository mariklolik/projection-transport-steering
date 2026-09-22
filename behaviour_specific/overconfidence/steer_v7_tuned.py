# Run, unchanged, the configuration each method won on the tuning split
# (sweep_parity) over an evaluation subset. This is what makes the head-to-head
# an equal-budget comparison rather than a comparison of defaults.
# Run: python -m behaviour_specific.overconfidence.steer_v7_tuned --seed 7 --outdir v3_mmlu_s7

from __future__ import annotations

import argparse
import json
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from behaviour_specific.overconfidence.steer_v2 import build_conditions, q_at
from behaviour_specific.overconfidence.steer_v3_optimal import make_subspace_fn
from behaviour_specific.overconfidence.steer_v6_online import load_gate
from general.behavioral_subspace import displacement_subspace, projected_moments
from behaviour_specific.overconfidence.steer_overconfidence import (
    mean_activation_norm, score_m2_batch, score_m4_batch,
)
from general.online_gate import SequentialGate, posterior_dose
from general.paths import RESULTS_DIR
from general.steering import (
    bw_map, steer_ablate, steer_add, steer_fullspace_affine, steer_perneuron_affine, steering_hook,
)
from general.storage import read_jsonl, write_json, write_jsonl
from models_specific.active import chat_prompt


def parse(winner: str) -> dict:
    """`gate_q30_ablate` / `alpha-0.75` / `tau_cr_q50_alpha-0.375` / `L14_lam0.5` -> fields.

    The action suffix is stripped first, so a `clamp_q50` action cannot be
    mistaken for the gate quantile that precedes it.
    """
    out = {}
    for action in ("clamp_q50", "clamp_q30", "ablate", "mimic"):
        if winner.endswith(action):
            out["action"] = action
            winner = winner[: -len(action)].rstrip("_")
            break
    for part in winner.split("_"):
        for key in ("alpha", "lam", "reg", "kappa", "rho", "at"):
            if part.startswith(key):
                out[key] = float(part[len(key):])
        if part.startswith("gl") and part[2:].isdigit():
            out["gate_layer"] = int(part[2:])
        if part.startswith("k") and part[1:].isdigit():
            out["k"] = int(part[1:])
        if part.startswith("q") and part[1:].isdigit() and "q" not in out:
            out["q"] = int(part[1:]) / 100
        if part.startswith("L") and part[1:].isdigit():
            out["layer"] = int(part[1:])
    if "cr" in winner.split("_"):
        out["cr_q"] = out.get("q")
    return out


def _selftest():
    assert parse("gate_q30_ablate") == {"q": 0.30, "action": "ablate"}
    assert parse("gate_q30_clamp_q50") == {"q": 0.30, "action": "clamp_q50"}
    assert parse("alpha-0.75")["alpha"] == -0.75
    assert parse("tau_cr_q50_alpha-0.375") == {"q": 0.5, "alpha": -0.375, "cr_q": 0.5}
    assert parse("L14_lam0.5") == {"lam": 0.5, "layer": 14}
    assert parse("L10_reg0.01") == {"reg": 0.01, "layer": 10}
    assert parse("q30_at64") == {"q": 0.30, "at": 64.0}
    assert parse("k8_q30") == {"k": 8, "q": 0.30}
    assert parse("gl16_q30_ablate") == {"gate_layer": 16, "q": 0.30, "action": "ablate"}
    assert parse("L18_q30_mimic") == {"layer": 18, "q": 0.30, "action": "mimic"}
    assert parse("gl16_q10_kappa1.0") == {"gate_layer": 16, "q": 0.10, "kappa": 1.0}
    assert parse("rho4") == {"rho": 4.0}
    print("steer_v7_tuned self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--gate-layer", type=int, default=16)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--parity-dir", default="v3_parity")
    ap.add_argument("--methods", default="additive,cast,mimic,act,ours,oursk,gatedmimic,ourssoft,ourspost,online")
    ap.add_argument("--readouts", default="m2,m4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="v3_mmlu_s7")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    winners = {}
    for m in args.methods.split(","):
        p = RESULTS_DIR / args.parity_dir / f"parity_{m}.json"
        if p.exists():
            winners[m] = json.loads(p.read_text())["winner"]
        else:
            print(f"  no tuning result for {m}, skipping", flush=True)
    print("tuned configurations:", winners, flush=True)

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    joint = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]
    tstats = pstats["trace_stats"]["ocw_vs_cr"]

    model, tok = load_model()
    dev = model.device
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    u = pts["dirs"]["ocw_vs_cr"].float()
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    lids, ids4, bs = letter_token_ids(tok), yes_no_token_ids(tok), args.batch_size
    scorers = {}
    for m in args.readouts.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    baselines = {m: list(read_jsonl(out_dir / f"baseline_{m}__shard0.jsonl")) for m in scorers}
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)
    meta = {"config": vars(args), "winners": winners, "gates": {}}

    def emit(tag: str, rows_by_id: dict, mname: str) -> None:
        merged = [rows_by_id.get(r["id"], r) for r in baselines[mname]]
        write_jsonl(out_dir / f"{tag}_{mname}__shard0.jsonl", merged)

    def run_all(tag: str, fn, subset=None):
        for mname, scorer in scorers.items():
            t0 = time.time()
            recs = subset if subset is not None else records
            if recs:
                handle = steering_hook(model.model.layers[args.layer], fn)
                try:
                    rows = scorer(recs)
                finally:
                    handle.remove()
            else:
                rows = []
            emit(tag, {r["id"]: r for r in rows}, mname)
            print(f"  {tag}_{mname}: {len(rows)}/{len(baselines[mname])} ({time.time() - t0:.0f}s)", flush=True)

    trace_scores = {}

    def gate_scores(vec):
        """Trace-level score of a single direction, read at the action layer."""
        key = ("dir", id(vec))
        if key not in trace_scores:
            trace_scores[key] = {
                r["id"]: float((token_projections(model, tok, r["generations"][0]["prompt"],
                                                 r["generations"][0]["text"], args.layer)[1] @ vec).mean())
                for r in baselines["m2"]}
        return trace_scores[key]

    def gate_scores_at(V, head, layer):
        """Trace-level LDA score on V-projections read at `layer`."""
        key = ("lda", layer)
        if key not in trace_scores:
            sc = {}
            for r in baselines["m2"]:
                g = r["generations"][0]
                _, ht = token_projections(model, tok, g["prompt"], g["text"], layer)
                sc[r["id"]] = float((ht @ V.T).mean(0) @ head[0] + head[1])
            trace_scores[key] = sc
        return trace_scores[key]

    for method, winner in winners.items():
        w = parse(winner)
        if method == "additive":
            run_all("tuned_additive", lambda h, a=w["alpha"]: steer_add(h, v, a * norm))
        elif method == "cast":
            sc = gate_scores(u)
            tau = q_at(tstats["confident_right"], w["cr_q"])
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            meta["gates"]["cast"] = {"tau": float(tau), "n_flagged": len(flagged)}
            run_all("tuned_cast", lambda h, a=w["alpha"]: steer_add(h, v, a * norm), flagged)
        elif method in ("mimic", "act"):
            L = w["layer"]
            if method == "mimic":
                fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{L}.pt")
                d = fm["m_s"].shape[0]
                reg = w.get("reg", 1e-4)
                A = bw_map(fm["S_s"].double() + reg * torch.eye(d, dtype=torch.float64),
                           fm["S_t"].double() + reg * torch.eye(d, dtype=torch.float64)).float().to(dev)
                args.layer = L
                run_all("tuned_mimic",
                        lambda h, A=A, fm=fm: steer_fullspace_affine(h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)))
            else:
                pn = torch.load(DIRECTIONS_DIR / f"perneuron_stats_L{L}.pt")
                args.layer = L
                run_all("tuned_act", lambda h, pn=pn, lam=w["lam"]: steer_perneuron_affine(
                    h, pn["mu_s"].to(dev), pn["sig_s"].to(dev), pn["mu_t"].to(dev), pn["sig_t"].to(dev), lam))
            args.layer = 14
        elif method in ("ours", "ourstok"):
            gl = w.get("gate_layer", args.gate_layer)
            fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{gl}.pt")
            head = (fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"]))
            sc = gate_scores_at(fit["V"].float(), head, gl)
            tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, w["q"])
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            meta["gates"][method] = {"tau": float(tau), "gate_layer": gl, "n_flagged": len(flagged)}
            fn = (lambda h: steer_ablate(h, v)) if w["action"] == "ablate" else conds[w["action"]](v)
            run_all(f"tuned_{method}", fn, flagged)
        elif method == "oursk":
            gl = w.get("gate_layer", args.gate_layer)
            fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{gl}.pt")
            sc = gate_scores_at(fit["V"].float(),
                                (fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])), gl)
            tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, w["q"])
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")
            Vk = displacement_subspace(fm["m_s"], fm["S_s"], fm["m_t"], fm["S_t"], w["k"],
                                       seed=joint["V"].float()[: min(w["k"], 2)])
            ms, Ss, mt, St = projected_moments(Vk, fm["m_s"], fm["S_s"], fm["m_t"], fm["S_t"])
            meta["gates"]["oursk"] = {"tau": float(tau), "k": w["k"], "n_flagged": len(flagged)}
            run_all("tuned_oursk", make_subspace_fn(Vk.to(dev), ms.to(dev), Ss.to(dev),
                                                    mt.to(dev), St.to(dev)), flagged)
        elif method == "ourssoft":
            gl = w.get("gate_layer", args.gate_layer)
            fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{gl}.pt")
            sc = gate_scores_at(fit["V"].float(),
                                (fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])), gl)
            qs = fit["gate_lda"]["score_quantiles"]["all"].tolist()
            tau, spread = q_at({"q": qs}, w["q"]), float(torch.tensor(qs).std())
            kappa = w.get("kappa", 0.0)
            buckets: dict[float, list] = {}
            for i, x in sc.items():
                if x <= tau or i not in by_id:
                    continue
                lam = 1.0 if kappa == 0 else min(max((x - tau) / (kappa * spread), 0.0), 1.0)
                if lam > 0:
                    buckets.setdefault(round(min(4, max(1, round(lam * 4))) / 4, 2), []).append(by_id[i])
            meta["gates"]["ourssoft"] = {"tau": float(tau), "kappa": kappa, "gate_layer": gl,
                                         "n_flagged": sum(len(x) for x in buckets.values()),
                                         "buckets": {str(k): len(x) for k, x in buckets.items()}}
            for mname, scorer in scorers.items():
                t0, merged = time.time(), {}
                for lam, recs in buckets.items():
                    handle = steering_hook(model.model.layers[args.layer],
                                           lambda h, L=lam: h + L * (steer_ablate(h, v) - h))
                    try:
                        merged.update({r["id"]: r for r in scorer(recs)})
                    finally:
                        handle.remove()
                emit("tuned_ourssoft", merged, mname)
                print(f"  tuned_ourssoft_{mname}: {len(merged)}/{len(baselines[mname])} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        elif method == "ourspost":
            gl = args.gate_layer
            cal = torch.load(DIRECTIONS_DIR / f"gatecal_L{gl}.pt")
            fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{gl}.pt")
            sc = gate_scores_at(fit["V"].float(),
                                (fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])), gl)
            par = json.loads((RESULTS_DIR / args.parity_dir / "parity_ourspost.json").read_text())
            a_cal, c_cal = par["posterior"]["a"], par["posterior"]["c"]
            ids = [i for i in sc if i in by_id]
            lam = posterior_dose(torch.tensor([sc[i] for i in ids]), a_cal, c_cal,
                                 par["posterior"]["tau"], w["rho"])
            buckets: dict[float, list] = {}
            for i, L in zip(ids, lam.tolist()):
                if L > 0:
                    buckets.setdefault(round(min(4, max(1, round(L * 4))) / 4, 2), []).append(by_id[i])
            meta["gates"]["ourspost"] = {"rho": w["rho"], "tau": par["posterior"]["tau"],
                                         "gate_layer": gl, "mean_dose": round(float(lam.mean()), 4),
                                         "n_flagged": sum(len(x) for x in buckets.values()),
                                         "buckets": {str(k): len(x) for k, x in buckets.items()}}
            for mname, scorer in scorers.items():
                t0, merged = time.time(), {}
                for lv, recs in buckets.items():
                    handle = steering_hook(model.model.layers[args.layer],
                                           lambda h, L=lv: h + L * (steer_ablate(h, v) - h))
                    try:
                        merged.update({r["id"]: r for r in scorer(recs)})
                    finally:
                        handle.remove()
                emit("tuned_ourspost", merged, mname)
                print(f"  tuned_ourspost_{mname}: {len(merged)}/{len(baselines[mname])} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        elif method == "gatedmimic":
            gl = w.get("gate_layer", args.gate_layer)
            fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{gl}.pt")
            sc = gate_scores_at(fit["V"].float(),
                                (fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])), gl)
            tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, w["q"])
            flagged = [by_id[i] for i, x in sc.items() if x > tau and i in by_id]
            L = w["layer"]
            fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{L}.pt")
            d = fm["m_s"].shape[0]
            A = bw_map(fm["S_s"].double() + 1e-2 * torch.eye(d, dtype=torch.float64),
                       fm["S_t"].double() + 1e-2 * torch.eye(d, dtype=torch.float64)).float().to(dev)
            meta["gates"]["gatedmimic"] = {"tau": float(tau), "layer": L, "n_flagged": len(flagged)}
            args.layer = L
            run_all("tuned_gatedmimic",
                    lambda h, A=A, fm=fm: steer_fullspace_affine(h, fm["m_s"].to(dev), A, fm["m_t"].to(dev)),
                    flagged)
            args.layer = 14
        elif method == "online":
            V_c, w_c, b_g, tau_g, sigma = load_gate(args.gate_layer, w["q"], 0.05, int(w["at"]))
            gate = SequentialGate(V_c.to(dev, torch.float32), w_c.to(dev, torch.float32), b_g, tau_g,
                                  sigma, warmup=8, decide_at=int(w["at"]))
            for mname, scorer in scorers.items():
                t0 = time.time()
                g = SequentialGate(V_c.to(dev, torch.float32), w_c.to(dev, torch.float32), b_g, tau_g,
                                   sigma, warmup=8, decide_at=int(w["at"]))
                ha = steering_hook(model.model.layers[args.layer], g.actor(lambda h: steer_ablate(h, v)))
                hr = steering_hook(model.model.layers[args.gate_layer], g.reader)
                try:
                    rows = scorer(records)
                finally:
                    ha.remove()
                    hr.remove()
                steps = [x for x in g.fire_steps() if x is not None]
                meta["gates"][f"online_{mname}"] = {
                    "tau": float(tau_g), "decide_at": int(w["at"]),
                    "fire_rate": round(len(steps) / max(len(g.fire_steps()), 1), 4)}
                emit("tuned_online", {r["id"]: r for r in rows}, mname)
                print(f"  tuned_online_{mname}: {len(rows)} ({time.time() - t0:.0f}s) "
                      f"{meta['gates'][f'online_{mname}']}", flush=True)

    write_json(RESULTS_DIR / args.outdir / f"meta_v7_{args.benchmark}_s{args.seed}.json", meta)
    print("done ->", out_dir)
