from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR, pooled_rows
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.steer_v2 import q_at
from general.paths import RESULTS_DIR
from general.storage import write_json


def load_feats(dirs: list[str]) -> dict[str, torch.Tensor]:
    out = {}
    for d in dirs:
        for f in sorted((RESULTS_DIR / d).glob("feats__shard*.pt")):
            blob = torch.load(f)
            out.update({i: x for i, x in zip(blob["ids"], blob["X"].float())})
    return out


def published_detectors() -> dict[str, tuple]:
    li = {L: k for k, L in enumerate(LAYERS)}
    u = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["ocw_vs_cr"].float()
    tstats = torch.load(DIRECTIONS_DIR / "projection_stats_L14.pt")["trace_stats"]["ocw_vs_cr"]
    dets = {"u@L14": (lambda x: float(x[li[14]] @ u), q_at(tstats["confident_right"], 0.50))}
    for L, name in ((14, "joint_stats_L14"), (16, "layerfit_L16")):
        if not (DIRECTIONS_DIR / f"{name}.pt").exists():
            continue
        fit = torch.load(DIRECTIONS_DIR / f"{name}.pt")
        V, w, b = fit["V"].float(), fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
        tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, 0.50)
        dets[f"lda@L{L}"] = (lambda x, V=V, w=w, b=b, k=li[L]: float((V @ x[k]) @ w + b), tau)
    return dets


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    r = s.argsort().argsort() + 1.0
    npos, nneg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def law(base, after, scores, taus) -> dict:
    ocw = np.array([r["state"] == OCW for r in base])
    cr = np.array([r["state"] == CR for r in base])
    left = np.array([r["state"] != OCW for r in after])
    lost = np.array([r["state"] != CR for r in after])
    rho_o, rho_c = left[ocw].mean(), lost[cr].mean()
    rows = []
    for tau in taus:
        f = scores > tau
        tpr, fpr = f[ocw].mean(), f[cr].mean()
        sim = (f & left)[ocw].mean() - (f & lost)[cr].mean()
        rows.append({"tau": float(tau), "fire": float(f.mean()), "tpr": float(tpr), "fpr": float(fpr),
                     "sel_sim": float(sim), "sel_law": float(tpr * rho_o - fpr * rho_c),
                     "rho_o_flag": float(left[ocw & f].mean()) if (ocw & f).any() else None,
                     "rho_c_flag": float(lost[cr & f].mean()) if (cr & f).any() else None})
    return {"rho_o": float(rho_o), "rho_c": float(rho_c), "curve": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=",".join(f"v3_mmlu_s{s}" for s in (7, 11, 23, 31, 47)))
    ap.add_argument("--feat-dirs", default=",".join(f"v4_eval{s}" for s in (7, 11, 23, 31, 47)))
    ap.add_argument("--actions", default="m4_conf_ablate,m4_conf_clamp_q50,m4_conf_clamp_q30,m4_conf_otq_cal,"
                                         "m4_conf_add_a-0.75,tuned_additive,mimic_full,act_l10")
    ap.add_argument("--checks", default="sweep_ocwcr-crq50_ablate:u@L14:m4_conf_ablate,"
                                        "sweep_ocwcr-crq50_clamp_q50:u@L14:m4_conf_clamp_q50,"
                                        "sweep_ocwcr-crq50_clamp_q30:u@L14:m4_conf_clamp_q30,"
                                        "sweep_lda-allq50_ablate:lda@L14:m4_conf_ablate,"
                                        "cast_gate-cr_q50_add-0.75:u@L14:m4_conf_add_a-0.75")
    ap.add_argument("--detectors")
    ap.add_argument("--readout", default="m4")
    ap.add_argument("--out", default="v4_pooled/gate_law.json")
    args = ap.parse_args()

    feats = load_feats(args.feat_dirs.split(","))
    dets = published_detectors()
    for name in filter(None, (args.detectors or "").split(",")):
        det = torch.load(DIRECTIONS_DIR / f"{name}.pt")
        rows = [LAYERS.index(L) for L in (det["layers"] if det["layer"] is None else [det["layer"]])]
        dets[name] = (lambda x, rows=rows, w=det["w"].float(), b=det["b"]: float(x[rows].flatten() @ w + b),
                      q_at({"q": det["score_quantiles"].tolist()}, 0.50))
    actions = args.actions.split(",")
    checks = [c.split(":") for c in args.checks.split(",") if c]
    conds = sorted(set(actions) | {c[0] for c in checks})
    base, after = pooled_rows(args.dirs.split(","), args.readout, conds)
    keep = [i for i, r in enumerate(base) if r["id"] in feats]
    base = [base[i] for i in keep]
    after = {c: [rows[i] for i in keep] for c, rows in after.items()}
    X = [feats[r["id"]] for r in base]
    y = np.array([r["state"] == OCW for r in base])
    mask = np.array([r["state"] in (OCW, CR) for r in base])
    res = {"n": len(base), "detectors": {}, "checks": []}
    for dn, (fn, tau) in dets.items():
        s = np.array([fn(x) for x in X])
        taus = np.quantile(s, np.linspace(0, 0.95, 20)).tolist() + [tau]
        res["detectors"][dn] = {"auroc": auroc(s[mask], y[mask].astype(float)), "tau_published": tau,
                                "actions": {a: law(base, after[a], s, taus) for a in actions}}
        print(f"{dn}: AUROC {res['detectors'][dn]['auroc']:.3f}", flush=True)
        for a in actions:
            pt = res["detectors"][dn]["actions"][a]["curve"][-1]
            best = max(res["detectors"][dn]["actions"][a]["curve"], key=lambda r: r["sel_sim"])
            print(f"   {a:22s} at τ_pub: tpr {pt['tpr']:.2f} fpr {pt['fpr']:.2f} sim {pt['sel_sim']:+.3f} "
                  f"law {pt['sel_law']:+.3f} | best sim {best['sel_sim']:+.3f} fire {best['fire']:.2f}", flush=True)
    for cond, dn, act in checks:
        fn, tau = dets[dn]
        s = np.array([fn(x) for x in X])
        sim = law(base, after[act], s, [tau])["curve"][0]
        real = law(base, after[cond], np.ones(len(base)), [0.0])["curve"][0]
        res["checks"].append({"cond": cond, "detector": dn, "action": act, "sel_sim": sim["sel_sim"],
                              "sel_law": sim["sel_law"], "sel_real": real["sel_sim"]})
        print(f"check {cond}: real {real['sel_sim']:+.3f} sim {sim['sel_sim']:+.3f} law {sim['sel_law']:+.3f}")
    write_json(RESULTS_DIR / args.out, res)
