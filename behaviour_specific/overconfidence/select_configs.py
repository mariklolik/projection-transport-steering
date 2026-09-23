from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR
from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.fit_detector import load_blob
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.overconfidence.label_pool import LAYERS
from behaviour_specific.overconfidence.steer_v2 import q_at
from behaviour_specific.overconfidence.sweep_parity import pick
from general.paths import RESULTS_DIR
from general.storage import write_json

ALPHAS = ("-0.125", "-0.25", "-0.375", "-0.5", "-0.75", "-1.0", "-1.25", "-1.5")
QS = (0.3, 0.4, 0.5, 0.6)
GATED_SHIFTS = ("-0.25", "-0.375", "-0.5", "-0.75")
PTS_ACTIONS = ("ablate", "alpha-0.25", "alpha-0.375", "alpha-0.75")
FAMILIES = {
    "additive": [f"alpha{a}" for a in ALPHAS],
    "mimic": [f"L{L}_reg{r}" for L in (10, 14, 18) for r in ("0.0001", "0.01", "0.1")],
    "act": [f"L{L}_lam{x}" for L in (10, 14, 18) for x in (0.25, 0.5, 1.0)],
}


def metrics(base: list[dict], after: list[dict]) -> dict:
    ocw = [i for i, r in enumerate(base) if r["state"] == OCW]
    cr = [i for i, r in enumerate(base) if r["state"] == CR]
    rm = np.mean([after[i]["state"] != OCW for i in ocw])
    keep = np.mean([after[i]["state"] == CR for i in cr])
    dacc = np.mean([after[i]["is_correct"] for i in range(len(base))]) - np.mean([r["is_correct"] for r in base])
    return {"ocw_rm": float(rm), "cr_keep": float(keep), "selectivity": float(rm - (1 - keep)), "d_acc": float(dacc)}


def gated(base: list[dict], after: list[dict], flags: np.ndarray) -> list[dict]:
    return [a if f else b for b, a, f in zip(base, after, flags)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="v4_tuning")
    ap.add_argument("--readout", default="m5")
    ap.add_argument("--detector", default="detector_m5")
    ap.add_argument("--out", default="v4_parity_m5")
    args = ap.parse_args()

    runs = collect(RESULTS_DIR / args.dir / "rollouts")
    base = runs[f"baseline_{args.readout}"]
    ids = [r["id"] for r in base]
    runs = {k[:-len(args.readout) - 1]: {r["id"]: r for r in v} for k, v in runs.items() if k.endswith(f"_{args.readout}")}
    arm = lambda tag: [runs[tag][i] for i in ids]  # noqa: E731
    yb = np.array([r["state"] == OCW for r in base], float)
    mask = np.array([r["state"] in (OCW, CR) for r in base])
    feats, prompts = load_blob([args.dir], "feats"), load_blob([args.dir], "prompt")
    X = torch.stack([feats[i] for i in ids])
    u = torch.load(DIRECTIONS_DIR / "pts_L14.pt")["dirs"]["ocw_vs_cr"].float()
    tstats = torch.load(DIRECTIONS_DIR / "projection_stats_L14.pt")["trace_stats"]["ocw_vs_cr"]
    s_u = (X[:, LAYERS.index(14)] @ u).numpy()
    det = torch.load(DIRECTIONS_DIR / f"{args.detector}.pt")
    s_d = (X[:, LAYERS.index(det["layer"])] @ det["w"].float() + det["b"]).numpy()
    pdet = torch.load(DIRECTIONS_DIR / f"{args.detector}_prompt.pt")
    P = torch.stack([prompts[i] for i in ids])[:, ("last", "mean").index(pdet["pos"]), LAYERS.index(pdet["layer"])]
    s_p = (P @ pdet["w"].float() + pdet["b"]).numpy()
    tau = lambda d, q: q_at({"q": d["score_quantiles"].tolist()}, q)  # noqa: E731

    table = {m: {c: metrics(base, arm(f"plain_{c}")) for c in cfgs if f"plain_{c}" in runs}
             for m, cfgs in FAMILIES.items()}
    table["cast"] = {f"tau_cr_q{int(100 * q)}_alpha{a}": metrics(base, gated(base, arm(f"plain_alpha{a}"),
                                                                         s_u > q_at(tstats["confident_right"], q)))
                     for q in (0.3, 0.5, 0.7, 0.9) for a in GATED_SHIFTS}
    table["castprompt"] = {f"q{int(100 * q)}_alpha{a}": metrics(base, gated(base, arm(f"plain_alpha{a}"),
                                                                            s_p > tau(pdet, q)))
                           for q in QS for a in GATED_SHIFTS}
    table["pts"] = {f"q{int(100 * q)}_{act}": metrics(base, gated(base, arm(f"plain_{act}"), s_d > tau(det, q)))
                    for q in QS for act in PTS_ACTIONS}
    table["ungated"] = {c: metrics(base, arm(f"plain_{c}")) for c in ("ablate", "clamp_q30", "clamp_q50", "otq_cal")
                        if f"plain_{c}" in runs and len(runs[f"plain_{c}"]) == len(ids)}
    table["published_gate"] = {"u_crq50_ablate": metrics(base, gated(base, arm("plain_ablate"),
                                                                     s_u > q_at(tstats["confident_right"], 0.5)))}
    table["online"] = {k[len("detonline_"):]: metrics(base, arm(k)) for k in runs
                       if k.startswith("detonline_") and len(runs[k]) == len(ids)}
    pm = torch.stack([prompts[i] for i in ids])[:, 1]
    table["castdim"] = {}
    for L in (14, 16):
        cd = torch.load(DIRECTIONS_DIR / f"cast_condition_L{L}.pt")
        z = pm[:, LAYERS.index(L)]
        cos = ((z @ cd["c"].float()) / (z.norm(dim=1) * cd["c"].float().norm())).numpy()
        for q in (0.3, 0.5):
            for a in GATED_SHIFTS:
                table["castdim"][f"L{L}_q{int(100 * q)}_alpha{a}"] = metrics(
                    base, gated(base, arm(f"plain_alpha{a}"), cos > tau(cd, q)))
    aurocs = {name: auroc(s[mask], yb[mask]) for name, s in (("u", s_u), ("trace", s_d), ("prompt", s_p))}
    print("tuning-split AUROC", aurocs)
    winners = {m: pick(t) for m, t in table.items() if m != "ungated"}
    for m, t in table.items():
        print(f"== {m}" + (f"  -> {winners[m]}" if m in winners else ""))
        for c, v in sorted(t.items(), key=lambda kv: -kv[1]["selectivity"]):
            print(f"   {c:26s} sel {v['selectivity']:+.3f} rm {v['ocw_rm']:.3f} keep {v['cr_keep']:.3f} dacc {v['d_acc']:+.3f}")
    for m, wv in winners.items():
        write_json(RESULTS_DIR / args.out / f"parity_{m}.json", {"winner": wv, "results": table[m], "n": len(base)})
    write_json(RESULTS_DIR / args.out / "selection.json",
               {"n": len(base), "table": table, "winners": winners, "auroc_tuning": aurocs})
