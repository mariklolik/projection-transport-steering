# Aggregate the multi-seed MMLU runs: mean ± std over eval-subset seeds for the
# finalist conditions. Seeds are DATA-SUBSET seeds (greedy decoding is
# deterministic): robustness over question resampling, arguably stronger than
# sampling-seed robustness. Writes paper/gen/multiseed_table.tex.
#   python3 paper/make_multiseed.py

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = {"7": "steering_v2", "11": "steering_v2_s11", "23": "steering_v2_s23",
        "31": "steering_v2_s31", "47": "steering_v2_s47"}
CONDS = {
    "m4_conf_add_a-0.75": r"additive $-0.75$ \cite{rimsky2024caa}",
    "cast_gate-cr_q50_add-0.75": r"CAST-style gate$+$add \cite{lee2024cast}",
    "mimic_full": r"MiMiC full-space affine \cite{singh2024mimic}",
    "m4_conf_clamp_q50": r"clamp $q_{50}$ (ours)",
    "m4_conf_otq_nonocw": r"quantile-OT non-\ocw{} (ours)",
    "m4_conf_gate-ocw_vs_cr-cr_q50_clamp_q50": r"gated clamp (ours)",
    "sweep_ocwcr-crq50_ablate": r"gated ablation (ours)",
    "sweep_lda-allq50_ablate": r"LDA-gated ablation (ours)",
}


def collect(readout="m4"):
    per_cond = {k: {"sel": [], "dacc": [], "ece": [], "aurc": []} for k in CONDS}
    base = {"ece": [], "aurc": [], "acc": []}
    seeds_found = []
    for seed, d in DIRS.items():
        p = ROOT / "results" / d / "analysis_v2.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        seeds_found.append(seed)
        b = a["baselines"][readout]
        base["ece"].append(b["ece"])
        base["acc"].append(b["accuracy"])
        base["aurc"].append(b.get("selective", {}).get("aurc"))
        for cond in CONDS:
            tag = f"{cond}_{readout}"
            c = a["conditions"].get(tag)
            if c is None:
                continue
            per_cond[cond]["sel"].append(c["surgical"]["selectivity"] if c["surgical"] else None)
            per_cond[cond]["dacc"].append(c["deltas"]["d_acc"]["point"])
            per_cond[cond]["ece"].append(c["summary"]["ece"])
            per_cond[cond]["aurc"].append(c.get("selective", {}).get("aurc"))
    return seeds_found, base, per_cond


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"${vals[0]:+.3f}$"
    return f"${st.mean(vals):+.3f}\\pm{st.stdev(vals):.3f}$"


def pm(vals, signed=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "---"
    fmt = "+.3f" if signed else ".3f"
    if len(vals) == 1:
        return f"${vals[0]:{fmt}}$"
    return f"${st.mean(vals):{fmt}}\\pm{st.stdev(vals):.3f}$"


if __name__ == "__main__":
    seeds, base, per = collect("m4")
    lines = [f"% seeds: {','.join(seeds)} (n=300 each, M4 readout)"]
    lines.append(f"baseline & --- & {pm(base['ece'])} & {pm(base['aurc'])} & --- \\\\")
    for cond, pretty in CONDS.items():
        v = per[cond]
        lines.append(f"{pretty} & {pm(v['dacc'], signed=True)} & {pm(v['ece'])} & "
                     f"{pm(v['aurc'])} & {pm(v['sel'], signed=True)} \\\\")
    lines.append("\\hline")
    out = ROOT / "paper" / "gen" / "multiseed_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"{out} written; seeds found: {seeds}")
