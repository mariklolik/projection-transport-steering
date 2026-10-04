from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAIN_DIRS = {s: f"v3_mmlu_s{s}" for s in ("7", "11", "23", "31", "47")}
MAIN_CONDS = {
    "prompt_hedge": "prompting",
    "m4_conf_add_a-0.75": r"additive $-0.75$ \cite{rimsky2024caa}",
    "cast_gate-cr_q50_add-0.75": r"CAST-style gate$+$add \cite{lee2024cast}",
    "mimic_full": r"MiMiC full-space affine \cite{singh2024mimic}",
    "act_l05": r"Linear-AcT $\lambda{=}0.5$ \cite{rodriguez2025act}",
    "act_l10": r"Linear-AcT $\lambda{=}1.0$ \cite{rodriguez2025act}",
    "m4_conf_clamp_q50": r"clamp $q_{50}$ (ours)",
    "m4_conf_otq_nonocw": r"quantile-OT non-\ocw{} (ours)",
    "m4_conf_gate-ocw_vs_cr-cr_q50_clamp_q50": r"gated clamp (ours)",
    "sweep_ocwcr-crq50_ablate": r"gated ablation (ours)",
    "sweep_lda-allq50_ablate": r"LDA-gated ablation (ours)",
    "online_d0p05_q50_L16_ablate": r"online-gated ablation (ours, single pass)",
}
TUNED_CONDS = {
    "prompt_hedge": "prompting",
    "tuned_additive": r"additive CAA \cite{rimsky2024caa}",
    "tuned_cast": r"CAST-style \cite{lee2024cast}",
    "tuned_mimic": r"MiMiC \cite{singh2024mimic}",
    "tuned_act": r"Linear-AcT \cite{rodriguez2025act}",
    "tuned_ours": "gated local action (ours)",
    "tuned_gatedmimic": r"gated full-rank transport (ours)",
    "tuned_ourssoft": r"score-proportional dose (ours)",
    "tuned_online": "sequential gate (ours, single pass)",
}
ABLATION_CONDS = {
    "tuned_ours": "gated ablation, $k{=}2$",
    "tuned_oursk": r"gated transport, tuned $k$",
    "tuned_ourstok": "gated token-local clamp",
    "tuned_ourssoft": "score-proportional dose",
    "tuned_ourspost": "posterior-weighted dose",
    "tuned_gatedmimic": r"gated full-rank transport ($k{=}d$)",
}
NINEB_DIRS = {s: f"steering_9b_s{s}" for s in ("7", "11", "23")}
NINEB_CONDS = {
    "m4_conf_add_a-0.75": r"additive $-0.75$",
    "cast_gate-cr_q50_add-0.75": r"CAST-style gate$+$add",
    "m4_conf_clamp_q50": r"clamp $q_{50}$ (ours)",
    "sweep_ocwcr-crq50_ablate": r"gated ablation (ours)",
}


def collect(dirs: dict, conds: dict, readout: str = "m4"):
    per = {k: {"sel": [], "dacc": [], "ece": [], "aurc": [], "keep": []} for k in conds}
    base = {"ece": [], "aurc": [], "acc": []}
    seeds = []
    for seed, d in dirs.items():
        p = ROOT / "results" / d / "analysis_v2.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        seeds.append(seed)
        b = a["baselines"][readout]
        base["ece"].append(b["ece"])
        base["acc"].append(b["accuracy"])
        base["aurc"].append(b.get("selective", {}).get("aurc"))
        for cond in conds:
            c = a["conditions"].get(f"{cond}_{readout}")
            if c is None:
                continue
            surg = c.get("surgical")
            per[cond]["sel"].append(surg["selectivity"] if surg else None)
            per[cond]["keep"].append(surg["cr_retention"] if surg else None)
            per[cond]["dacc"].append(c["deltas"]["d_acc"]["point"])
            per[cond]["ece"].append(c["summary"]["ece"])
            per[cond]["aurc"].append(c.get("selective", {}).get("aurc"))
    return seeds, base, per


def pm(vals, signed=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "---"
    fmt = "+.3f" if signed else ".3f"
    if len(vals) == 1:
        return f"${vals[0]:{fmt}}$"
    return f"${st.mean(vals):{fmt}}\\pm{st.stdev(vals):.3f}$"


def write(out_name: str, dirs: dict, conds: dict, cols: tuple[str, ...]) -> None:
    seeds, base, per = collect(dirs, conds)
    lines = [f"% seeds: {','.join(seeds)} (n=300 each, M4 readout)"]
    cells = {"dacc": f"acc {pm(base['acc'])}", "ece": pm(base["ece"]),
             "aurc": pm(base["aurc"]), "sel": "---", "keep": "---"}
    lines.append("baseline & " + " & ".join(cells[c] for c in cols) + r" \\")
    for cond, pretty in conds.items():
        v = per[cond]
        if not any(v[c] for c in cols):
            continue
        lines.append(f"{pretty} & " + " & ".join(
            pm(v[c], signed=(c in ("dacc", "sel"))) for c in cols) + r" \\")
    (ROOT / "paper" / "gen" / out_name).write_text("\n".join(lines) + "\n\\hline\n")
    print(f"{out_name}: seeds {seeds}, {len(lines) - 1} rows")


if __name__ == "__main__":
    write("multiseed_table.tex", MAIN_DIRS, MAIN_CONDS, ("dacc", "ece", "aurc", "sel"))
    write("table_9b.tex", NINEB_DIRS, NINEB_CONDS, ("dacc", "ece", "keep", "aurc", "sel"))
    write("tuned_table.tex", MAIN_DIRS, TUNED_CONDS, ("dacc", "ece", "aurc", "keep", "sel"))
    write("ablation_table.tex", MAIN_DIRS, ABLATION_CONDS, ("dacc", "ece", "keep", "sel"))
