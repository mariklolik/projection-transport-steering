# Generate the paper's LaTeX table fragments + figure CSVs from analysis_v2.json
# files (local copies under results/). Run after every analysis refresh:
#   python3 paper/make_tables.py
# Writes paper/gen/*.tex and paper/pic/*.csv. No torch needed (pure json).

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"
GEN.mkdir(exist_ok=True)

PRETTY = {
    "add_a-0.75": r"additive $-0.75$",
    "ablate": "ablation",
    "clamp_q30": r"clamp $q_{30}$",
    "clamp_q50": r"clamp $q_{50}$",
    "clamp_q70": r"clamp $q_{70}$",
    "otg_cal": "Gaussian-OT cal.",
    "otq_cal": "quantile-OT cal.",
    "otq_nonocw": r"quantile-OT non-\ocw{}",
    "gate-ocw_vs_cr-cr_q50_clamp_q50": r"gated clamp $q_{50}$",
    "gate-ocw_vs_cr-cr_q70_clamp_q50": r"gated clamp $q_{70}$",
    "gate-ocw_vs_cr-cr_q50_otq_cal": r"gated OT $q_{50}$",
    "gate-ocw_vs_cr-cr_q70_otq_cal": r"gated OT $q_{70}$",
}


WHOLE = {
    "sweep_ocwcr-crq50_ablate": "gated ablation",
    "sweep_ocwcr-crq50_clamp_q50": r"gated clamp $q_{50}$ (sweep)",
    "sweep_lda-allq50_ablate": "LDA-gated ablation",
    "online_d0p05_q50_L16_ablate": "sequential-gate ablation",
    "online_d0p05_q50_L16_clamp_q50": r"sequential-gate clamp $q_{50}$",
    "online_d0p05_q50_L16_k2p0_ablate": r"sequential-gate ablation, soft dose",
    "cast_gate-cr_q50_add-0.75": "CAST-style gate$+$add",
    "mimic_full": "MiMiC full-space affine",
    "act_l05": r"Linear-AcT $\lambda{=}0.5$",
    "act_l10": r"Linear-AcT $\lambda{=}1.0$",
    "prompt_hedge": "prompting",
    "tuned_additive": "additive CAA (tuned)",
    "tuned_cast": "CAST-style (tuned)",
    "tuned_mimic": "MiMiC (tuned)",
    "tuned_act": "Linear-AcT (tuned)",
    "tuned_ours": "gated local action (tuned, ours)",
    "tuned_oursk": r"gated $k$-dim transport (tuned, ours)",
    "tuned_gatedmimic": "gated full-rank transport (tuned, ours)",
    "tuned_ourssoft": "score-proportional dose (tuned, ours)",
    "tuned_ourstok": "gated token-local clamp (tuned, ours)",
    "tuned_online": "sequential gate (tuned, ours)",
}


def norm_name(tag: str) -> str:
    if tag in WHOLE:
        return WHOLE[tag]
    for p in ("m4_conf_", "subbw_", "npbw_"):
        if tag.startswith(p):
            core = tag[len(p):]
            if p == "subbw_":
                return "subspace-BW " + core.replace("tgt_", "").replace("_", r"\_")
            if p == "npbw_":
                core = core.replace("tgt_calibrated-", "")
                return r"NP-gate$\times$BW " + core.replace("_", r"\_")
            return PRETTY.get(core, core.replace("_", r"\_"))
    return tag.replace("_", r"\_")


def load(dirname: str) -> dict | None:
    p = ROOT / "results" / dirname / "analysis_v2.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ci(d: dict) -> str:
    return f"${d['point']:+.3f}\\,[{d['lo']:+.3f},{d['hi']:+.3f}]$"


def rows_for(analysis: dict, readout: str = "m4"):
    base = analysis["baselines"][readout]
    out = [("baseline", "---", f"{base['ece']:.3f}",
            str(base["states"].get("overconfident_wrong", 0)), "---", "---")]
    for tag in sorted(analysis["conditions"]):
        if not tag.endswith(f"_{readout}"):
            continue
        c = analysis["conditions"][tag]
        s, d, surg = c["summary"], c["deltas"], c["surgical"]
        sel = "---"
        if surg is not None:
            lo, hi = c.get("selectivity_ci", ["", ""])
            sel = f"${surg['selectivity']:+.3f}$" + (f"\\,$[{lo:+.2f},{hi:+.2f}]$" if lo != "" else "")
        out.append((norm_name(tag[: -len(readout) - 1]), fmt_ci(d["d_acc"]),
                    f"{s['ece']:.3f}", str(s["states"].get("overconfident_wrong", 0)),
                    f"{surg['cr_retention']:.2f}" if surg else "---", sel))
    return out


def compact_rows(analysis: dict, readout: str = "m4"):
    """{pretty_name: (dacc_str, ece_str, sel_str)} — point estimates, * if CI excludes 0."""
    out = {}
    base = analysis["baselines"][readout]
    out["baseline"] = ("---", f"{base['ece']:.3f}", "---")
    for tag, c in analysis["conditions"].items():
        if not tag.endswith(f"_{readout}"):
            continue
        d, s, surg = c["deltas"]["d_acc"], c["summary"], c["surgical"]
        star = "^{*}" if d["lo"] > 0 or d["hi"] < 0 else ""
        dacc = f"${d['point']:+.3f}{star}$"
        sel = "---"
        if surg is not None:
            lo, hi = c.get("selectivity_ci", [None, None])
            sstar = "^{*}" if lo is not None and lo > 0 else ""
            sel = f"${surg['selectivity']:+.3f}{sstar}$"
        out[norm_name(tag[: -len(readout) - 1])] = (dacc, f"{s['ece']:.3f}", sel)
    return out


def write_bench_table():
    """Transfer table: one column block per benchmark, key conditions only."""
    keys = ["baseline", "additive CAA (tuned)", "CAST-style (tuned)", "MiMiC (tuned)",
            "Linear-AcT (tuned)", "clamp $q_{50}$", "gated local action (tuned, ours)",
            "gated full-rank transport (tuned, ours)",
            "score-proportional dose (tuned, ours)", "sequential gate (tuned, ours)"]
    blocks = {}
    for name, d in (("MMLU", "v3_mmlu_s7"), ("ARC", "v3_arc"), ("GSM8K", "v3_gsm8k")):
        a = load(d)
        if a is None:
            continue
        blocks[name] = compact_rows(a, "m4")
    lines = []
    for key in keys:
        cells = [key]
        for name in ("MMLU", "ARC", "GSM8K"):
            r = blocks.get(name, {}).get(key)
            cells += (list(r) if r else ["--", "--", "--"])
        lines.append(" & ".join(cells) + r" \\")
    (GEN / "transfer_table.tex").write_text("\n".join(lines) + "\n\\hline\n")
    print("transfer table:", len(lines), "rows,", list(blocks))


PARETO = {
    "tuned_additive": (r"additive \cite{rimsky2024caa}", "cadd", "*", 2.6, "right=3pt"),
    "tuned_cast": (r"CAST-style \cite{lee2024cast}", "cadd", "*", 2.6, "right=3pt"),
    "prompt_hedge": ("prompting", "cadd", "*", 2.6, "below=3pt"),
    "tuned_mimic": (r"MiMiC \cite{singh2024mimic} ($\Theta(d^2)$)", "cot", "triangle*", 3.0, "above left=1pt"),
    "tuned_act": (r"Linear-AcT \cite{rodriguez2025act}", "cot", "triangle*", 3.0, "above=3pt"),
    "m4_conf_clamp_q50": (r"clamp $q_{50}$", "cclamp", "square*", 2.6, "above right=1pt"),
    "tuned_oursk": (r"gated $k$-dim transport", "cclamp", "square*", 2.6, "below right=1pt"),
    "tuned_gatedmimic": (r"\textbf{gated full-rank transport (ours)}", "cgate", "diamond*", 3.6, "above right=2pt"),
    "tuned_ourssoft": (r"\textbf{score-proportional dose (ours)}", "cgate", "diamond*", 3.6, "above left=2pt"),
    "tuned_ours": (r"\textbf{gated ablation (ours)}", "cgate", "diamond*", 3.6, "below=11pt"),
    "tuned_online": (r"\textbf{sequential gate (ours)}", "cgate", "pentagon*", 3.6, "above right=2pt"),
}


def write_pareto(dirname: str = "v3_mmlu_s7", readout: str = "m4"):
    """The selectivity-accuracy frontier, as \\addplot lines read from the analysis."""
    a = load(dirname)
    if a is None:
        return print("skip pareto (no analysis)")
    lines = []
    for tag, (label, color, mark, size, anchor) in PARETO.items():
        c = a["conditions"].get(f"{tag}_{readout}")
        if c is None or c["surgical"] is None:
            continue
        x = c["surgical"]["selectivity"]
        y = c["deltas"]["d_acc"]["point"]
        lines.append(f"\\addplot[only marks, mark={mark}, mark size={size}pt, {color}] "
                     f"coordinates {{({x:.4f},{y:.4f})}}\n"
                     f"  node[{anchor},font=\\scriptsize]{{{label}}};")
    (GEN / "pareto.tex").write_text("\n".join(lines) + "\n")
    print("pareto:", len(lines), "points")


MAIN_ROWS = ["m4_conf_add_a-0.75", "m4_conf_ablate", "m4_conf_clamp_q50",
             "m4_conf_otq_nonocw", "m4_conf_gate-ocw_vs_cr-cr_q50_clamp_q50",
             "m4_conf_gate-ocw_vs_cr-cr_q70_clamp_q50", "sweep_ocwcr-crq50_ablate",
             "online_d0p05_q50_L16_ablate"]
CITE = {"m4_conf_add_a-0.75": r" \\cite{rimsky2024caa}", "m4_conf_ablate": r" \\cite{arditi2024refusal}"}


def write_main_table(dirname: str = "v3_mmlu_s7", readout: str = "m4"):
    """The curated single-subset table, with every method the text discusses."""
    a = load(dirname)
    if a is None:
        return print("skip main table")
    base = a["baselines"][readout]
    lines = [f"baseline & --- & --- & {base['ece']:.3f} & "
             f"{base['states'].get('overconfident_wrong', 0)} & --- & --- \\\\"]
    for tag in MAIN_ROWS:
        c = a["conditions"].get(f"{tag}_{readout}")
        if c is None:
            continue
        sm, d, surg = c["summary"], c["deltas"], c["surgical"]
        lo, hi = c.get("selectivity_ci", [None, None])
        sel = "---" if surg is None else (
            f"${surg['selectivity']:+.3f}$" + (f"\\,$[{lo:+.2f},{hi:+.2f}]$" if lo is not None else ""))
        lines.append(" & ".join([
            norm_name(tag) + CITE.get(tag, ""),
            f"${d['d_acc']['point']:+.3f}$",
            f"$[{d['d_acc']['lo']:+.2f},{d['d_acc']['hi']:+.2f}]$",
            f"{sm['ece']:.3f}",
            str(sm["states"].get("overconfident_wrong", 0)),
            "---" if surg is None else f"{surg['cr_retention']:.3f}",
            sel]) + r" \\")
    (GEN / "main_table.tex").write_text("\n".join(lines) + "\n\\hline\n")
    print("main table:", len(lines), "rows; baseline acc", round(base["accuracy"], 4))


def write_full_table(dirname: str, outname: str):
    a = load(dirname)
    if a is None:
        print(f"skip {dirname} (no analysis)")
        return
    lines = []
    for readout in ("m4", "m2"):
        for r in rows_for(a, readout):
            lines.append(" & ".join([r[0], readout] + list(r[1:])) + r" \\")
    (GEN / outname).write_text("\n".join(lines) + "\n\\hline\n")
    print(outname, len(lines), "rows")


if __name__ == "__main__":
    write_pareto()
    write_main_table()
    write_bench_table()
    write_full_table("v3_mmlu_s7", "full_mmlu.tex")
    write_full_table("v3_arc", "full_arc.tex")
    write_full_table("v3_gsm8k", "full_gsm8k.tex")
    for seed in (11, 23, 31, 47):
        write_full_table(f"v3_mmlu_s{seed}", f"full_mmlu_s{seed}.tex")
