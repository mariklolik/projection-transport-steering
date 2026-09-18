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


def norm_name(tag: str) -> str:
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
    keys = ["baseline", "additive $-0.75$", "clamp $q_{50}$", r"quantile-OT non-\ocw{}",
            "gated clamp $q_{50}$"]
    blocks = {}
    for name, d in (("MMLU", "steering_v2"), ("ARC", "steering_v2_arc"), ("GSM8K", "steering_v2_gsm8k")):
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
    write_bench_table()
    write_full_table("steering_v2", "full_mmlu.tex")
    write_full_table("steering_v2_arc", "full_arc.tex")
    write_full_table("steering_v2_gsm8k", "full_gsm8k.tex")
    write_full_table("steering_v2_gpqa", "full_gpqa.tex")
    for seed in (11, 23, 31, 47):
        write_full_table(f"steering_v2_s{seed}", f"full_mmlu_s{seed}.tex")
