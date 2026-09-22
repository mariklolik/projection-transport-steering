# LaTeX fragments for the added evidence: layer sweep, equal-budget tuning,
# capability preservation, on-manifold diagnostics and end-to-end serving cost.
#   python3 paper/make_rebuttal_tables.py

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "gen"
RES = ROOT / "results"
GEN.mkdir(exist_ok=True)

PRETTY = {
    "baseline": "unsteered", "additive": r"additive $-0.75$ \cite{rimsky2024caa}",
    "ablation": r"ablation \cite{arditi2024refusal}", "clamp_q50": r"clamp $q_{50}$ (ours)",
    "quantile-OT": r"quantile-OT (ours)", "MiMiC": r"MiMiC \cite{singh2024mimic}",
    "Linear-AcT": r"Linear-AcT \cite{rodriguez2025act}",
    "online-gated ablation": r"online-gated ablation (ours)",
    "additive (tuned)": r"additive, tuned dose \cite{rimsky2024caa}",
    "MiMiC (tuned)": r"MiMiC, tuned layer \cite{singh2024mimic}",
    "gated full-rank transport": r"gated full-rank transport (ours)",
    "random norm-matched": "random direction, norm-matched",
    "gated ablation (two-pass)": r"gated ablation, post-hoc gate (ours)",
    "gated ablation (online, ours)": r"gated ablation, sequential gate (ours)",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def write(name: str, lines: list[str]) -> None:
    body = lines or [r"pending & \\"]
    (GEN / name).write_text("\n".join(body) + "\n\\hline\n")
    print(f"{name}: {len(lines)} rows")


def layers_table() -> None:
    a = load(RES / "v3_layers" / "layer_sweep.json")
    if not a:
        return print("skip layer table")
    rows = []
    for L, v in sorted(a["layers"].items(), key=lambda kv: int(kv[0])):
        star = r"\,$\star$" if int(L) == a["best_layer_by_gate_auroc"] else ""
        rows.append(f"{L}{star} & {v['cohen_d_confident_vs_not_tokens']:+.3f} & "
                    f"{v['cohen_d_ocw_vs_cr_tokens']:+.3f} & {v['gate_auroc_ocw_vs_cr']:.3f} & "
                    f"{v['ceiling_selectivity']:+.3f} \\\\")
    write("layer_table.tex", rows)


def prefix_table() -> None:
    """How much a gate loses by having to decide after t generated tokens."""
    rows = []
    for L in (14, 16):
        a = load(ROOT / "behaviour_specific" / "overconfidence" / "directions" / f"gatecal_L{L}.json")
        if not a:
            continue
        cells = [f"layer {L}"] + [f"{a['prefix_auroc'][str(t)]:.3f}"
                                  for t in (8, 16, 32, 64, 128, 256) if str(t) in a["prefix_auroc"]]
        rows.append(" & ".join(cells) + r" \\")
    write("prefix_table.tex", rows)


def parity_table() -> None:
    rows, names = [], {"additive": r"additive CAA \cite{rimsky2024caa}",
                       "cast": r"CAST-style \cite{lee2024cast}",
                       "mimic": r"MiMiC \cite{singh2024mimic}",
                       "act": r"Linear-AcT \cite{rodriguez2025act}",
                       "ours": "gated local action (ours)",
                       "oursk": r"gated $k$-dim transport (ours)",
                       "ourssoft": "score-proportional dose (ours)",
             "ourstok": "gated token-local clamp (ours)",
                       "online": "sequential gate (ours, single pass)",
                       "gatedmimic": r"NP-gate $\times$ MiMiC (Thm.~\ref{thm:joint})"}
    for key, pretty in names.items():
        a = load(RES / "v3_parity_n1200" / f"parity_{key}.json")
        if not a:
            continue
        w = a["winner_stats"]
        rows.append(f"{pretty} & {a['budget']} & \\texttt{{{a['winner'].replace('_', chr(92) + '_')}}} & "
                    f"{w['d_acc']:+.3f} & {w['ocw_rm']:.3f} & {w['cr_keep']:.3f} & "
                    f"{w['selectivity']:+.3f} \\\\")
    write("parity_table.tex", rows)


def capability_table() -> None:
    he = load(RES / "v3_openended" / "humaneval.json")
    oe = load(RES / "v3_openended" / "openended.json")
    wt = load(RES / "v3_openended" / "wikitext.json")
    jd = (load(RES / "v3_openended" / "judge.json") or {}).get("methods", {})
    if not (he or oe or wt):
        return print("skip capability table")
    order = ["baseline", "additive", "additive (tuned)", "MiMiC", "MiMiC (tuned)",
             "Linear-AcT", "clamp_q50", "quantile-OT", "online-gated ablation",
             "gated full-rank transport"]
    rows = []
    for k in order:
        h = (he or {}).get("methods", {}).get(k, {})
        o = (oe or {}).get("methods", {}).get(k, {})
        w = (wt or {}).get("methods", {}).get(k, {})
        if not (h or o or w):
            continue
        fr = o.get("fire_rate", h.get("fire_rate"))
        gated = "gated" in k
        rows.append(" & ".join([
            PRETTY.get(k, k),
            f"{h['pass@1']:.3f}" if h else "--",
            # a teacher-forced corpus has no generated prefix for a gate to read
            "n/a" if gated else (f"{w['ppl']:.1f}" if w else "--"),
            f"{o['fluency_ppl']:.2f}" if o else "--",
            f"{o['distinct3']:.3f}" if o else "--",
            (lambda j: f"{j['adjusted_win_rate']:.3f}" if j else "--")(
                jd.get(k) or jd.get(k.replace("_", " ")) or jd.get(k.replace("-", " "))),
            (f"{fr:.3f}" if fr is not None else "--")]) + r" \\")
    write("capability_table.tex", rows)


def manifold_table() -> None:
    blocks = {}
    for bench in ("mmlu", "arc", "gsm8k"):
        a = load(RES / "v3_manifold" / f"manifold_{bench}.json")
        if a:
            blocks[bench] = a
    if not blocks:
        return print("skip manifold table")
    order = ["random norm-matched", "additive", "additive (tuned)", "MiMiC",
             "Linear-AcT", "clamp_q50", "quantile-OT", "ablation"]
    rows = []
    for k in order:
        cells = [PRETTY.get(k, k)]
        for bench in ("mmlu", "arc", "gsm8k"):
            v = blocks.get(bench, {}).get("ops", {}).get(k)
            cells += ([f"{v['rel_displacement_mean']:.3f}", f"{v['nn_ratio']:.2f}",
                       f"{100 * v.get('cond_exceed_99', float('nan')):.1f}",
                       f"{v.get('kl_next_token', float('nan')):.3f}"] if v else ["--"] * 4)
        rows.append(" & ".join(cells) + r" \\")
    write("manifold_table.tex", rows)


def split_table() -> None:
    """How much the equal-budget winner moves when the tuning split triples."""
    names = {"additive": r"additive CAA \cite{rimsky2024caa}",
             "cast": r"CAST-style \cite{lee2024cast}",
             "mimic": r"MiMiC \cite{singh2024mimic}",
             "act": r"Linear-AcT \cite{rodriguez2025act}",
             "ours": "gated local action (ours)",
             "oursk": r"gated $k$-dim transport (ours)",
             "ourssoft": "score-proportional dose (ours)",
             "ourstok": "gated token-local clamp (ours)",
             "online": "sequential gate (ours, single pass)",
             "gatedmimic": r"NP-gate $\times$ MiMiC (Thm.~\ref{thm:joint})"}
    rows = []
    for key, pretty in names.items():
        a = load(RES / "v3_parity" / f"parity_{key}.json")
        b = load(RES / "v3_parity_n1200" / f"parity_{key}.json")
        if not (a and b):
            continue
        tag = lambda j: j["winner"].replace("_", chr(92) + "_")
        rows.append(f"{pretty} & \\texttt{{{tag(a)}}} & {a['winner_stats']['selectivity']:+.3f} & "
                    f"\\texttt{{{tag(b)}}} & {b['winner_stats']['selectivity']:+.3f} & "
                    f"{b['winner_stats']['selectivity'] - a['winner_stats']['selectivity']:+.3f} \\\\")
    write("split_table.tex", rows)


TRACE_ROWS = {
    "baseline_m2": "unsteered",
    "m4_conf_add_a-0.75_m2": r"additive $-0.75$ \cite{rimsky2024caa}",
    "cast_gate-cr_q50_add-0.75_m2": r"CAST-style \cite{lee2024cast}",
    "mimic_full_m2": r"MiMiC \cite{singh2024mimic}",
    "m4_conf_clamp_q50_m2": r"clamp $q_{50}$ (ours)",
    "sweep_ocwcr-crq50_ablate_m2": "gated ablation (ours)",
    "online_d0p05_q50_L16_t16_ablate_m2": "sequential gate (ours)",
}


def trace_table(dirname: str = "v3_mmlu_s7") -> None:
    """What the intervention does to the reasoning trace itself."""
    a = load(RES / dirname / "trace_analysis.json")
    if not a:
        return print("skip trace table")
    m = a.get("metrics", a)
    rows = []
    for tag, pretty in TRACE_ROWS.items():
        v = m.get(tag)
        if not v:
            continue
        rows.append(f"{pretty} & {v['mean_words']:.0f} & {v['close_rate']:.2f} & "
                    f"{v['boxed_rate']:.2f} & {v['mean_distinct3']:.2f} & "
                    f"{100 * v['pct_degenerate']:.1f} & {v.get('hedge_per_trace', 0):.2f} & "
                    f"{v.get('certainty_per_trace', 0):.2f} \\\\")
    write("trace_table.tex", rows)


def serving_table() -> None:
    a = load(RES / "v3_serving" / "serving.json")
    if not a:
        return print("skip serving table")
    rows = []
    for k, v in a["methods"].items():
        extra = ""
        if "n_regenerated" in v:
            extra = f" ({v['n_regenerated']}/{a['n']} regenerated)"
        rows.append(f"{PRETTY.get(k, k)}{extra} & {v['passes']} & {v['seconds']:.1f} & "
                    f"{v['latency_x_baseline']:.2f}$\\times$ \\\\")
    write("serving_table.tex", rows)


if __name__ == "__main__":
    layers_table()
    prefix_table()
    parity_table()
    split_table()
    capability_table()
    manifold_table()
    trace_table()
    serving_table()
