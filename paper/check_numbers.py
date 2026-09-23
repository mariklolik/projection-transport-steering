# Every quantity the prose asserts, printed straight from the analyses, so the
# text can be checked against the runs it claims to report.
#   python3 paper/check_numbers.py [run-dir-prefix]

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "v3_mmlu_s"
SEEDS = ("7", "11", "23", "31", "47")
KEY = {
    "baseline": None,
    "prompting": "prompt_hedge",
    "additive": "m4_conf_add_a-0.75",
    "ablation": "m4_conf_ablate",
    "CAST-style": "cast_gate-cr_q50_add-0.75",
    "MiMiC": "mimic_full",
    "Linear-AcT l0.5": "act_l05",
    "Linear-AcT l1.0": "act_l10",
    "clamp q50": "m4_conf_clamp_q50",
    "quantile-OT non-ocw": "m4_conf_otq_nonocw",
    "gated clamp": "m4_conf_gate-ocw_vs_cr-cr_q50_clamp_q50",
    "gated ablation": "sweep_ocwcr-crq50_ablate",
    "LDA-gated ablation": "sweep_lda-allq50_ablate",
    "online-gated ablation": "online_d0p05_q50_L16_ablate",
    "TUNED additive": "tuned_additive",
    "TUNED cast": "tuned_cast",
    "TUNED mimic": "tuned_mimic",
    "TUNED act": "tuned_act",
    "TUNED ours": "tuned_ours",
    "TUNED oursk": "tuned_oursk",
    "TUNED gatedmimic": "tuned_gatedmimic",
    "TUNED ourssoft": "tuned_ourssoft",
    "TUNED ourspost": "tuned_ourspost",
    "TUNED online": "tuned_online",
}


def analyses(dirs):
    for d in dirs:
        p = RES / d / "analysis_v2.json"
        if p.exists():
            yield d, json.loads(p.read_text())


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:+.3f}"
    return f"{st.mean(vals):+.3f}+-{st.stdev(vals):.3f}"


def per_readout(dirs, readout):
    print(f"\n### {readout} over {dirs}")
    rows = list(analyses(dirs))
    if not rows:
        return print("  (no analyses)")
    b = [a["baselines"][readout] for _, a in rows]
    print(f"  baseline: acc {agg([x['accuracy'] for x in b])} ece {agg([x['ece'] for x in b])} "
          f"aurc {agg([x.get('selective', {}).get('aurc') for x in b])} "
          f"ocw {[x['states'].get('overconfident_wrong') for x in b]} "
          f"cr {[x['states'].get('confident_right') for x in b]}")
    for name, tag in KEY.items():
        if tag is None:
            continue
        cs = [a["conditions"].get(f"{tag}_{readout}") for _, a in rows]
        cs = [c for c in cs if c]
        if not cs:
            continue
        print(f"  {name:24s} n={len(cs)} dacc {agg([c['deltas']['d_acc']['point'] for c in cs])} "
              f"ece {agg([c['summary']['ece'] for c in cs])} "
              f"aurc {agg([c.get('selective', {}).get('aurc') for c in cs])} "
              f"sel {agg([c['surgical']['selectivity'] if c['surgical'] else None for c in cs])} "
              f"keep {agg([c['surgical']['cr_retention'] if c['surgical'] else None for c in cs])} "
              f"ocw_rm {agg([c['surgical']['ocw_removal'] if c['surgical'] else None for c in cs])}")


if __name__ == "__main__":
    per_readout([f"{PREFIX}{s}" for s in SEEDS], "m4")
    per_readout([f"{PREFIX}{s}" for s in SEEDS], "m2")
    for d in ("v3_arc", "v3_gsm8k"):
        per_readout([d], "m4")
    per_readout([f"steering_9b_s{s}" for s in ("7", "11", "23")], "m4")
