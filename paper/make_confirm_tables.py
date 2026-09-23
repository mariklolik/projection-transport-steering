from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GEN = ROOT / "paper" / "gen"
MODELS = (("gemma", "\\texttt{gemma-2-2b-it}"), ("qwen", "\\texttt{Qwen2.5-7B-Instruct}"))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def cell(x: dict, signed: bool = True, ci: bool = True) -> str:
    f = "+.3f" if signed else ".3f"
    s = f"${x['point']:{f}}$"
    return s + (f"\\,{{\\scriptsize$[{x['ci'][0]:{f}},{x['ci'][1]:{f}}]$}}" if ci else "")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def block(key: str, title: str, readout: str = "m5") -> list[str]:
    proto = RES / {"gemma": "v4_confirm", "qwen": "v5q_confirm"}[key] / "protocol.json"
    res = RES / "v4_pooled" / f"confirm_{key}.json"
    if not proto.exists() or not res.exists():
        return [f"\\multicolumn{{7}}{{l}}{{{title}: pending}} \\\\"]
    p, r = json.loads(proto.read_text()), json.loads(res.read_text())[readout]
    extra = RES / "v4_pooled" / f"confirm_{key}_extra.json"
    if extra.exists():
        r["methods"] = {**json.loads(extra.read_text())[readout]["methods"], **r["methods"]}
    adj = holm({a["tag"]: r["methods"][a["tag"]]["vs_ref"]["sel"]["p_two_sided"] for a in p["arms"] if a.get("primary")})
    rows = [f"\\multicolumn{{7}}{{l}}{{{title}, $n={r['n']}$, "
            f"$n_{{\\ocw}}={r['n_ocw']}$, $n_{{\\crok}}={r['n_cr']}$, unsteered accuracy ${r['baseline_acc']:.3f}$}} \\\\",
            "\\hline"]
    for a in p["arms"]:
        m = r["methods"][a["tag"]]
        vs = m.get("vs_ref", {}).get("sel")
        diff = "---" if vs is None else cell(vs) + (f" ({pval(adj[a['tag']])})" if a["tag"] in adj else "")
        name = f"\\textbf{{{a['name']}}}" if a["tag"] == p["ref"] else a["name"]
        rows.append(f"{name} & {a['budget']} & {cell(m['dacc'])} & {cell(m['ocw_rm'], False, False)} & "
                    f"{cell(m['cr_keep'], False, False)} & {cell(m['sel'])} & {diff} \\\\")
    return rows + ["\\hline"]


FAMILY_NAMES = {"additive": "additive CAA", "mimic": "MiMiC", "act": "Linear-AcT", "cast": "CAST-style, trace condition",
                "castdim": "CAST, prompt condition", "pts": "PTS post-hoc", "online": "PTS, prefix decision",
                "castprompt": "PTS, prompt decision", "published_gate": "earlier gate + ablation", "ungated": "ungated actions"}


def tuning_table(key: str, src: str) -> None:
    path = RES / src / "selection.json"
    if not path.exists():
        return
    sel = json.loads(path.read_text())
    rows = []
    for fam, name in FAMILY_NAMES.items():
        for cfg, v in sorted(sel["table"].get(fam, {}).items(), key=lambda kv: -kv[1]["selectivity"]):
            star = r"$\star$" if sel["winners"].get(fam) == cfg else ""
            cfg_tex = cfg.replace("_", r"\_")
            rows.append(rf"{name} & \texttt{{{cfg_tex}}}{star} & ${v['d_acc']:+.3f}$ & "
                        rf"{v['ocw_rm']:.3f} & {v['cr_keep']:.3f} & ${v['selectivity']:+.3f}$ \\")
    (GEN / f"tuning_{key}.tex").write_text("\n".join(rows) + "\n")


def capability_table() -> None:
    src = RES / "v4_capability"
    rows = []
    he, oe, wt = (json.loads((src / f"{t}.json").read_text()) if (src / f"{t}.json").exists() else None
                  for t in ("humaneval", "openended", "wikitext"))
    if not (he and oe and wt):
        return
    names = [k for k in he["methods"]]
    for k in names:
        h, o, w = he["methods"][k]["pass"], oe["methods"][k], wt["methods"][k]["nll"]
        rows.append(f"{k} & {h['mean']:.3f} & ${h['delta']:+.3f}$ {{\\scriptsize$[{h['delta_ci'][0]:+.3f},{h['delta_ci'][1]:+.3f}]$}} & "
                    f"{o['distinct3']['mean']:.3f} & {o['fluency_nll']['ppl']:.2f} & {w['ppl']:.1f} \\\\")
    fires = (f"% fire rates: humaneval post-hoc {he['fire_posthoc']:.2f} prompt {he['fire_prompt']:.2f}; "
             f"openended post-hoc {oe['fire_posthoc']:.2f} prompt {oe['fire_prompt']:.2f}; "
             f"wikitext post-hoc {wt['fire_posthoc']:.2f} prompt {wt['fire_prompt']:.2f}")
    (GEN / "capability_v4.tex").write_text(fires + "\n" + "\n".join(rows) + "\n")


LEGACY = {"tuned_ours": "gated ablation (tuned)", "tuned_additive": "additive CAA (tuned)",
          "m4_conf_ablate": "directional ablation", "tuned_cast": "CAST-style (tuned)", "tuned_mimic": "MiMiC (tuned)",
          "tuned_act": "Linear-AcT (tuned)", "prompt_hedge": "prompting", "tuned_ourssoft": "score-proportional dose (tuned)",
          "tuned_online": "sequential gate (tuned)", "sweep_ocwcr-crq50_ablate": "gated ablation (published)",
          "m4_conf_add_a-0.75": "additive $-0.75$ (published)"}


def legacy_table() -> None:
    src = RES / "v4_pooled" / "pooled_v3.json"
    if not src.exists():
        return
    r = json.loads(src.read_text())
    rows = []
    for tag, name in LEGACY.items():
        m4, m2 = r["m4"]["methods"][tag], r["m2"]["methods"][tag]
        vs = m4.get("vs_ref", {}).get("sel")
        rows.append(f"{name} & {cell(m4['sel'])} & {cell(vs) if vs else '---'} & {cell(m2['sel'])} & "
                    f"{cell(m2['dacc'], ci=False)} \\\\")
    (GEN / "legacy_pooled.tex").write_text("\n".join(rows) + "\n")


def secondary_table(key: str) -> None:
    src = RES / "v4_pooled" / f"confirm_{key}.json"
    proto = RES / {"gemma": "v4_confirm", "qwen": "v5q_confirm"}[key] / "protocol.json"
    if not src.exists() or not proto.exists():
        return
    r, p = json.loads(src.read_text()), json.loads(proto.read_text())
    if "m5m2" not in r or "m4" not in r:
        return
    rows = []
    for a in p["arms"]:
        m2, m4 = r["m5m2"]["methods"].get(a["tag"]), r["m4"]["methods"].get(a["tag"])
        if not m2 or not m4:
            continue
        d2, d4 = m2.get("vs_ref", {}).get("sel"), m4.get("vs_ref", {}).get("sel")
        rows.append(f"{a['name']} & {cell(m2['sel'])} & {cell(d2) if d2 else '---'} & "
                    f"{cell(m4['sel'])} & {cell(d4) if d4 else '---'} \\\\")
    (GEN / f"secondary_{key}.tex").write_text("\n".join(rows) + "\n")


def side_by_side() -> None:
    blocks = {}
    for key in ("gemma", "qwen"):
        proto = RES / {"gemma": "v4_confirm", "qwen": "v5q_confirm"}[key] / "protocol.json"
        res = RES / "v4_pooled" / f"confirm_{key}.json"
        if not proto.exists() or not res.exists():
            return
        p, r = json.loads(proto.read_text()), json.loads(res.read_text())["m5"]
        extra = RES / "v4_pooled" / f"confirm_{key}_extra.json"
        if extra.exists():
            r["methods"] = {**json.loads(extra.read_text())["m5"]["methods"], **r["methods"]}
        adj = holm({a["tag"]: r["methods"][a["tag"]]["vs_ref"]["sel"]["p_two_sided"] for a in p["arms"] if a.get("primary")})
        blocks[key] = (p, r, adj)
    rows = []
    for i, arm in enumerate(blocks["gemma"][0]["arms"]):
        cells = []
        for key in ("gemma", "qwen"):
            p, r, adj = blocks[key]
            a = p["arms"][i]
            m = r["methods"][a["tag"]]
            ok = "" if m["dacc"]["ci"][0] > -0.02 else "$^\\dagger$"
            vs = m.get("vs_ref", {}).get("sel")
            d = "---" if vs is None else f"${vs['point']:+.3f}$" + (f" ({pval(adj[a['tag']])})" if a["tag"] in adj else "")
            sel = m["sel"]
            cells.append(f"${m['dacc']['point']:+.3f}${ok} & {m['cr_keep']['point']:.2f} & ${sel['point']:.3f}$ "
                         f"{{\\tiny$[{sel['ci'][0]:.2f},{sel['ci'][1]:.2f}]$}} & {d}")
        short = {"prompt_hedge": "prompting", "tuned_additive": "additive CAA", "plain_ablate": "dir. ablation",
                 "tuned_mimic": "MiMiC", "tuned_act": "Linear-AcT", "tuned_cast": "CAST-style (trace)"}
        t = arm["tag"]
        name = short.get(t) or ("CAST (prompt)" if t.startswith("castdim") else "PTS, prefix" if t.startswith("detonline")
                                else "PTS, prompt" if t.startswith("detprompt") else "PTS, post-hoc")
        name = f"\\textbf{{{name}}}" if arm["tag"] == blocks["gemma"][0]["ref"] else name
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if arm["tag"] == blocks["gemma"][0]["ref"]:
            rows.append("\\hline")
    (GEN / "confirm_side.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    side_by_side()
    secondary_table("gemma")
    secondary_table("qwen")
    legacy_table()
    tuning_table("gemma", "v4_parity_m5")
    tuning_table("qwen", "v5q_parity_m5")
    capability_table()
    lines = []
    for key, title in MODELS:
        lines += block(key, title)
    (GEN / "confirm_table.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
