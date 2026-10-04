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
        rows.append(f"{k.replace('PGS', 'PTS')} & {h['mean']:.3f} & ${h['delta']:+.3f}$ {{\\scriptsize$[{h['delta_ci'][0]:+.3f},{h['delta_ci'][1]:+.3f}]$}} & "
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
        rows.append(f"{arm_name(a['tag'])} & {cell(m2['sel'])} & {cell(d2) if d2 else '---'} & "
                    f"{cell(m4['sel'])} & {cell(d4) if d4 else '---'} \\\\")
    (GEN / f"secondary_{key}.tex").write_text("\n".join(rows) + "\n")


SETTINGS = (("gemma", "v4_confirm"), ("qwen", "v5q_confirm"), ("arc", "v6a_confirm"))
SHORT = {"prompt_hedge": "prompting", "tuned_additive": "additive CAA", "plain_ablate": "dir. ablation",
         "tuned_mimic": "MiMiC", "tuned_act": "Linear-AcT", "tuned_cast": "CAST-style (trace)", "plain_null": "rerun, no edit"}


def arm_name(t: str) -> str:
    return {"ref:null": "same decision, no edit", "ref:rand": "same decision, random direction",
            "ref:randm": "same decision, random direction", "ref:override": "same decision, override readout"}.get(t) or SHORT.get(t) or ("CAST (prompt)" if t.startswith("castdim") else "PTS, prefix" if t.startswith("detonline")
                            else "PTS, prompt" if t.startswith("detprompt") else "null: decision, no edit"
                            if t.startswith("detnull") else "null: decision, random dir." if t.startswith("detrand")
                            else "PTS, post-hoc")


def load_setting(key: str, d: str):
    p, r = json.loads((RES / d / "protocol.json").read_text()), json.loads((RES / "v4_pooled" / f"confirm_{key}.json").read_text())["m5"]
    extra = RES / "v4_pooled" / f"confirm_{key}_extra.json"
    if extra.exists():
        r["methods"] = {**json.loads(extra.read_text())["m5"]["methods"], **r["methods"]}
    net = json.loads((RES / "v4_pooled" / f"net_{key}.json").read_text())
    adj = holm({a["tag"]: r["methods"][a["tag"]]["vs_ref"]["sel"]["p_two_sided"] for a in p["arms"] if a.get("primary")})
    return p, r, net, adj


def ref_holm() -> dict[tuple[str, str, str], float]:
    ps = {}
    for key, _ in SETTINGS:
        for arm, r in json.loads((RES / "v4_pooled" / f"references_{key}.json").read_text())["arms"].items():
            ps[(key, arm, "null")], ps[(key, arm, "rand")] = r["vs_null"]["p_one_sided"], r["vs_random"]["p_one_sided"]
    return holm(ps)


def setting_cells(key: str, block, rh, full: bool) -> list[tuple[str, str]]:
    p, r, net, adj = block
    ref = json.loads((RES / "v4_pooled" / f"references_{key}.json").read_text())["arms"][p["ref"]]
    star = lambda x: "" if x["ci"][0] < 0 < x["ci"][1] else "$^*$"  # noqa: E731
    out = []
    for a in p["arms"]:
        tag = a["tag"]
        m, n = r["methods"][tag], net["arms"][tag]
        ok = "" if m["dacc"]["ci"][0] > -0.02 else "$^\\dagger$"
        vs = m.get("vs_ref", {}).get("sel")
        d = "---" if vs is None else f"${vs['point']:+.3f}$" + (f" ({pval(adj[tag])})" if tag in adj else "")
        sel = m["sel"]
        selc = f"${sel['point']:.3f}$ {{\\tiny$[{sel['ci'][0]:.2f},{sel['ci'][1]:.2f}]$}}"
        ece = f"${n['d_ece']['point']:+.3f}$" + star(n["d_ece"])
        if full:
            ocw = f"${100 * n['d_ocw']['point']:+.1f}$" + star(n["d_ocw"])
            out.append((tag, f"${m['dacc']['point']:+.3f}${ok} & {m['ocw_rm']['point']:.2f} & {m['cr_keep']['point']:.2f} & "
                             f"{selc} & {d} & {ocw} & {ece}"))
        else:
            out.append((tag, f"${m['dacc']['point']:+.3f}${ok} & {selc} & {d} & {ece}"))
    nl = net["arms"][net["nulls"]["gated_null"]]
    dense = list(json.loads((RES / "v4_pooled" / f"dense_{key}.json").read_text())["arms"].values())[0]
    dr = dense["random"]
    refs = [("ref:null", f"${nl['dacc']['point']:+.3f}$", ref["null"], f"${-ref['vs_null']['point']:+.3f}$ ($<$0.001)",
             f"${nl['d_ece']['point']:+.3f}$" + star(nl["d_ece"])),
            ("ref:rand" + ("m" if key == "qwen" else ""), "---", {"point": dr["mean"], "ci": None},
             f"${-dense['vs_random']['point']:+.3f}$ ({pval(dense['vs_random']['p_one_sided'])})", "---")]
    ov = ref["override_matched"]
    refs.append(("ref:override", "$+0.000$", ov,
                 f"${-ref['steer_minus_override']['point']:+.3f}$", "---"))
    for tag, dacc, sl, d, ece in refs:
        selc = f"${sl['point']:.3f}$" + ("" if sl["ci"] is None else f" {{\\tiny$[{sl['ci'][0]:.2f},{sl['ci'][1]:.2f}]$}}")
        out.append((tag, f"{dacc} & --- & --- & {selc} & {d} & --- & {ece}" if full else f"{dacc} & {selc} & {d} & {ece}"))
    return out


def write_rows(fname: str, cols: list[list[tuple[str, str]]], ref_tag: str) -> None:
    rows, n_arms = [], len(cols[0]) - 3
    for i in range(len(cols[0])):
        tag = cols[0][i][0]
        name = arm_name(tag)
        name = f"\\textbf{{{name}}}" if tag == ref_tag else name
        if i == n_arms:
            rows.append("\\hline")
        rows.append(f"{name} & " + " & ".join(c[i][1] for c in cols) + " \\\\")
        if tag == ref_tag:
            rows.append("\\hline")
    (GEN / fname).write_text("\n".join(rows) + "\n")


def side_by_side() -> None:
    blocks = {k: load_setting(k, d) for k, d in SETTINGS}
    rh = ref_holm()
    ref = blocks["gemma"][0]["ref"]
    write_rows("confirm_gemma.tex", [setting_cells("gemma", blocks["gemma"], rh, True)], ref)
    write_rows("confirm_qa.tex", [setting_cells(k, blocks[k], rh, False) for k in ("qwen", "arc")], ref)


def net_table() -> None:
    rows = []
    for key, d in SETTINGS:
        p, r, net, _ = load_setting(key, d)
        title = {"gemma": "Gemma-2-2B, MMLU", "qwen": "Qwen2.5-7B, MMLU", "arc": "Gemma-2-2B, ARC"}[key]
        rows += [f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for tag in [a["tag"] for a in p["arms"]] + list(net["nulls"].values()):
            m = net["arms"][tag]
            f = lambda x: f"${100 * m[x]['point']:+.1f}$"  # noqa: E731
            rows.append(f"{arm_name(tag)} & ${m['sel']['point']:.3f}$ & ${m['net_gated_null']['point']:+.3f}$ & {f('d_ocw')} & "
                        f"{f('d_conf_wrong')} & {f('d_conf_right')} & {cell(m['d_ece'], ci=False)} & {cell(m['d_brier'], ci=False)} & "
                        f"{f('d_forced')} & ${m['sel_unforced']:.3f}$ \\\\")
        rows.append("\\hline")
    (GEN / "net_table.tex").write_text("\n".join(rows) + "\n")


def transitions_table() -> None:
    states = ("overconfident_wrong", "nonconfident_wrong", "nonconfident_right", "confident_right")
    short = ("OCW", "NCW", "NCR", "CR")
    rows = []
    for key, arm, title in (("gemma", "real | det_q60_alpha-0.375", "Gemma, PTS"), ("qwen", "real | det_q40_ablate", "Qwen, PTS")):
        src = RES / "v4_pooled" / f"matrix_{key}.json"
        if not src.exists():
            continue
        t = json.loads(src.read_text())["arms"][arm]["transitions"]
        for i, st in enumerate(states):
            lead = f"\\multirow{{4}}{{*}}{{{title}}}" if i == 0 else ""
            rows.append(f"{lead} & {short[i]} & " + " & ".join(str(t[st][u]) for u in states) + " \\\\")
        rows.append("\\hline")
    (GEN / "transitions.tex").write_text("\n".join(rows) + "\n")


def calibration_table() -> None:
    rows = []
    spec = {"gemma": [("unsteered", None), ("shift $-0.25$, ungated", "ungated | plain_alpha-0.25"),
                      ("ablation, ungated", "ungated | plain_ablate"), ("PTS, post-hoc", "real | det_q60_alpha-0.375"),
                      ("override, probe on answer", "probe, answer | override"),
                      ("override, CAST condition", "CAST condition, prompt | override")],
            "qwen": [("unsteered", None), ("shift $-0.25$, ungated", "ungated | plain_alpha-0.25"),
                     ("ablation, ungated", "ungated | plain_ablate"), ("PTS, post-hoc", "real | det_q40_ablate"),
                     ("override, probe on answer", "probe, answer | override"),
                     ("override, CAST condition", "CAST condition, prompt | override")]}
    data = {k: json.loads((RES / "v4_pooled" / f"matrix_{k}.json").read_text()) for k in spec
            if (RES / "v4_pooled" / f"matrix_{k}.json").exists()}
    if len(data) < 2:
        return
    for i in range(len(spec["gemma"])):
        cells = []
        for k in ("gemma", "qwen"):
            name, arm = spec[k][i]
            c = data[k]["baseline_calibration"] if arm is None else data[k]["arms"][arm]["calibration"]
            cells.append(f"{c['ece']:.3f} & {c['brier']:.3f} & {c['auroc_conf']:.3f}")
        rows.append(f"{spec['gemma'][i][0]} & " + " & ".join(cells) + " \\\\")
    (GEN / "calibration.tex").write_text("\n".join(rows) + "\n")


def ref_table() -> None:
    names = {"trace": "PTS, post-hoc", "prompt": "PTS, prompt", "u": "CAST-style (trace)", "castdim": "CAST (prompt)"}
    rows = []
    for key, title in (("gemma", "Gemma, MMLU"), ("qwen", "Qwen, MMLU"), ("arc", "Gemma, ARC")):
        src = RES / "v4_pooled" / f"references_{key}.json"
        if not src.exists():
            continue
        arms = json.loads(src.read_text())["arms"]
        for i, a in enumerate(arms.values()):
            lead = f"\\multirow{{{len(arms)}}}{{*}}{{{title}}}" if i == 0 else ""
            ov, d = a["override_matched"], a["steer_minus_override"]
            vn, vr = a["vs_null"], a["vs_random"]
            sig = lambda x: "$^*$" if x["ci"][0] > 0 else ""  # noqa: E731
            rows.append(f"{lead} & {names[a['decision']]} & ${a['sel']['point']:.3f}$ & ${a['null']['point']:.3f}$ & "
                        f"${vn['point']:+.3f}${sig(vn)} & ${a['random']['mean']:.3f}$ & ${vr['point']:+.3f}${sig(vr)} & "
                        f"${ov['point']:.3f}$ & ${d['point']:+.3f}$ {{\\tiny$[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]$}} & "
                        f"{'yes' if a['eq2_agrees'] else 'no'} \\\\")
        rows.append("\\hline")
    (GEN / "references.tex").write_text("\n".join(rows) + "\n")


def dense_table() -> None:
    rows = []
    for key, title in (("gemma", "Gemma, MMLU (100)"), ("arc", "Gemma, ARC (100)"), ("qwen", "Qwen, MMLU (30, matched)")):
        a = list(json.loads((RES / "v4_pooled" / f"dense_{key}.json").read_text())["arms"].values())[0]
        r, v = a["random"], a["vs_random"]
        rows.append(f"{title} & ${a['sel']['point']:.3f}$ & ${r['mean']:.3f}$ & ${r['p95']:.3f}$ & ${r['max']:.3f}$ & "
                    f"{r['rank']} & ${v['point']:+.3f}$ $[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]$ \\\\")
    (GEN / "dense.tex").write_text("\n".join(rows) + "\n")


def tox_table() -> None:
    def load_m(m):
        d = json.loads((RES / "toxicity" / m / "confirm.json").read_text())["arms"]
        r = RES / "toxicity" / m / "rates.json"
        rates = json.loads(r.read_text()) if r.exists() else {k: {"removal": v["removal"], "retention": v["retention"]} for k, v in d.items()}
        return d, rates
    models = ("gemma_2_2b_it", "qwen2_5_7b_it", "mistral_7b_it")
    data = {m: load_m(m) for m in models}
    ot = json.loads((RES / "toxicity" / "gemma_2_2b_it" / "transport_confirm.json").read_text())
    pv = lambda p: pval(p)  # noqa: E731
    rows = []
    names = [("prompt", "prompting"), ("ungated", "global shift"), ("cast", "CAST"), ("probe", "\\textbf{PTS}, shift"),
             ("ot", "\\textbf{PTS}, scaled transport"), ("post", "PTS, post-hoc")]
    for key, name in names:
        cells = []
        for m in models:
            d, r = data[m]
            if key == "ot":
                if m != "gemma_2_2b_it":
                    cells.append("\\multicolumn{3}{c|}{---}" if m != models[-1] else "\\multicolumn{3}{c}{---}")
                    continue
                o = ot["ot"]
                cells.append(f"{100 * o['removal']:.0f}/{100 * o['retention']:.0f} & ${o['sel']:.3f}$ {{\\tiny$[{o['ci'][0]:.2f},{o['ci'][1]:.2f}]$}} & "
                             f"${ot['ot_minus_pts_shift']['point']:+.3f}$ ({pv(2 * ot['ot_minus_pts_shift']['p_one_sided'])})")
                continue
            a = d[key]
            vs = "---" if key == "probe" else f"${a['vs_pgs']['point']:+.3f}$ ({pv(a['vs_pgs']['p_two_sided'])})"
            cells.append(f"{100 * r[key]['removal']:.0f}/{100 * r[key]['retention']:.0f} & ${a['sel']['point']:.3f}$ "
                         f"{{\\tiny$[{a['sel']['ci'][0]:.2f},{a['sel']['ci'][1]:.2f}]$}} & {vs}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    rows.append("\\hline")
    for label, name in (("random", "same decision, random direction"), ("override", "same decision, refusal")):
        cells = []
        for m in models:
            d, r = data[m]
            a = d["probe"]
            if label == "random":
                cells.append(f"--- & ${a['random']['mean']:.3f}$ & ${-a['vs_random']['point']:+.3f}$ ($<$0.001)")
            else:
                rem = r["probe"].get("override_removal", a["override"].get("removal"))
                ret = r["probe"].get("override_retention", a["override"].get("retention"))
                cells.append(f"{100 * rem:.0f}/{100 * ret:.0f} & ${a['override']['point']:.3f}$ & ${-a['steer_minus_override']['point']:+.3f}$ ($<$0.001)")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (GEN / "toxicity.tex").write_text("\n".join(rows) + "\n")


def law_table() -> None:
    names = {"trace": "PTS post-hoc", "prompt": "PTS at the prompt", "u": "CAST-style, trace", "castdim": "CAST, prompt"}
    rows = []
    for key, title in (("m5", "Gemma, MMLU"), ("qwen", "Qwen, MMLU"), ("arc", "Gemma, ARC")):
        for i, c in enumerate(json.loads((RES / "v4_pooled" / f"gate_law_confirm_{key}.json").read_text())["checks"]):
            lead = f"\\multirow{{4}}{{*}}{{{title}}}" if i == 0 else ""
            rows.append(f"{lead} & {names[c['decision']]} & {c['tpr']:.3f} & {c['fpr']:.3f} & {c['rho_o']:.3f} & {c['rho_c']:.3f} & "
                        f"${c['sel_law']:+.3f}$ & ${c['sel_sim']:+.3f}$ & ${c['sel_real']:+.3f}$ \\\\")
        rows.append("\\hline")
    (GEN / "law_confirm.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    dense_table()
    tox_table()
    ref_table()
    law_table()
    calibration_table()
    transitions_table()
    side_by_side()
    net_table()
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
