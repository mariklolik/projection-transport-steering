from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v4_pooled"
PIC = ROOT / "paper" / "pic"
STAGES = (("prompt", 0), ("prefix16", 16), ("prefix32", 32), ("prefix64", 64), ("prefix128", 128), ("trace", 256))
ACTIONS = {"plain_ablate": "ablate", "plain_alpha-0.25": "shift", "plain_alpha-0.375": "shift",
           "plain_alpha-0.5": "shift", "plain_L10_reg0.0001": "mimic", "plain_clamp_q50": "clamp"}


def law() -> None:
    src = RES / "gate_law_tuning_m5.json"
    if not src.exists():
        return
    r = json.loads(src.read_text())
    by_kind: dict[str, list[str]] = {}
    for det in ("u@L14", "detector_m5"):
        for act, kind in ACTIONS.items():
            by_kind.setdefault(kind, []).extend(f"{c['sel_law']:.4f},{c['sel_sim']:.4f}"
                                               for c in r["detectors"][det]["actions"][act]["curve"][:-1])
    for kind, rows in by_kind.items():
        name = "v4_law.csv" if kind == "shift" else f"v4_law_{kind}.csv"
        (PIC / name).write_text("law,sim,kind\n" + "\n".join(f"{x},{kind}" for x in rows) + "\n")
    real = RES / "gate_law_confirm_m5.json"
    if real.exists():
        rows = [f"{c['sel_law']:.4f},{c['sel_real']:.4f}" for c in json.loads(real.read_text())["checks"]]
        (PIC / "v4_law_real.csv").write_text("law,real\n" + "\n".join(rows) + "\n")


def law_qwen() -> None:
    src = RES / "gate_law_tuning_qwen.json"
    if not src.exists():
        return
    r = json.loads(src.read_text())
    rows = [f"{c['sel_law']:.4f},{c['sel_sim']:.4f}" for det in r["detectors"].values()
            for a in det["actions"].values() for c in a["curve"][:-1]]
    (PIC / "v4_law_qwen.csv").write_text("law,sim\n" + "\n".join(rows) + "\n")


def auroc_sel() -> None:
    for model in ("gemma", "qwen", "arc"):
        src = RES / f"matrix_{model}.json"
        if not src.exists():
            continue
        r = json.loads(src.read_text())
        rows = ["name,auroc,steer,steer_lo,steer_hi,override,ov_lo,ov_hi"]
        for name, info in r["confirm"].items():
            st, ov = r["arms"][f"{name} | steer"]["sel"], r["arms"][f"{name} | override"]["sel"]
            rows.append(f"{name.replace(',', ';')},{info['auroc_heldout']:.4f},{st['point']:.4f},{st['ci'][0]:.4f},{st['ci'][1]:.4f},"
                        f"{ov['point']:.4f},{ov['ci'][0]:.4f},{ov['ci'][1]:.4f}")
        (PIC / f"v4_auroc_sel_{model}.csv").write_text("\n".join(rows) + "\n")


def decision() -> None:
    for model in ("gemma", "qwen"):
        src = RES / f"decision_auroc_{model}.json"
        if not src.exists():
            continue
        r = json.loads(src.read_text())
        rows = ["t,auroc,lo,hi"] + [f"{t},{r[k]['auroc']:.4f},{r[k]['ci'][0]:.4f},{r[k]['ci'][1]:.4f}" for k, t in STAGES]
        (PIC / f"v4_decision_{model}.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    law()
    law_qwen()
    auroc_sel()
    decision()
