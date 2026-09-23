from __future__ import annotations

import argparse

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR
from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.label_pool import split_records
from general.paths import RESULTS_DIR
from general.storage import write_json

CELLS = {"removed": (OCW, lambda s: s != OCW), "missed": (OCW, lambda s: s == OCW),
         "kept": (CR, lambda s: s == CR), "lost": (CR, lambda s: s != CR)}


def view(r: dict) -> dict:
    return {"answer": r["final_answer"], "p_yes": round(r["confidence"], 3), "state": r["state"],
            "trace": r["generations"][0]["text"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="v4_confirm")
    ap.add_argument("--arm", default="det_q60_alpha-0.375")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--out", default="v4_pooled/examples_gemma.json")
    args = ap.parse_args()

    runs = collect(RESULTS_DIR / args.dir / "rollouts")
    base = {r["id"]: r for r in runs["baseline_m5"]}
    after = {r["id"]: r for r in runs[f"{args.arm}_m5"]}
    recs = {r["id"]: r for r in split_records("confirm")}
    steered = sorted((i for i in after if after[i]["generations"][0]["text"] != base[i]["generations"][0]["text"]),
                     key=lambda i: int(i.split("-")[1]))
    out = {"rule": f"first {args.per_cell} steered questions by MMLU index in each cell", "arm": args.arm, "cells": {}}
    for cell, (start, test) in CELLS.items():
        ids = [i for i in steered if base[i]["state"] == start and test(after[i]["state"])][: args.per_cell]
        out["cells"][cell] = [{"id": i, "question": recs[i]["question"], "options": recs[i]["options"],
                               "gold": base[i]["gold"], "before": view(base[i]), "after": view(after[i])} for i in ids]
    out["counts"] = {cell: sum(base[i]["state"] == s and t(after[i]["state"]) for i in steered)
                     for cell, (s, t) in CELLS.items()}
    write_json(RESULTS_DIR / args.out, out)
    print(out["counts"])
