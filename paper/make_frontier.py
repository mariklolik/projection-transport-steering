from __future__ import annotations

import csv
from pathlib import Path

from recompute_from_records import CR, OCW, load

PIC = Path(__file__).resolve().parent / "pic"


def group(arm: str, ref: str) -> str | None:
    if arm == ref:
        return "pts"
    if arm.startswith(("detonline_", "detprompt_")):
        return "pts_onepass"
    if arm.startswith("rand") and arm.endswith(ref):
        return "random"
    if arm.startswith(("plain_alpha", "tuned_additive")):
        return "shift"
    if arm in ("plain_ablate", "tuned_mimic", "tuned_act", "prompt_hedge"):
        return "baseline"
    if arm.startswith("castdim_") or arm == "tuned_cast":
        return "cast"
    return None


if __name__ == "__main__":
    for split, ref in (("v4_confirm", "det_q60_alpha-0.375"), ("v6a_confirm", "det_q30_alpha-0.25")):
        arms = load(split)
        base = arms.pop("baseline")
        ocw = [i for i, r in base.items() if r["state"] == OCW]
        cr = [i for i, r in base.items() if r["state"] == CR]
        rows = []
        for arm, a in arms.items():
            g = group(arm, ref)
            if g is None or any(i not in a for i in ocw + cr):
                continue
            rm = sum(a[i]["state"] != OCW for i in ocw) / len(ocw)
            kp = sum(a[i]["state"] == CR for i in cr) / len(cr)
            rows.append({"arm": arm, "group": g, "removal": round(rm, 4), "retention": round(kp, 4), "sel": round(rm - 1 + kp, 4)})
        for g in {r["group"] for r in rows}:
            with open(PIC / f"frontier_{split}_{g}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["arm", "group", "removal", "retention", "sel"])
                w.writeheader()
                w.writerows(r for r in rows if r["group"] == g)
        print(split, {g: sum(r["group"] == g for r in rows) for g in {r["group"] for r in rows}})
        for r in sorted(rows, key=lambda r: -r["sel"])[:12]:
            print(r)
