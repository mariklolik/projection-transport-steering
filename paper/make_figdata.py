# Figure data for paper/pic from the run outputs: the token-level projection
# CDFs, the per-benchmark behavioral planes, the measured before/after
# movements, and the label-oracle comparison.
#   python3 -m behaviour_specific.overconfidence.proj_dump   # writes results/viz
#   python3 paper/make_figdata.py

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "results" / "viz"
OUT = ROOT / "paper" / "pic" / "viz"
SHORT = {"overconfident_wrong": "ocw", "confident_right": "cr",
         "nonconfident_right": "ncr", "nonconfident_wrong": "ncw"}
CALIBRATED = {"confident_right", "nonconfident_wrong"}


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def dump(path: Path, header: str, lines: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(lines) + "\n")
    print(f"{path.name}: {len(lines)}")


def planes() -> dict[str, list[dict]]:
    out = {}
    for p in sorted(VIZ.glob("plane_*.csv")):
        bench = p.stem.split("_", 1)[1]
        rs = rows(p)
        out[bench] = rs
        for full, short in SHORT.items():
            sel = [r for r in rs if r["state"] == full]
            dump(OUT / f"pl_{bench}_{short}.csv", "x,y",
                 [f"{r['x']},{r['y']}" for r in sel])
        dump(OUT / f"plane_{bench}.csv", "x,y", [f"{r['x']},{r['y']}" for r in rs])
    return out


def movements() -> None:
    for p in sorted(VIZ.glob("moves_*.csv")):
        name = p.stem.split("_", 1)[1]
        rs = rows(p)
        for full, short in (("overconfident_wrong", "ocw"), ("confident_right", "cr")):
            sel = [r for r in rs if r["state"] == full]
            dump(OUT / f"cls_{name}_{short}_ar.csv", "x,y,u,v",
                 [f"{r['x']},{r['y']},{r['u']},{r['v']}" for r in sel if r["changed"] == "1"])
            dump(OUT / f"cls_{name}_{short}_still.csv", "x,y",
                 [f"{r['x']},{r['y']}" for r in sel if r["changed"] != "1"])


def oracle(tau: float, method: str = "ours_gatedabl") -> None:
    """Label oracle (every OCW to the calibrated centroid) against our measured moves."""
    plane = rows(VIZ / "plane_mmlu.csv")
    cal = [r for r in plane if r["state"] in CALIBRATED]
    cx = sum(float(r["x"]) for r in cal) / len(cal)
    cy = sum(float(r["y"]) for r in cal) / len(cal)
    mv = rows(VIZ / f"moves_{method}.csv")
    ocw = [r for r in mv if r["state"] == "overconfident_wrong"]
    cr = [r for r in mv if r["state"] == "confident_right"]
    dump(OUT / "orc_oracle.csv", "x,y,u,v",
         [f"{r['x']},{r['y']},{cx - float(r['x']):.2f},{cy - float(r['y']):.2f}" for r in ocw])
    hit = [r for r in ocw if float(r["y"]) > tau]
    dump(OUT / "orc_hits.csv", "x,y,u,v", [f"{r['x']},{r['y']},{r['u']},{r['v']}" for r in hit])
    dump(OUT / "orc_miss.csv", "x,y", [f"{r['x']},{r['y']}" for r in ocw if float(r["y"]) <= tau])
    dump(OUT / "orc_ff.csv", "x,y,u,v",
         [f"{r['x']},{r['y']},{r['u']},{r['v']}" for r in cr if float(r["y"]) > tau])
    print(f"oracle: centroid ({cx:.2f},{cy:.2f}) tau {tau:.2f} "
          f"caught {len(hit)}/{len(ocw)} false-fires {sum(float(r['y']) > tau for r in cr)}/{len(cr)}")


if __name__ == "__main__":
    import json
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else \
        json.loads((ROOT / "results" / "viz" / "tau.json").read_text())["tau"]
    planes()
    movements()
    oracle(tau)
