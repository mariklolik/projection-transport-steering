from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "results" / "release" / "records.csv.gz"
OCW, CR = "overconfident_wrong", "confident_right"


def load(split: str) -> dict[str, dict[str, dict]]:
    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    with gzip.open(SRC, "rt") as f:
        for r in csv.DictReader(f):
            if r["split"] == split and r["readout"] == "m5":
                arms[r["arm"]][r["id"]] = r
    return arms


def selectivity(base: dict, after: dict, ids: list[str], w: np.ndarray) -> np.ndarray:
    ocw = np.array([base[i]["state"] == OCW for i in ids], float)
    cr = np.array([base[i]["state"] == CR for i in ids], float)
    left = np.array([after[i]["state"] != OCW for i in ids], float)
    kept = np.array([after[i]["state"] == CR for i in ids], float)
    return (w * ocw * left).sum(-1) / (w * ocw).sum(-1) - (w * cr * (1 - kept)).sum(-1) / (w * cr).sum(-1)


if __name__ == "__main__":
    split, ref = sys.argv[1], sys.argv[2]
    arms = load(split)
    base = arms.pop("baseline")
    ids = sorted(i for i in base if all(i in a for a in arms.values()))
    w = np.random.default_rng(0).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=10000).astype(float)
    one = np.ones(len(ids))
    ref_bs = selectivity(base, arms[ref], ids, w)
    for name, a in sorted(arms.items()):
        pt, bs = float(selectivity(base, a, ids, one)), selectivity(base, a, ids, w)
        d = bs - ref_bs
        print(f"{name:40s} Sel {pt:+.3f} [{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}] "
              f"vs {ref}: {float(selectivity(base, a, ids, one) - selectivity(base, arms[ref], ids, one)):+.3f} "
              f"p={2 * min((d <= 0).mean(), (d >= 0).mean()):.4f}")
