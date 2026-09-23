from __future__ import annotations

import argparse
import csv
import gzip

from behaviour_specific.overconfidence.analyze_steering import collect
from general.paths import RESULTS_DIR

FIELDS = ("model", "split", "arm", "readout", "id", "gold", "answer", "is_correct", "confidence", "state", "m2_state")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for spec in args.dirs.split(","):
            model, d = spec.split(":")
            for tag, rows in sorted(collect(RESULTS_DIR / d / "rollouts").items()):
                arm, readout = tag.rsplit("_", 1)
                for r in rows:
                    w.writerow((model, d, arm, readout, r["id"], r.get("gold"), r.get("final_answer"), int(r["is_correct"]),
                                round(r["confidence"], 4), r["state"], r.get("m2_state", "")))
    print("->", out)
