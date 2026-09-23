from __future__ import annotations

import argparse
import json

import numpy as np

from behaviour_specific.overconfidence.analyze_pooled import CR, OCW, ci, pooled_rows
from general.paths import RESULTS_DIR
from general.storage import write_json

BINS = 10


def ece(w: np.ndarray, c: np.ndarray, y: np.ndarray) -> np.ndarray:
    b = np.minimum((c * BINS).astype(int), BINS - 1)
    gap = np.stack([(w * (c - y) * (b == k)).sum(-1) for k in range(BINS)])
    return np.abs(gap).sum(0) / w.sum(-1)


def measures(w: np.ndarray, base: dict, rows: dict) -> dict[str, np.ndarray]:
    m = lambda x: (w * x).sum(-1) / w.sum(-1)  # noqa: E731
    wrong, right = 1 - rows["y"], rows["y"]
    return {"sel": (w * base["ocw"] * (1 - rows["ocw"])).sum(-1) / (w * base["ocw"]).sum(-1)
                   - (w * base["cr"] * (1 - rows["cr"])).sum(-1) / (w * base["cr"]).sum(-1),
            "d_ocw": m(rows["ocw"]) - m(base["ocw"]),
            "d_conf_wrong": (w * wrong * rows["conf"]).sum(-1) / (w * wrong).sum(-1)
                            - (w * (1 - base["y"]) * base["conf"]).sum(-1) / (w * (1 - base["y"])).sum(-1),
            "d_conf_right": (w * right * rows["conf"]).sum(-1) / (w * right).sum(-1)
                            - (w * base["y"] * base["conf"]).sum(-1) / (w * base["y"]).sum(-1),
            "d_ece": ece(w, rows["c"], rows["y"]) - ece(w, base["c"], base["y"]),
            "d_brier": m((rows["c"] - rows["y"]) ** 2) - m((base["c"] - base["y"]) ** 2),
            "dacc": m(rows["y"]) - m(base["y"]),
            "d_forced": m(rows["forced"]) - m(base["forced"])}


def arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    c = np.array([r["confidence"] for r in rows], float)
    return {"ocw": np.array([r["state"] == OCW for r in rows], float), "cr": np.array([r["state"] == CR for r in rows], float),
            "y": np.array([r["is_correct"] for r in rows], float), "c": c, "conf": (c >= 0.5).astype(float),
            "forced": np.array([r.get("forced_box") in (True, "True") for r in rows], float)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = json.loads((RESULTS_DIR / args.dir / "protocol.json").read_text())
    ref, cfg = proto["ref"], proto["ref"][len("det_"):]
    nulls = {"gated_null": f"detnull_{cfg}", "gated_random": f"detrand_{cfg}", "ungated_null": "plain_null"}
    conds = [a["tag"] for a in proto["arms"]] + list(nulls.values())
    base, after = pooled_rows([args.dir], "m5", conds)
    n = len(base)
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=args.iters).astype(float)
    b = arrays(base)
    one = {c: measures(np.ones((1, n)), b, arrays(r)) for c, r in after.items()}
    bs = {c: measures(w, b, arrays(r)) for c, r in after.items()}
    res = {"n": n, "ref": ref, "nulls": nulls, "arms": {}}
    for c in conds:
        row = {k: {"point": round(float(one[c][k][0]), 4), "ci": ci(bs[c][k])} for k in one[c]}
        keep = 1 - arrays(after[c])["forced"]
        row["sel_unforced"] = round(float(measures(keep[None], b, arrays(after[c]))["sel"][0]), 4)
        for label, null in (("net_gated_null", nulls["gated_null"]), ("net_ungated_null", nulls["ungated_null"]),
                            ("vs_ref", ref)):
            d = bs[c]["sel"] - bs[null]["sel"]
            row[label] = {"point": round(float(one[c]["sel"][0] - one[null]["sel"][0]), 4), "ci": ci(d),
                          "p_one_sided": round(float((d <= 0).mean()), 4)}
        res["arms"][c] = row
        print(f"{c:36s} sel {row['sel']['point']:+.3f} net {row['net_gated_null']['point']:+.3f} "
              f"{row['net_gated_null']['ci']} dOCW {row['d_ocw']['point']:+.4f} {row['d_ocw']['ci']} "
              f"dECE {row['d_ece']['point']:+.4f} {row['d_ece']['ci']} dBrier {row['d_brier']['point']:+.4f} "
              f"{row['d_brier']['ci']}", flush=True)
    write_json(RESULTS_DIR / args.out, res)
