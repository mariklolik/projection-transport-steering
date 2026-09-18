# Experiment 1 — detector-vector comparison across extraction methods.
#
# For each of the five confidence-measurement methods we now have a balanced
# dataset (exp1_directions). This module answers "how well can each APPROACH
# read overconfidence back out?" — the DETECTOR half of the experiment — and
# compares three families on the same per-method data:
#
#   CAA   : project activations onto the diff-in-means direction (mass-mean readout);
#   probe : a trained logistic detector (dense, held-out CV AUROC);
#   SAE   : the single best Gemma-Scope feature (1-sparse readout).
#
# Then two structural analyses:
#   - TRANSFER matrix: detector built from method i's labels, scored on method j's
#     labels (does one method's confidence direction detect another's?);
#   - COSINE matrix of the five CAA steering vectors (are the per-method vectors
#     actually different, or the same axis relabeled? — the §3 result, per method).
#
# CPU-only. Reads results/exp1/dataset.json (the balanced-set ids per method),
# results/features/behavioral_acts.pt (dense activations), and — if present —
# results/features/sae_behavioral_feats.pt (SAE features). Writes
# results/exp1/detectors.json. `python -m ...exp1_detectors [--layer 14]`.

from __future__ import annotations

import numpy as np
import torch

from behaviour_specific.overconfidence.exp1_directions import DIRECTIONS_DIR, METHODS
from behaviour_specific.overconfidence.features_caa_behavioral import BEHAVIORAL_ACTS, CONFIDENT_STATES
from general.paths import RESULTS_DIR
from general.storage import read_json, write_json

SAE_FEATS = RESULTS_DIR / "features" / "sae_behavioral_feats.pt"


def projection_auroc(acts: torch.Tensor, direction: torch.Tensor, y: np.ndarray) -> float:
    """AUROC of the raw projection onto `direction` as a confidence score."""
    from sklearn.metrics import roc_auc_score

    v = direction / direction.norm()
    return float(roc_auc_score(y, (acts @ v).numpy()))


def best_single_feature_auroc(feats: torch.Tensor, y: np.ndarray) -> tuple[int, float]:
    """(feature index, AUROC) for the single SAE feature that best separates y."""
    from sklearn.metrics import roc_auc_score

    x = feats.numpy()
    aucs = np.array([roc_auc_score(y, x[:, j]) for j in range(x.shape[1])])
    j = int(np.argmax(np.abs(aucs - 0.5)))     # strongest separation either polarity
    return j, float(aucs[j])


def _selftest():
    torch.manual_seed(0)
    n, d = 60, 8
    y = np.array([1] * 30 + [0] * 30)
    v = torch.randn(d)
    acts = torch.stack([(3 if i < 30 else -3) * v + 0.1 * torch.randn(d) for i in range(n)])
    assert projection_auroc(acts, v, y) > 0.95
    feats = torch.rand(n, 5)
    feats[:, 2] += torch.tensor(y * 3.0)
    j, a = best_single_feature_auroc(feats, y)
    assert j == 2 and a > 0.95
    print("exp1_detectors self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="gemmascope")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    _selftest()

    manifest = read_json(RESULTS_DIR / "exp1" / "dataset.json")
    L = args.layer
    blob = torch.load(BEHAVIORAL_ACTS)
    acts_by_id = {it["id"]: blob["acts"][k].float() for k, it in enumerate(blob["items"])}

    sae = None
    if SAE_FEATS.exists():
        sblob = torch.load(SAE_FEATS)
        sae_layers = sorted(sblob["feats"])
        sae_L = min(sae_layers, key=lambda x: abs(x - L))     # nearest available SAE layer
        sae = {it["id"]: sblob["feats"][sae_L][k].float() for k, it in enumerate(sblob["items"])}
        print(f"SAE features present: layers {sae_layers}, using L{sae_L} for the L{L} comparison")

    # per-method detector families
    caa_dirs = {mk: torch.load(DIRECTIONS_DIR / f"exp1_caa_{mk}.pt")["directions"][L] for mk in METHODS}
    dettab = {}
    print(f"\n=== detector AUROC @ L{L} (confident vs non-confident, balanced per method) ===")
    print(f"{'method':8s} {'n/pole':>7} {'CAA proj':>9} {'probe CV':>9} {'SAE 1-feat':>11} {'SAE feat#':>10}")
    for mk in METHODS:
        m = manifest["methods"][mk]
        ids = m["pos_ids"] + m["neg_ids"]
        y = np.array([1] * len(m["pos_ids"]) + [0] * len(m["neg_ids"]))
        acts = torch.stack([acts_by_id[i] for i in ids])[:, L]
        caa_auc = projection_auroc(acts, caa_dirs[mk], y)
        probe_auc = torch.load(DIRECTIONS_DIR / f"exp1_probe_{mk}.pt")["aurocs"][L]
        row = {"balanced_n": m["balanced_n"], "caa_proj_auroc": round(caa_auc, 4),
               "probe_cv_auroc": round(probe_auc, 4)}
        sae_str = "     -"
        if sae is not None:
            feats = torch.stack([sae[i] for i in ids])
            j, sae_auc = best_single_feature_auroc(feats, y)
            row["sae_best_feature"] = j
            row["sae_best_auroc"] = round(sae_auc, 4)
            sae_str = f"{sae_auc:.3f}"
            sae_feat = f"{j}"
        else:
            sae_feat = "-"
        dettab[mk] = row
        print(f"{mk:8s} {m['balanced_n']:>7} {caa_auc:>9.3f} {probe_auc:>9.3f} {sae_str:>11} {sae_feat:>10}")

    # transfer: CAA direction from method i, scored on method j's balanced labels
    print(f"\n=== transfer AUROC @ L{L}  (rows: detector method; cols: label method) ===")
    print(f"{'':6}" + "".join(f"{mk:>8}" for mk in METHODS))
    transfer = {}
    for di in METHODS:
        transfer[di] = {}
        cells = []
        for dj in METHODS:
            mj = manifest["methods"][dj]
            ids = mj["pos_ids"] + mj["neg_ids"]
            y = np.array([1] * len(mj["pos_ids"]) + [0] * len(mj["neg_ids"]))
            acts = torch.stack([acts_by_id[i] for i in ids])[:, L]
            a = projection_auroc(acts, caa_dirs[di], y)
            transfer[di][dj] = round(a, 4)
            cells.append(f"{a:>8.3f}")
        print(f"{di:6}" + "".join(cells))

    # cosine matrix of the five steering vectors
    print(f"\n=== cosine between the five CAA steering vectors @ L{L} ===")
    print(f"{'':6}" + "".join(f"{mk:>8}" for mk in METHODS))
    cosmat = {}
    for a in METHODS:
        cosmat[a] = {}
        cells = []
        for b in METHODS:
            c = float(torch.nn.functional.cosine_similarity(caa_dirs[a], caa_dirs[b], dim=0))
            cosmat[a][b] = round(c, 4)
            cells.append(f"{c:>+8.3f}")
        print(f"{a:6}" + "".join(cells))

    write_json(RESULTS_DIR / "exp1" / "detectors.json",
               {"layer": L, "detectors": dettab, "transfer_auroc": transfer, "cosine_matrix": cosmat})
    print(f"\nsaved {RESULTS_DIR / 'exp1' / 'detectors.json'}")
