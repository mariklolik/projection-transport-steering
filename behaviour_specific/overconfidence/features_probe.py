# Linear probe for overconfidence, per layer — on REASONING traces.
#
# HYPOTHESIS: if overconfidence is linearly encoded in the reasoning process, a
# logistic-regression probe on trace-pooled activations can separate traces
# written under the over-confident persona from calibrated ones. Payoffs:
#   1. per-layer AUROC = "is it linearly encoded, and where?" (picks BEST_LAYER);
#   2. the probe weight vector is a second candidate steering direction (compared
#      to the CAA direction by cosine).
#
# Reuses results/features/persona_acts.pt saved by features_caa (no model, no
# regeneration) when it exists; otherwise collects activations itself.
#
# `python -m behaviour_specific.overconfidence.features_probe [--n N] [--seed S]`

from __future__ import annotations

import numpy as np
import torch

from behaviour_specific.overconfidence.features_caa import (
    DIRECTIONS_DIR, PERSONA_ACTS, collect_pole_activations, extraction_records,
)
from behaviour_specific.overconfidence.personas import CALIBRATED_PERSONA, OVERCONFIDENT_PERSONA


def probe_layer(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Train logistic regression on one layer's activations.

    Returns (cross-val AUROC, unit weight vector). x: [n, d], y: [n] in {0,1}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    clf = LogisticRegression(max_iter=1000, C=1.0)
    proba = cross_val_predict(clf, x, y, cv=5, method="predict_proba")[:, 1]  # honest OOS AUROC
    auroc = float(roc_auc_score(y, proba))
    clf.fit(x, y)
    w = clf.coef_[0]
    return auroc, w / (np.linalg.norm(w) + 1e-8)


def probe_all_layers(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> tuple[list[float], torch.Tensor]:
    """Probe every layer. pos/neg: [n, n_layers, d]. Returns (aurocs, [n_layers, d] dirs)."""
    n_layers = pos_acts.shape[1]
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    aurocs, dirs = [], []
    for layer in range(n_layers):
        x = torch.cat([pos_acts[:, layer], neg_acts[:, layer]]).numpy()
        auroc, w = probe_layer(x, y)
        aurocs.append(auroc)
        dirs.append(torch.tensor(w, dtype=torch.float32))
    return aurocs, torch.stack(dirs)


def best_layer(aurocs: list[float]) -> int:
    """Layer with the highest separability."""
    return int(np.argmax(aurocs))


def _selftest():
    rng = np.random.default_rng(0)
    d, n = 12, 60
    w_true = rng.normal(size=d)
    pos0 = rng.normal(size=(n, d)) + 3 * w_true
    neg0 = rng.normal(size=(n, d)) - 3 * w_true
    noise = rng.normal(size=(2 * n, d))
    pos = torch.tensor(np.stack([pos0, noise[:n]], axis=1), dtype=torch.float32)
    neg = torch.tensor(np.stack([neg0, noise[n:]], axis=1), dtype=torch.float32)
    aurocs, dirs = probe_all_layers(pos, neg)
    assert aurocs[0] > 0.95 and aurocs[1] < 0.75, aurocs
    assert best_layer(aurocs) == 0
    print("features_probe self-test passed  (aurocs:", [round(a, 2) for a in aurocs], ")")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    _selftest()

    if PERSONA_ACTS.exists():
        blob = torch.load(PERSONA_ACTS)
        pos, neg = blob["pos"], blob["neg"]
        print(f"loaded pole activations from {PERSONA_ACTS} (n={len(pos)})")
    else:
        from models_specific.active import load_model

        model, tok = load_model()
        records = extraction_records(args.n, args.seed)
        pos = collect_pole_activations(model, tok, records, OVERCONFIDENT_PERSONA)
        neg = collect_pole_activations(model, tok, records, CALIBRATED_PERSONA)
    aurocs, dirs = probe_all_layers(pos, neg)

    bl = best_layer(aurocs)
    print("AUROC by layer:")
    for i, a in enumerate(aurocs):
        print(f"  layer {i:2d}: {a:.3f}{'  <- best' if i == bl else ''}")

    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIRECTIONS_DIR / "probe.pt"
    torch.save({"directions": dirs, "method": "probe", "aurocs": aurocs, "best_layer": bl}, path)
    print(f"saved probe directions -> {path} (best_layer={bl}, AUROC={aurocs[bl]:.3f})")

    caa_path = DIRECTIONS_DIR / "caa.pt"
    if caa_path.exists():
        import torch.nn.functional as F
        caa = torch.load(caa_path)["directions"][bl]
        print(f"cos(probe, CAA) @ layer {bl} = {F.cosine_similarity(dirs[bl], caa, dim=0):.3f}")
