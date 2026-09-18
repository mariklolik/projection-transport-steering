# Cross-analysis of the extracted overconfidence directions (CPU-only, no model).
#
# Inputs (produced by features_caa / features_probe / features_caa_behavioral):
#   results/features/persona_acts.pt     pos/neg [n, n_layers, d] persona-trace activations
#   results/features/behavioral_acts.pt  acts [m, n_layers, d] + per-trace M2/M4 labels
#   directions/*.pt                      every saved candidate direction
#
# Questions answered:
#   1. WHERE is overconfidence linearly encoded on-policy? Per-layer CV-AUROC of
#      a logistic probe on the natural traces, under M2 / M4 / consensus labels.
#   2. Does the PERSONA proxy transfer on-policy? Project behavioral activations
#      onto the persona directions -> AUROC against behavioral labels (no
#      training), per layer. And the reverse (behavioral probe on persona acts).
#   3. Do different labeling methods give the SAME direction? Cosine between all
#      candidates, per layer.
#
# Also saves the on-policy M4-label probe as a steering candidate
# (directions/probe_behavioral_m4.pt). Writes results/features/analysis.json.
#
# `python -m behaviour_specific.overconfidence.analyze_features`

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR, PERSONA_ACTS
from behaviour_specific.overconfidence.features_caa_behavioral import BEHAVIORAL_ACTS, CONFIDENT_STATES
from general.storage import write_json

DIRECTION_NAMES = ["caa", "probe", "caa_conf_split", "caa_ocw_vs_cr", "caa_ocw_vs_rest",
                   "caa_m4_conf", "caa_consensus_conf", "probe_behavioral_m4"]


def cv_auroc(x: np.ndarray, y: np.ndarray) -> float:
    """5-fold cross-validated AUROC of a logistic probe on (x, y)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    proba = cross_val_predict(LogisticRegression(max_iter=1000), x, y, cv=5,
                              method="predict_proba", n_jobs=-1)[:, 1]
    return float(roc_auc_score(y, proba))


def fit_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Full-data logistic probe weight vector, unit norm."""
    from sklearn.linear_model import LogisticRegression

    w = LogisticRegression(max_iter=1000).fit(x, y).coef_[0]
    return w / (np.linalg.norm(w) + 1e-8)


def probe_curve(acts: torch.Tensor, y: np.ndarray) -> tuple[list[float], torch.Tensor]:
    """CV-AUROC per layer + full-fit unit probe directions [n_layers, d]."""
    aurocs, dirs = [], []
    for layer in range(acts.shape[1]):
        x = acts[:, layer].float().numpy()
        aurocs.append(cv_auroc(x, y))
        dirs.append(torch.tensor(fit_direction(x, y), dtype=torch.float32))
        print(f"  probe layer {layer:2d}: AUROC {aurocs[-1]:.3f}", flush=True)
    return aurocs, torch.stack(dirs)


def transfer_curve(acts: torch.Tensor, y: np.ndarray, dirs: torch.Tensor) -> list[float]:
    """AUROC of the raw projection onto dirs[layer], per layer (no training).

    >0.5 means the direction, learned elsewhere, still separates these labels
    with the SAME sign; <0.5 means it is anti-aligned.
    """
    from sklearn.metrics import roc_auc_score

    out = []
    for layer in range(acts.shape[1]):
        v = dirs[layer] / dirs[layer].norm()
        proj = (acts[:, layer].float() @ v).numpy()
        out.append(float(roc_auc_score(y, proj)))
    return out


def cosine_curves(directions: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """cos(a[layer], b[layer]) for every direction pair, per layer."""
    out = {}
    for a, b in combinations(directions, 2):
        da, db = directions[a], directions[b]
        cos = torch.nn.functional.cosine_similarity(da, db, dim=-1)
        out[f"{a}|{b}"] = [round(c, 4) for c in cos.tolist()]
    return out


def _selftest():
    rng = np.random.default_rng(0)
    n, d = 80, 10
    w = rng.normal(size=d)
    y = np.array([1] * n + [0] * n)
    signal = np.concatenate([rng.normal(size=(n, d)) + 2 * w, rng.normal(size=(n, d)) - 2 * w])
    noise = rng.normal(size=(2 * n, d))
    acts = torch.tensor(np.stack([signal, noise], axis=1), dtype=torch.float32)

    aurocs, dirs = probe_curve(acts, y)
    assert aurocs[0] > 0.95 and aurocs[1] < 0.75
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(2), atol=1e-4)

    true_dirs = torch.tensor(np.stack([w, w]), dtype=torch.float32)
    t = transfer_curve(acts, y, true_dirs)
    assert t[0] > 0.95 and abs(t[1] - 0.5) < 0.15

    cc = cosine_curves({"a": dirs, "b": true_dirs})
    assert cc["a|b"][0] > 0.8
    print("analyze_features self-test passed")


if __name__ == "__main__":
    _selftest()

    if not (PERSONA_ACTS.exists() and BEHAVIORAL_ACTS.exists()):
        raise SystemExit("need persona_acts.pt + behavioral_acts.pt "
                         "(run features_caa and features_caa_behavioral --rollouts first)")

    pa = torch.load(PERSONA_ACTS)
    pos, neg = pa["pos"], pa["neg"]
    ba = torch.load(BEHAVIORAL_ACTS)
    bacts, items = ba["acts"].float(), ba["items"]
    n_layers = bacts.shape[1]
    print(f"persona acts: {tuple(pos.shape)} x2   behavioral acts: {tuple(bacts.shape)}")

    m2y = np.array([it["m2_state"] in CONFIDENT_STATES for it in items], dtype=int)
    has4 = np.array([it["m4_state"] is not None for it in items])
    m4y = np.array([it["m4_state"] in CONFIDENT_STATES if it["m4_state"] else False
                    for it in items], dtype=int)
    agree = has4 & (m2y == m4y)
    print(f"labels: m2 conf {m2y.sum()}/{len(m2y)}   m4 conf {m4y[has4].sum()}/{has4.sum()}   "
          f"consensus subset {agree.sum()} (conf {m2y[agree].sum()})")

    persona_y = np.array([1] * len(pos) + [0] * len(neg))
    persona_acts = torch.cat([pos, neg])

    probe_pt = DIRECTIONS_DIR / "probe.pt"
    if probe_pt.exists():
        persona_aurocs = torch.load(probe_pt)["aurocs"]
        print("\n[1/4] persona probe curve: loaded from probe.pt")
    else:
        print("\n[1/4] persona probe curve")
        persona_aurocs, _ = probe_curve(persona_acts, persona_y)
    print("[2/4] behavioral probe curve, M2 labels")
    m2_aurocs, _ = probe_curve(bacts, m2y)
    print("[3/4] behavioral probe curve, M4 labels")
    m4_aurocs, m4_probe_dirs = probe_curve(bacts[has4], m4y[has4])
    print("[4/4] behavioral probe curve, consensus labels")
    cons_aurocs, _ = probe_curve(bacts[agree], m2y[agree])

    torch.save({"directions": m4_probe_dirs, "method": "probe_behavioral_m4",
                "aurocs": m4_aurocs}, DIRECTIONS_DIR / "probe_behavioral_m4.pt")

    directions = {}
    for name in DIRECTION_NAMES:
        p = DIRECTIONS_DIR / f"{name}.pt"
        if p.exists():
            directions[name] = torch.load(p)["directions"]

    transfers = {
        "caa->m4": transfer_curve(bacts[has4], m4y[has4], directions["caa"]),
        "caa->m2": transfer_curve(bacts, m2y, directions["caa"]),
        "probe->m4": transfer_curve(bacts[has4], m4y[has4], directions["probe"]),
        "beh_probe->persona": transfer_curve(persona_acts, persona_y, m4_probe_dirs),
    }

    focus = int(np.argmax(m4_aurocs))
    norms = bacts.norm(dim=-1).mean(0)
    print(f"\n{'layer':>5} {'persona':>8} {'beh-m2':>7} {'beh-m4':>7} {'beh-cons':>8} "
          f"{'caa->m4':>8} {'probe->m4':>9} {'||h||':>7}")
    for layer in range(n_layers):
        mark = "  <- focus" if layer == focus else ""
        print(f"{layer:>5} {persona_aurocs[layer]:>8.3f} {m2_aurocs[layer]:>7.3f} {m4_aurocs[layer]:>7.3f} "
              f"{cons_aurocs[layer]:>8.3f} {transfers['caa->m4'][layer]:>8.3f} "
              f"{transfers['probe->m4'][layer]:>9.3f} {norms[layer]:>7.1f}{mark}")

    cosines = cosine_curves(directions)
    print(f"\ncosines @ focus layer {focus}:")
    for pair, curve in cosines.items():
        print(f"  {pair:44s} {curve[focus]:+.3f}")

    out = {
        "n_persona": int(len(pos)), "n_behavioral": int(len(items)),
        "labels": {"m2_conf": int(m2y.sum()), "m4_conf": int(m4y[has4].sum()),
                   "has_m4": int(has4.sum()), "consensus_n": int(agree.sum()),
                   "consensus_conf": int(m2y[agree].sum())},
        "probe_auroc": {"persona": persona_aurocs, "behavioral_m2": m2_aurocs,
                        "behavioral_m4": m4_aurocs, "behavioral_consensus": cons_aurocs},
        "transfer_auroc": transfers,
        "cosine_curves": cosines,
        "mean_act_norm": [round(v, 2) for v in norms.tolist()],
        "focus_layer": focus,
    }
    write_json(BEHAVIORAL_ACTS.parent / "analysis.json", out)
    print(f"\nsaved {BEHAVIORAL_ACTS.parent / 'analysis.json'}")
