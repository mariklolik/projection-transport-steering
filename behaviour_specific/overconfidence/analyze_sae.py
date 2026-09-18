# Cross-analysis of the SAE arm (CPU-only, no model).
#
# Connects the dictionary view to everything learned in §3:
#   1. OVERLAP — do the persona-contrast top features and the behavioral top
#      features agree? (§3 predicts: barely — the persona contrast selects
#      prompt features, the behavioral contrast selects the actual behavior.)
#   2. SINGLE-FEATURE AUROC — can ONE dictionary feature predict on-policy M4
#      confidence? The dictionary analogue of the §3 probe, but 1-sparse. Also
#      scores the top PERSONA feature on the same labels (feature-level
#      transfer test).
#   3. GEOMETRY — cos(top behavioral features' decoder dirs, caa_m4_conf) — is
#      the diff-means confidence axis aligned with named dictionary features?
#
# Reads results/features/sae_behavioral_feats.pt + the sae_*/sae_beh_* direction
# files. Writes results/sae_analysis.json.
#
# `python -m behaviour_specific.overconfidence.analyze_sae [--sae gemmascope]`

from __future__ import annotations

import numpy as np
import torch

from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.features_caa_behavioral import CONFIDENT_STATES
from general.paths import RESULTS_DIR
from general.storage import write_json

FEATS = RESULTS_DIR / "features" / "sae_behavioral_feats.pt"


def single_feature_auroc(col: np.ndarray, y: np.ndarray) -> float:
    """AUROC of one feature's activation as a confidence score."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, col))


def top_overlap(a: list[int], b: list[int]) -> int:
    """How many features two top-K lists share."""
    return len(set(a) & set(b))


def _selftest():
    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [0] * 40)
    f = rng.normal(size=(80, 10))
    f[:, 4] += 3 * y  # planted confidence feature
    assert single_feature_auroc(f[:, 4], y) > 0.95
    assert abs(single_feature_auroc(f[:, 0], y) - 0.5) < 0.2
    assert top_overlap([1, 2, 3], [3, 4, 1]) == 2
    print("analyze_sae self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="gemmascope")
    args = ap.parse_args()

    _selftest()

    if not FEATS.exists():
        raise SystemExit(f"{FEATS} missing — run features_sae --rollouts first")
    blob = torch.load(FEATS)
    feats, items = blob["feats"], blob["items"]
    has4 = np.array([it["m4_state"] is not None for it in items])
    m4y = np.array([it["m4_state"] in CONFIDENT_STATES if it["m4_state"] else False
                    for it in items], dtype=int)[has4]
    caa_m4 = torch.load(DIRECTIONS_DIR / "caa_m4_conf.pt")["directions"]
    print(f"{len(items)} traces, {int(m4y.sum())} M4-confident")

    rows = []
    print(f"\n{'layer':>6} {'overlap/20':>10} {'auc(top beh)':>13} {'best of 20':>11} "
          f"{'auc(top persona)':>17} {'cos(#1,m4dir)':>14} {'max|cos|':>9}")
    for L in sorted(feats):
        beh = torch.load(DIRECTIONS_DIR / f"sae_beh_{args.sae}_l{L}.pt")
        per_path = DIRECTIONS_DIR / f"sae_{args.sae}_l{L}.pt"
        per = torch.load(per_path) if per_path.exists() else None

        f = feats[L].float().numpy()[has4]
        aucs = [single_feature_auroc(f[:, i], m4y) for i in beh["feature_idxs"]]
        v = caa_m4[L] / caa_m4[L].norm()
        cos_all = beh["decoder_dirs"] @ v
        row = {"layer": L,
               "overlap_top20": top_overlap(beh["feature_idxs"], per["feature_idxs"]) if per else None,
               "auroc_top_beh_feature": aucs[0], "auroc_best_of_top20": max(aucs),
               "best_feature": int(beh["feature_idxs"][int(np.argmax(aucs))]),
               "auroc_top_persona_feature": single_feature_auroc(f[:, per["feature_idxs"][0]], m4y) if per else None,
               "cos_top1_to_m4dir": round(float(cos_all[0]), 4),
               "max_abs_cos_to_m4dir": round(float(cos_all.abs().max()), 4)}
        rows.append(row)
        ov = f"{row['overlap_top20']}" if per else "-"
        pa = f"{row['auroc_top_persona_feature']:.3f}" if per else "    -"
        print(f"{L:>6} {ov:>10} {row['auroc_top_beh_feature']:>13.3f} {row['auroc_best_of_top20']:>11.3f} "
              f"{pa:>17} {row['cos_top1_to_m4dir']:>+14.3f} {row['max_abs_cos_to_m4dir']:>9.3f}")

    write_json(RESULTS_DIR / "sae_analysis.json", {"sae": args.sae, "rows": rows})
    print(f"\nsaved {RESULTS_DIR / 'sae_analysis.json'}")
