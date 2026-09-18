# Full-d logistic probe gate + Linear-AcT per-neuron stats + MiMiC full-space
# moments, from the saved traces.
# Run: python -m behaviour_specific.overconfidence.extract_probe_gate --layer 14

from __future__ import annotations

import argparse

import numpy as np
import torch

from behaviour_specific.overconfidence.extract_projection_stats import auroc, token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from general.storage import read_jsonl, write_json

OCW = "overconfident_wrong"
CALIBRATED = {"confident_right", "nonconfident_wrong"}
QUANTILES = torch.linspace(0.01, 0.99, 41)


def fit_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, float]:
    """L2 logistic regression (standardized), 5-fold CV AUROC -> (w', b', cv_auroc).

    Returned (w', b') fold the standardization in, so score = w'·x + b' on RAW
    trace-mean activations.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, y):
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Xs[tr], y[tr])
        sc = clf.decision_function(Xs[te])
        aucs.append(auroc(sc.tolist(), y[te].tolist()))
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(Xs, y)
    w = (clf.coef_[0] / sd).astype(np.float32)
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    return w, b, float(np.mean(aucs))


def _selftest():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 8)), rng.normal(1.2, 1, (100, 8))])
    y = np.array([0] * 100 + [1] * 100)
    w, b, cv = fit_probe(X, y)
    assert cv > 0.85, cv
    sc = X @ w + b
    assert auroc(sc.tolist(), y.tolist()) > 0.85
    print("extract_probe_gate self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max-tokens-per-trace", type=int, default=120,
                    help="subsample per-token stats collection (memory bound)")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    rows = read_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.layer}.jsonl")
    print(f"{len(rows)} saved traces")

    model, tok = load_model()
    pooled, tok_src, tok_tgt, states = [], [], [], []
    for i, r in enumerate(rows):
        _, ht = token_projections(model, tok, r["prompt"], r["trace"], args.layer)
        pooled.append(ht.mean(0))
        idx = torch.randperm(ht.shape[0])[: args.max_tokens_per_trace]
        tok_src.append(ht[idx].half())
        if r["m4_state"] in CALIBRATED:
            tok_tgt.append(ht[idx].half())
        states.append(r["m4_state"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    pooled = torch.stack(pooled)                       # [n, d] fp32
    src = torch.cat(tok_src).float()                   # [~48k, d]
    tgt = torch.cat(tok_tgt).float()

    # 1) probe gate
    y = np.array([1 if s == OCW else 0 for s in states])
    w, b, cv_auc = fit_probe(pooled.numpy(), y)
    scores = pooled.numpy() @ w + b
    is_cr = np.array([s == "confident_right" for s in states])
    both = (y == 1) | is_cr
    auc_cr = auroc(scores[both].tolist(), y[both].tolist())
    print(f"probe gate: CV AUROC(OCW vs rest) = {cv_auc:.3f}; in-sample OCW-vs-CR = {auc_cr:.3f}")
    sc_t = torch.tensor(scores)
    torch.save({"w": torch.tensor(w), "b": b, "cv_auroc_ocw_vs_rest": round(cv_auc, 4),
                "auroc_ocw_vs_cr_insample": round(auc_cr, 4),
                "score_quantiles": {"all": sc_t.quantile(QUANTILES),
                                    "cr": sc_t[torch.tensor(is_cr)].quantile(QUANTILES)}},
               DIRECTIONS_DIR / f"probe_gate_L{args.layer}.pt")

    # 2) Linear-AcT per-neuron stats
    torch.save({"mu_s": src.mean(0), "sig_s": src.std(0).clamp(min=1e-4),
                "mu_t": tgt.mean(0), "sig_t": tgt.std(0).clamp(min=1e-4),
                "n_src": src.shape[0], "n_tgt": tgt.shape[0]},
               DIRECTIONS_DIR / f"perneuron_stats_L{args.layer}.pt")

    # 3) MiMiC full-space moments
    torch.save({"m_s": src.mean(0), "S_s": torch.cov(src.T), "m_t": tgt.mean(0),
                "S_t": torch.cov(tgt.T)},
               DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")

    write_json(DIRECTIONS_DIR / f"probe_gate_L{args.layer}_summary.json",
               {"cv_auroc_ocw_vs_rest": round(cv_auc, 4),
                "auroc_ocw_vs_cr_insample": round(auc_cr, 4),
                "ceiling_selectivity_2auc_minus_1": round(2 * cv_auc - 1, 4),
                "n_src_tokens": int(src.shape[0]), "n_tgt_tokens": int(tgt.shape[0])})
    print("saved probe_gate / perneuron_stats / fullspace_moments")
