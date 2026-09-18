# Joint 2-D stats on V = span{m4_conf, ocw_vs_cr}: token-level mean+cov per
# group (BW transport) and an LDA gate on trace-mean projections (NP-optimal
# detector). Reads trace_projections_L<layer>.jsonl; forward passes only.
# Run: python -m behaviour_specific.overconfidence.extract_joint_stats --layer 14

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from behaviour_specific.overconfidence.extract_projection_stats import auroc, token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from general.steering import orthonormalize
from general.storage import read_jsonl, write_json

CALIBRATED = {"confident_right", "nonconfident_wrong"}
OCW = "overconfident_wrong"
QUANTILES = torch.linspace(0.01, 0.99, 41)


def lda_fit(x_pos: torch.Tensor, x_neg: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Two-class LDA: w = Σ_pooled⁻¹ (μ+ − μ−), b so that scores are centered.

    Under equal-covariance Gaussians the LDA score is a monotone function of the
    likelihood ratio, hence the Neyman–Pearson-optimal statistic.
    """
    mu_p, mu_n = x_pos.mean(0), x_neg.mean(0)
    n_p, n_n = len(x_pos), len(x_neg)
    S = ((n_p - 1) * torch.cov(x_pos.T) + (n_n - 1) * torch.cov(x_neg.T)) / (n_p + n_n - 2)
    w = torch.linalg.solve(S + 1e-6 * torch.eye(S.shape[0]), mu_p - mu_n)
    b = -float(w @ (mu_p + mu_n) / 2)
    return w, b


def _selftest():
    torch.manual_seed(0)
    xp = torch.randn(200, 2) + torch.tensor([2.0, 0.0])
    xn = torch.randn(200, 2)
    w, b = lda_fit(xp, xn)
    sc = torch.cat([xp @ w + b, xn @ w + b])
    lb = [1] * 200 + [0] * 200
    assert auroc(sc.tolist(), lb) > 0.9
    assert (xp @ w + b).mean() > 0 > (xn @ w + b).mean()
    print("extract_joint_stats self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--dirs", default="m4_conf,ocw_vs_cr", help="2 direction names spanning V")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    names = args.dirs.split(",")
    V = orthonormalize(torch.stack([pts["dirs"][n] for n in names]).float())

    # read_jsonl (splits on \n only): traces may contain raw U+2028/NEL that
    # str.splitlines() would treat as line breaks, corrupting records.
    rows = read_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.layer}.jsonl")
    print(f"{len(rows)} saved traces; V = span{{{args.dirs}}} (orthonormalized)")

    model, tok = load_model()
    tok_by_group: dict[str, list] = defaultdict(list)
    trace_means, states = [], []
    for i, r in enumerate(rows):
        _, ht = token_projections(model, tok, r["prompt"], r["trace"], args.layer)
        q = ht @ V.T                        # [n_tokens, 2]
        st = r["m4_state"]
        tok_by_group["src_all"].append(q)
        if st in CALIBRATED:
            tok_by_group["tgt_calibrated"].append(q)
        if st != OCW:
            tok_by_group["tgt_non_ocw"].append(q)
        tok_by_group[f"state_{st}"].append(q)
        trace_means.append(q.mean(0))
        states.append(st)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    token_stats = {}
    for g, chunks in tok_by_group.items():
        x = torch.cat(chunks)
        token_stats[g] = {"m": x.mean(0), "S": torch.cov(x.T), "n": int(x.shape[0])}
        print(f"{g:22s} n={token_stats[g]['n']:>6} m={token_stats[g]['m'].tolist()}")

    X = torch.stack(trace_means)
    is_ocw = torch.tensor([s == OCW for s in states])
    is_cr = torch.tensor([s == "confident_right" for s in states])
    w, b = lda_fit(X[is_ocw], X[~is_ocw])
    scores = (X @ w + b)
    a_rest = auroc(scores.tolist(), is_ocw.int().tolist())
    both = is_ocw | is_cr
    a_cr = auroc(scores[both].tolist(), is_ocw[both].int().tolist())
    print(f"LDA gate: AUROC OCW-vs-rest = {a_rest:.3f}, OCW-vs-CR = {a_cr:.3f}")

    gate = {"w": w, "b": b, "auroc_ocw_vs_rest": round(a_rest, 4), "auroc_ocw_vs_cr": round(a_cr, 4),
            "score_quantiles": {
                "all": scores.quantile(QUANTILES),
                "cr": scores[is_cr].quantile(QUANTILES),
                "non_ocw": scores[~is_ocw].quantile(QUANTILES)}}
    torch.save({"layer": args.layer, "dir_names": names, "V": V,
                "token": token_stats, "gate_lda": gate},
               DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    write_json(DIRECTIONS_DIR / f"joint_stats_L{args.layer}_summary.json",
               {"dirs": names, "gate_auroc_ocw_vs_rest": round(a_rest, 4),
                "gate_auroc_ocw_vs_cr": round(a_cr, 4),
                "groups": {g: s["n"] for g, s in token_stats.items()}})
    print(f"saved -> {DIRECTIONS_DIR}/joint_stats_L{args.layer}.pt")
