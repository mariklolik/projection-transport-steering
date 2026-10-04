from __future__ import annotations


import torch

from behaviour_specific.overconfidence.extract_projection_stats import auroc

CALIBRATED = {"confident_right", "nonconfident_wrong"}
OCW = "overconfident_wrong"
QUANTILES = torch.linspace(0.01, 0.99, 41)


def lda_fit(x_pos: torch.Tensor, x_neg: torch.Tensor) -> tuple[torch.Tensor, float]:

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
