from __future__ import annotations


import torch

from general.paths import DIRECTIONS_DIR
from behaviour_specific.overconfidence.actions import q_at


def gate_sigma(cov: torch.Tensor, w: torch.Tensor) -> float:
    return float((w @ cov.float() @ w).clamp(min=1e-12).sqrt())


def load_gate(layer: int, tau_q: float, delta: float = 0.05, decide_at: int | None = None):
    fit = torch.load(DIRECTIONS_DIR / f"layerfit_L{layer}.pt")
    w, b = fit["gate_lda"]["w"].float(), float(fit["gate_lda"]["b"])
    tau = q_at({"q": fit["gate_lda"]["score_quantiles"]["all"].tolist()}, tau_q)
    cal_path = DIRECTIONS_DIR / f"gatecal_L{layer}.pt"
    sigma = gate_sigma(fit["S_tok"], w)
    if cal_path.exists():
        cal = torch.load(cal_path)
        key = min(cal["sigma_eff"], key=lambda k: abs(float(k) - delta))
        sigma = float(cal["sigma_eff"][key])
        if decide_at is not None:
            heads = cal["prefix_heads"]
            t = min(heads, key=lambda k: abs(int(k) - decide_at))
            w, b = heads[t]["w"].float(), float(heads[t]["b"])
            tau = q_at({"q": heads[t]["score_quantiles"]["all"].tolist()}, tau_q)
    elif decide_at is not None:
        raise SystemExit(f"no gatecal_L{layer}.pt — run calibrate_gate first")
    return fit["V"].float(), w, b, float(tau), sigma


def _selftest():
    S = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    assert abs(gate_sigma(S, torch.tensor([1.0, 0.0])) - 2.0) < 1e-5
    print("steer_v6_online self-test passed")
