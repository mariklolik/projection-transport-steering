from __future__ import annotations


import torch

from general.steering import (
    steer_ablate, steer_add, steer_clamp, steer_ot_gauss, steer_ot_quantile,
)

QGRID = torch.linspace(0.01, 0.99, 41)


def q_at(group: dict, q: float) -> float:

    idx = int(torch.argmin((QGRID - q).abs()))
    return group["q"][idx]


def build_conditions(stats: dict, norm: float) -> dict[str, callable]:

    src, cal, non = stats["src_all"], stats["tgt_calibrated"], stats["tgt_non_ocw"]
    src_q = torch.tensor(src["q"])
    conds = {
        "add_a-0.75": lambda v: (lambda h: steer_add(h, v, -0.75 * norm)),
        "ablate": lambda v: (lambda h: steer_ablate(h, v)),
        "clamp_q70": lambda v: (lambda h, t=q_at(src, 0.70): steer_clamp(h, v, t)),
        "clamp_q50": lambda v: (lambda h, t=q_at(src, 0.50): steer_clamp(h, v, t)),
        "clamp_q30": lambda v: (lambda h, t=q_at(src, 0.30): steer_clamp(h, v, t)),
    }
    if "q" in cal:
        conds["otg_cal"] = lambda v: (lambda h: steer_ot_gauss(h, v, src["mu"], src["sig"], cal["mu"], cal["sig"]))
        conds["otq_cal"] = lambda v: (lambda h, tq=torch.tensor(cal["q"]): steer_ot_quantile(h, v, src_q, tq))
    if "q" in non:
        conds["otq_nonocw"] = lambda v: (lambda h, tq=torch.tensor(non["q"]): steer_ot_quantile(h, v, src_q, tq))
    return conds


def _selftest():
    stats = {"src_all": {"q": torch.linspace(-2, 6, 41).tolist(), "mu": 2.0, "sig": 2.0},
             "tgt_calibrated": {"q": torch.linspace(-2, 2, 41).tolist(), "mu": 0.0, "sig": 1.0},
             "tgt_non_ocw": {"n": 5}}
    conds = build_conditions(stats, norm=100.0)
    assert "otq_cal" in conds and "otq_nonocw" not in conds
    v = torch.zeros(8); v[0] = 1.0
    h = torch.zeros(3, 8); h[:, 0] = torch.tensor([1.0, 3.0, 5.0])
    hc = conds["clamp_q50"](v)(h)
    assert torch.allclose(hc[:, 0], torch.tensor([1.0, 2.0, 2.0]), atol=1e-5)
    hq = conds["otq_cal"](v)(h)
    assert torch.allclose(hq[:, 0], torch.tensor([-0.5, 0.5, 1.5]), atol=1e-4)
    assert abs(q_at({"q": torch.linspace(0, 1, 41).tolist()}, 0.50) - 0.5) < 1e-6
    print("steer_v2 self-test passed")
