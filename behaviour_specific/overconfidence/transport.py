from __future__ import annotations


import torch

from general.steering import bw_map, steer_subspace_affine

OCW = "overconfident_wrong"


def make_subspace_fn(V: torch.Tensor, m_s: torch.Tensor, S_s: torch.Tensor,
                     m_t: torch.Tensor, S_t: torch.Tensor):

    A = bw_map(S_s.double(), S_t.double()).float()
    def fn(h):
        return steer_subspace_affine(h, V.to(h.device, h.dtype), m_s.to(h.device, h.dtype),
                                     A.to(h.device, h.dtype), m_t.to(h.device, h.dtype))
    return fn


def _selftest():
    torch.manual_seed(0)
    V = torch.eye(2, 8)
    m_s, S_s = torch.tensor([2.0, -1.0]), torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    m_t, S_t = torch.zeros(2), torch.eye(2) * 0.25
    fn = make_subspace_fn(V, m_s, S_s, m_t, S_t)
    L = torch.linalg.cholesky(S_s)
    h = torch.zeros(4096, 8)
    h[:, :2] = torch.randn(4096, 2) @ L.T + m_s
    q = fn(h)[:, :2]
    assert q.mean(0).abs().max() < 0.05 and (torch.cov(q.T) - S_t).abs().max() < 0.05
    print("steer_v3_optimal self-test passed")
