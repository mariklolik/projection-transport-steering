# Theorem 1 holds for any orthonormal V, so the behavioral subspace need not be
# the two interpretable axes. Given the full-space source and target moments a
# distribution-matching baseline would use, the directions that matter are the
# ones its map actually moves: the mean shift, and the eigenvectors of A - I
# where A is the Bures-Wasserstein matrix. Keeping the leading k of them gives
# the same kind of operator at O(kd) per token instead of O(d^2).
# Self-test: python -m general.behavioral_subspace

from __future__ import annotations

import torch

from general.steering import bw_map, orthonormalize


def displacement_subspace(m_s: torch.Tensor, S_s: torch.Tensor, m_t: torch.Tensor,
                          S_t: torch.Tensor, k: int, reg: float = 1e-2,
                          seed: torch.Tensor | None = None) -> torch.Tensor:
    """[k, d] orthonormal rows spanning where the full-space BW map does its work.

    `seed` optionally prepends rows that must be kept (the interpretable axes).
    """
    d = m_s.shape[0]
    eye = torch.eye(d, dtype=torch.float64)
    A = bw_map(S_s.double() + reg * eye, S_t.double() + reg * eye)
    vals, vecs = torch.linalg.eigh(A - eye)
    order = vals.abs().argsort(descending=True)
    rows = [] if seed is None else list(seed.double())
    shift = (m_t - m_s).double()
    if shift.norm() > 1e-8:
        rows.append(shift)
    for i in order:
        if len(rows) >= k:
            break
        rows.append(vecs[:, i])
    return orthonormalize(torch.stack(rows[:k]).float())


def projected_moments(V: torch.Tensor, m_s: torch.Tensor, S_s: torch.Tensor,
                      m_t: torch.Tensor, S_t: torch.Tensor):
    """Source and target moments of q = Vh, the only statistics the map needs."""
    return V @ m_s, V @ S_s @ V.T, V @ m_t, V @ S_t @ V.T


def _selftest():
    torch.manual_seed(0)
    d = 12
    m_s, m_t = torch.zeros(d), torch.zeros(d)
    m_t[0] = 3.0
    S_s = torch.eye(d)
    S_t = torch.eye(d)
    S_t[1, 1] = 9.0                                  # the map stretches exactly one axis
    V = displacement_subspace(m_s, S_s, m_t, S_t, k=2)
    assert V.shape == (2, d)
    assert torch.allclose(V @ V.T, torch.eye(2), atol=1e-4)
    # it must find the shift axis and the stretched axis, in that order
    assert V[0].abs().argmax().item() == 0 and V[1].abs().argmax().item() == 1
    ms, Ss, mt, St = projected_moments(V, m_s, S_s, m_t, S_t)
    assert abs(float(mt[0]) - 3.0) < 1e-3 and abs(float(St[1, 1]) - 9.0) < 1e-2
    assert displacement_subspace(m_s, S_s, m_t, S_t, k=5).shape == (5, d)
    print("general.behavioral_subspace self-tests passed")


if __name__ == "__main__":
    _selftest()
