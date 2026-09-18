# Steering operators on the residual stream. A direction v is a unit vector;
# every op rewrites only the projection p = <h, v> (PTS view: h' = h + (g(p)-p)v).
# Ops: add, ablate, conditional, clamp, 1-D OT (gauss/quantile), k-dim affine
# (Bures-Wasserstein), per-neuron affine (Linear-AcT), full-space affine (MiMiC).
# Self-tests: python -m general.steering

from __future__ import annotations

import torch


def project(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Signed length of `h` along direction `v` (dot with the normalized v)."""
    vhat = v / v.norm()
    return h @ vhat


def steer_add(h: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """Additive steering: push activations by alpha along v. alpha<0 suppresses."""
    return h + alpha * v


def steer_ablate(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Directional ablation: remove the component of h along v entirely."""
    vhat = v / v.norm()
    return h - (h @ vhat).unsqueeze(-1) * vhat


def steer_conditional(h: torch.Tensor, v: torch.Tensor, alpha: float, tau: float) -> torch.Tensor:
    """Gated steering: add alpha*v only at positions whose projection exceeds tau."""
    vhat = v / v.norm()
    mask = (h @ vhat > tau).unsqueeze(-1).to(h.dtype)
    return h + mask * alpha * v


def steer_clamp(h: torch.Tensor, v: torch.Tensor, t: float) -> torch.Tensor:
    """Projection clamp: p <- min(p, t). Compresses ONLY the tail above t.

    Input-dependent dose (t - p for p > t, zero otherwise): tokens already below
    the ceiling are untouched, far-out tokens are pulled in exactly to t.
    """
    vhat = v / v.norm()
    p = h @ vhat
    return h + (p.clamp(max=t) - p).unsqueeze(-1) * vhat


def steer_ot_gauss(h: torch.Tensor, v: torch.Tensor,
                   mu_s: float, sig_s: float, mu_t: float, sig_t: float) -> torch.Tensor:
    """1-D Gaussian optimal-transport map along v: p <- μ_t + (σ_t/σ_s)(p − μ_s).

    The unique monotone (= W2-optimal) map sending N(μ_s, σ_s²) to N(μ_t, σ_t²).
    Additive steering is the special case σ_t = σ_s (pure mean shift).
    """
    vhat = v / v.norm()
    p = h @ vhat
    g = mu_t + (sig_t / sig_s) * (p - mu_s)
    return h + (g - p).unsqueeze(-1) * vhat


def interp1d(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of x on knots (xs, ys); xs strictly increasing.

    Out-of-range x extrapolates linearly along the FIRST/LAST segment (the tails
    are exactly where overconfident activations live — flat clipping there would
    hide the signal the map needs to move).
    """
    idx = torch.searchsorted(xs, x.contiguous()).clamp(1, len(xs) - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def steer_ot_quantile(h: torch.Tensor, v: torch.Tensor,
                      src_q: torch.Tensor, tgt_q: torch.Tensor) -> torch.Tensor:
    """Empirical 1-D OT along v: p <- F_target⁻¹(F_source(p)) via matched quantile grids.

    `src_q`/`tgt_q` are same-length monotone quantile grids (e.g. q=1..99%) of the
    source and target projection distributions. Monotone rearrangement is the
    unique OT map for any convex cost in 1-D — no Gaussian assumption.
    """
    vhat = v / v.norm()
    p = h @ vhat
    g = interp1d(p, src_q.to(p), tgt_q.to(p))
    return h + (g - p).unsqueeze(-1) * vhat


def bw_map(sigma_s: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    """The Bures–Wasserstein matrix A = Σs^{-1/2}(Σs^{1/2} Σt Σs^{1/2})^{1/2} Σs^{-1/2}.

    T(x) = m_t + A(x − m_s) is the unique W2-optimal (Monge) map between
    N(m_s, Σs) and N(m_t, Σt) (Knott–Smith / Olkin–Pukelsheim). A is symmetric
    PSD; for k=1 it reduces to σt/σs; A=I is the additive special case.
    """
    def sqrtm(m):
        vals, vecs = torch.linalg.eigh(m)
        return vecs @ torch.diag(vals.clamp(min=1e-12).sqrt()) @ vecs.T

    s_half = sqrtm(sigma_s)
    s_ihalf = torch.linalg.inv(s_half)
    return s_ihalf @ sqrtm(s_half @ sigma_t @ s_half) @ s_ihalf


def steer_subspace_affine(h: torch.Tensor, V: torch.Tensor, m_s: torch.Tensor,
                          A: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
    """k-dim PTS: affine transport of the projection onto span(V), identity on V⊥.

    V: [k, d] with ORTHONORMAL rows. q = h Vᵀ ∈ ℝᵏ; q ← m_t + A(q − m_s);
    h' = h + (q' − q) V. With A = bw_map(Σs, Σt) this is the W2-optimal
    intervention among all maps supported on span(V) that produce the target
    Gaussian projection law (THEORY_V2 Thm 1+2).
    """
    q = h @ V.T                                        # [..., k]
    q_new = m_t + (q - m_s) @ A.T
    return h + (q_new - q) @ V


def steer_perneuron_affine(h: torch.Tensor, mu_s: torch.Tensor, sig_s: torch.Tensor,
                           mu_t: torch.Tensor, sig_t: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Linear-AcT baseline: independent per-neuron 1-D Gaussian OT, λ-blended.

    h'_i = (1−λ)·h_i + λ·(μt_i + (σt_i/σs_i)(h_i − μs_i)) — the coordinate-wise
    special case of transport (assumes neuron independence; cf. AcT).
    """
    mapped = mu_t + (sig_t / sig_s) * (h - mu_s)
    return (1 - lam) * h + lam * mapped


def steer_fullspace_affine(h: torch.Tensor, m_s: torch.Tensor, A: torch.Tensor,
                           m_t: torch.Tensor) -> torch.Tensor:
    """MiMiC baseline: unconditional full-space affine h' = m_t + A(h − m_s).

    With A = bw_map(S_s, S_t) this is the Gaussian-W2 map in ALL of ℝ^d — no
    locality constraint, every direction of the residual stream is rewritten.
    """
    return m_t + (h - m_s) @ A.T


def orthonormalize(dirs: torch.Tensor) -> torch.Tensor:
    """Gram–Schmidt on rows: [k, d] -> [k, d] with orthonormal rows (order kept)."""
    out = []
    for v in dirs:
        w = v.clone()
        for u in out:
            w = w - (w @ u) * u
        out.append(w / w.norm())
    return torch.stack(out)


def steering_hook(layer_module, fn):
    """Register `fn(h) -> h'` on a decoder layer's output. Returns the handle; .remove() it."""
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = fn(h.float()).to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return layer_module.register_forward_hook(hook)


class steered:
    """`with steered(model, layer, fn): ...` — steering hook that auto-removes.

    Combine with generate(..., stop_strings=["</think>"]) to steer only ONE phase
    of the reasoning (steer the thought, then generate the answer unsteered, or
    the reverse).
    """

    def __init__(self, model, layer: int, fn):
        self.module = model.model.layers[layer]
        self.fn = fn

    def __enter__(self):
        self.handle = steering_hook(self.module, self.fn)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


if __name__ == "__main__":
    torch.manual_seed(0)
    h = torch.randn(5, 16)
    v = torch.randn(16)

    assert torch.allclose(steer_add(h, v, 0.0), h)
    ha = steer_ablate(h, v)
    assert project(ha, v).abs().max() < 1e-4
    assert torch.allclose(steer_ablate(ha, v), ha, atol=1e-4)  # idempotent

    tau = project(h, v).median().item()
    hc = steer_conditional(h, v, alpha=1.0, tau=tau)
    changed = (hc - h).norm(dim=-1) > 1e-6
    assert changed.tolist() == (project(h, v) > tau).tolist()

    vhat = v / v.norm()
    assert project(vhat, v) > 0 and project(-vhat, v) < 0

    # clamp: only p > t moves, and lands exactly at t; orthogonal part untouched
    t = project(h, v).median().item()
    hcl = steer_clamp(h, v, t)
    p_before, p_after = project(h, v), project(hcl, v)
    assert torch.allclose(p_after, p_before.clamp(max=t), atol=1e-5)
    untouched = p_before <= t
    assert torch.allclose(hcl[untouched], h[untouched], atol=1e-6)
    ortho = h - p_before.unsqueeze(-1) * vhat
    ortho_after = hcl - p_after.unsqueeze(-1) * vhat
    assert torch.allclose(ortho, ortho_after, atol=1e-5)

    # gaussian OT: projections land on the target moments; σ_t=σ_s reduces to add
    big = torch.randn(4096, 16) * 3 + 5.0  # arbitrary loc/scale
    mu_s, sig_s = project(big, v).mean().item(), project(big, v).std().item()
    hg = steer_ot_gauss(big, v, mu_s, sig_s, mu_t=-1.0, sig_t=0.5)
    pg = project(hg, v)
    assert abs(pg.mean().item() + 1.0) < 0.05 and abs(pg.std().item() - 0.5) < 0.05
    ha_ot = steer_ot_gauss(h, v, 0.0, 1.0, 2.0, 1.0)
    assert torch.allclose(ha_ot, steer_add(h, vhat, 2.0), atol=1e-5)

    # interp1d: exact at knots, linear between, linear extrapolation at tails
    xs, ys = torch.tensor([0.0, 1.0, 3.0]), torch.tensor([0.0, 2.0, 4.0])
    got = interp1d(torch.tensor([0.0, 0.5, 2.0, 3.0, 4.0, -1.0]), xs, ys)
    assert torch.allclose(got, torch.tensor([0.0, 1.0, 3.0, 4.0, 5.0, -2.0]), atol=1e-6)

    # quantile OT: samples from source dist land on the target quantile grid
    qs = torch.linspace(0.01, 0.99, 41)
    src = torch.randn(8192) * 2 + 3
    tgt = torch.randn(8192) * 0.5 - 1
    src_q, tgt_q = src.quantile(qs), tgt.quantile(qs)
    hq = steer_ot_quantile(src.unsqueeze(-1) * vhat + ortho.mean(0), v, src_q, tgt_q)
    pq = project(hq, v)
    assert abs(pq.mean().item() - tgt.mean().item()) < 0.1
    assert abs(pq.std().item() - tgt.std().item()) < 0.1
    # monotone: order of projections preserved
    perm = torch.argsort(project(h, v))
    assert torch.all(torch.diff(project(steer_ot_quantile(h, v, src_q, tgt_q), v)[perm]) > -1e-5)

    # bw_map: 1-D reduces to sig_t/sig_s; identity when Σs=Σt; symmetric PSD
    A1 = bw_map(torch.tensor([[4.0]]), torch.tensor([[1.0]]))
    assert abs(A1.item() - 0.5) < 1e-5
    S = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    assert torch.allclose(bw_map(S, S), torch.eye(2), atol=1e-4)
    T2 = torch.tensor([[1.0, -0.2], [-0.2, 0.5]])
    A2 = bw_map(S, T2)
    assert torch.allclose(A2, A2.T, atol=1e-5)
    assert torch.linalg.eigvalsh(A2).min() > 0

    # subspace affine transport: moments of the transported projections match target
    Vd = orthonormalize(torch.randn(2, 16))
    assert torch.allclose(Vd @ Vd.T, torch.eye(2), atol=1e-5)
    n = 8192
    L = torch.linalg.cholesky(S)
    q_src = torch.randn(n, 2) @ L.T + torch.tensor([3.0, -1.0])
    hb = q_src @ Vd + torch.randn(n, 16) * 0.01  # mostly in-subspace + noise
    m_s = q_src.mean(0)
    Sig_s = torch.cov(q_src.T)
    m_t = torch.tensor([0.0, 1.0])
    hbt = steer_subspace_affine(hb, Vd, m_s, bw_map(Sig_s, T2), m_t)
    q_new = hbt @ Vd.T
    assert (q_new.mean(0) - m_t).abs().max() < 0.05
    assert (torch.cov(q_new.T) - T2).abs().max() < 0.1
    # orthogonal complement untouched
    P_perp = torch.eye(16) - Vd.T @ Vd
    assert torch.allclose(hb @ P_perp, hbt @ P_perp, atol=1e-4)

    print("general.steering self-tests passed")
