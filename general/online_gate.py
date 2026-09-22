# Single-pass (streaming) version of the trace gate. The post-hoc gate scores a
# COMPLETED trace and therefore needs a second generation for flagged traces;
# this one keeps a running mean of the same LDA statistic over every position
# already seen and switches the action on the moment an anytime-valid boundary
# is crossed. No regeneration, no KV-cache invalidation: positions before the
# switch keep their unsteered keys.
#
# Detection and action are separate layers. The reader sits at the depth where
# the detection contrast is most linearly readable; the action stays at the
# steering layer below it, so in one forward pass the action for token t sees
# the statistic accumulated over tokens < t (a one-position lag). Prompt
# positions are therefore observed for free during prefill and the action can
# start at the first generated token.
# Self-test: python -m general.online_gate

from __future__ import annotations

import torch


def boundary(n: torch.Tensor, sigma: float, delta: float) -> torch.Tensor:
    """Time-uniform deviation bound for a running mean of n sub-Gaussian terms.

    The stitched finite-LIL boundary of Howard et al. (Ann. Statist. 2021,
    Eq. 11) at eta=2, s=1.4: 1.7 sigma sqrt((loglog 2n + 0.72 log(5.2/delta))/n).
    It holds simultaneously at every n, so no alpha is spent on when we look.
    """
    nn = n.clamp(min=2.0)
    import math
    inner = torch.log(torch.log(2.0 * nn).clamp(min=1e-6)).clamp(min=0.0) + 0.72 * math.log(5.2 / delta)
    return 1.7 * sigma * (inner / nn).sqrt()


def platt_fit(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Calibrate an LDA score into a posterior: P(target | s) = sigmoid(a s + c)."""
    a = torch.ones(1, requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, c], max_iter=200)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(a * scores + c, labels.float())
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(c.detach())


def posterior_dose(scores: torch.Tensor, a: float, c: float, tau: float, rho: float) -> torch.Tensor:
    """Prop. 5 dose profile: clip(rho (P(target | s) - tau), 0, 1).

    rho -> infinity recovers the hard gate 1[P > tau]; rho = 0 leaves the model
    untouched. tau is the ratio of per-unit collateral cost to per-unit gain.
    """
    return ((torch.sigmoid(a * scores + c) - tau) * rho).clamp(0.0, 1.0)


class SequentialGate:
    """Anytime-valid running-mean gate; `reader` accumulates, `actor` intervenes.

    V: [k, d] orthonormal rows; (w, b): the trace-level LDA head on V-projections;
    tau: the threshold the post-hoc gate uses; sigma: per-token std of the gate
    contribution on the extraction split. Once a sequence fires it stays on (the
    decision is a commitment, not a per-token flicker).
    """

    def __init__(self, V: torch.Tensor, w: torch.Tensor, b: float, tau: float,
                 sigma: float, delta: float = 0.05, warmup: int = 8, kappa: float = 0.0,
                 read_prompt: bool = False, decide_at: int | None = None):
        self.V, self.w, self.b, self.tau = V, w, b, float(tau)
        self.sigma, self.delta, self.warmup = float(sigma), float(delta), int(warmup)
        self.read_prompt = read_prompt   # off: the statistic is the post-hoc one, trace tokens only
        # decide_at: commit once at a fixed budget, with a head fitted on exactly
        # that prefix; the anytime boundary is then unnecessary and is not used.
        self.decide_at = decide_at
        self.kappa = float(kappa)   # >0: dose ramps with the evidence instead of switching
        self.state = None
        self.fire_step: list[int | None] = []
        self.history: list[int | None] = []

    def _reset(self, batch: int, device, dtype) -> None:
        z = torch.zeros(batch, device=device, dtype=dtype)
        self.history.extend(self.fire_step)
        self.state = {"sum": z, "n": z.clone(), "on": z.bool(), "lam": z.clone()}
        self.fire_step = [None] * batch

    def fire_steps(self) -> list[int | None]:
        return self.history + self.fire_step

    def reader(self, h: torch.Tensor) -> torch.Tensor:
        """Hook at the detection layer: updates the running statistic, edits nothing."""
        if self.state is None:
            self._reset(h.shape[0], h.device, h.dtype)
        if h.shape[1] > 1 and not self.read_prompt:
            return h            # prefill is the prompt; the post-hoc score ignores it
        st = self.state
        s_tok = (h @ self.V.T) @ self.w + self.b
        t = h.shape[1]
        cnt = st["n"].unsqueeze(1) + torch.arange(1, t + 1, device=h.device, dtype=h.dtype)
        run = (st["sum"].unsqueeze(1) + s_tok.cumsum(-1)) / cnt
        if self.decide_at is None:
            fire = (run > self.tau + boundary(cnt, self.sigma, self.delta)) & (cnt >= self.warmup)
        else:
            fire = (run > self.tau) & (cnt >= self.decide_at) & (cnt < self.decide_at + 1)
        on = (st["on"].unsqueeze(1) | fire).cummax(-1).values
        if self.kappa > 0:
            ramp = ((run - self.tau - boundary(cnt, self.sigma, self.delta)) / self.kappa)
            lam = (ramp.clamp(0, 1) * on).cummax(-1).values.amax(-1)
            st["lam"] = torch.maximum(st["lam"], lam)
        for i in range(h.shape[0]):
            if self.fire_step[i] is None and bool(on[i, -1]):
                self.fire_step[i] = int(st["n"][i].item() + int(on[i].float().argmax().item()) + 1)
        st["sum"], st["n"], st["on"] = st["sum"] + s_tok.sum(-1), cnt[:, -1], on[:, -1]
        return h

    def actor(self, action):
        """Hook at the steering layer: applies `action` on sequences already fired."""
        def fn(h: torch.Tensor) -> torch.Tensor:
            if h.shape[1] > 1 or self.state is None or self.state["sum"].shape[0] != h.shape[0]:
                self._reset(h.shape[0], h.device, h.dtype)   # prefill starts a sequence
            on = self.state["on"]
            if self.kappa > 0:
                lam = (self.state["lam"] * on).to(h.dtype)[:, None, None]
                return h + lam * (action(h) - h)
            return torch.where(on[:, None, None], action(h), h)
        return fn


if __name__ == "__main__":
    torch.manual_seed(0)
    d, k = 16, 2
    V = torch.eye(d)[:k]
    w, b, tau = torch.tensor([1.0, 0.0]), 0.0, 1.0
    act = lambda h: h * 0.0

    g = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2, read_prompt=True)
    a = g.actor(act)
    hot = torch.zeros(1, 32, d)
    hot[..., 0] = 5.0
    assert torch.allclose(a(hot), hot)              # prefill is observed, not edited
    g.reader(hot)
    assert g.fire_step[0] is not None and g.fire_step[0] <= 4
    assert a(torch.ones(1, 1, d)).abs().max() == 0.0   # first decode step is steered

    g2 = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2, read_prompt=True)
    a2 = g2.actor(act)
    cold = torch.zeros(1, 32, d)
    cold[..., 0] = -5.0
    a2(cold)
    g2.reader(cold)
    assert g2.fire_step[0] is None and torch.allclose(a2(torch.ones(1, 1, d)), torch.ones(1, 1, d))

    # streaming equals one-shot: the accumulated state after 20 tokens is identical
    seq = torch.randn(2, 20, d)
    g3 = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2, read_prompt=True)
    g3.actor(act)(seq)
    g3.reader(seq)
    g4 = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2, read_prompt=True)
    a4 = g4.actor(act)
    for i in range(20):
        a4(seq[:, i:i + 1]) if i else a4(seq[:, :1])
        g4.reader(seq[:, i:i + 1])
    assert g3.fire_step == g4.fire_step
    assert torch.allclose(g3.state["sum"], g4.state["sum"], atol=1e-4)

    # soft dose: a barely-crossing trace gets a partial edit, a strong one the full edit
    weak = torch.zeros(1, 40, d); weak[..., 0] = 1.2
    strong = torch.zeros(1, 40, d); strong[..., 0] = 9.0
    outs = []
    for x in (weak, strong):
        gk = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2, kappa=4.0,
                            read_prompt=True)
        ak = gk.actor(act)
        ak(x)
        gk.reader(x)
        probe = torch.ones(1, 1, d)
        outs.append(float((probe - ak(probe)).abs().mean()))
    assert 0 < outs[0] < outs[1], outs

    # fixed-budget mode: the decision is taken once, at the budget it was fitted for
    gd = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, decide_at=10)
    ad = gd.actor(act)
    x = torch.zeros(1, 20, d)
    x[..., 0] = 2.0
    ad(x)
    for i in range(20):
        gd.reader(x[:, i:i + 1])
    assert gd.fire_step[0] == 10, gd.fire_step

    gd2 = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, decide_at=10)
    gd2.actor(act)(x)
    y = torch.zeros(1, 20, d)
    y[..., 0] = -2.0
    for i in range(20):
        gd2.reader(y[:, i:i + 1])
    assert gd2.fire_step[0] is None

    # default: the prompt is skipped, so the statistic is exactly the post-hoc one
    gp = SequentialGate(V, w, b, tau, sigma=0.1, delta=0.05, warmup=2)
    gp.actor(act)(hot)
    gp.reader(hot)
    assert gp.state["n"].max() == 0 and gp.fire_step[0] is None

    s_hi = torch.linspace(-4, 4, 200)
    a_, c_ = platt_fit(s_hi, (s_hi > 0).int())
    assert a_ > 0 and abs(c_) < 1.0, (a_, c_)
    lam = posterior_dose(torch.tensor([-8.0, 0.0, 8.0]), a_, c_, tau=0.35, rho=4.0)
    assert lam[0] == 0.0 and lam[2] == 1.0 and 0.0 <= lam[1] <= 1.0, lam
    assert posterior_dose(s_hi, a_, c_, 0.35, 1e6).unique().tolist() == [0.0, 1.0]

    n = torch.tensor([10.0, 1000.0])
    bd = boundary(n, 1.0, 0.05)
    assert bd[0] > bd[1] > 0
    print("general.online_gate self-tests passed")
