# Flow-step support hypotheses, version 1

Status: frozen before any flow-step-support output. The internal label is not authorized for manuscript use.

## Intervention

For the released FLAS checkpoint with `T=2` and `N=3`, let `v_k` denote the velocity evaluation at released time index `k` and let `Phi_S(h)` be the state obtained by executing only indices in ordered subset `S`, leaving the state unchanged at skipped indices. Define the equal-time support transform

`Psi_S(h) = h + (3 / |S|) (Phi_S(h) - h)`.

The fixed candidate is `S={1,2}`. It removes the empirically shared early step and retains the later concept-specific time indices at equal nominal update mass. Because skipped steps change later states, `Psi_S` is neither the original flow nor an accurate numerical integrator in general. No trajectory, manifold, or causal-behavior guarantee follows.

## Conditions

Evaluate the seven nonempty raw subsets `{0}`, `{1}`, `{2}`, `{0,1}`, `{0,2}`, `{1,2}`, and `{0,1,2}` for exact cooperative-game decomposition. Evaluate equal-time controls for all one- and two-step subsets, plus the released full path and the released checkpoint run with `N=2`. The primary candidate is fixed to equal-time `{1,2}`; equal-time `{0,1}` is the matched early-support mechanism control. No per-concept subset, factor, flow time, or endpoint selection is allowed.

For any scalar endpoint `f(S)` with the base model as `f(empty)`, report the exact three-player Shapley allocation over raw subsets. This is an algebraic attribution of the measured set function. It is not a causal identification result beyond the implemented subset intervention and does not make the Euler increments independent.

## Hypotheses

- FS.1: base replay, released full-path replay, subset ordering, state cleanup, finite outputs, source hashes, and serialization pass exactly.
- FS.2: on all 12 post-hoc design concepts, equal-time late support `{1,2}` has larger held-out worst-view contrast than released full FLAS in at least 9/12 concepts with positive mean paired difference.
- FS.3: the candidate lowers per-concept median neutral forward KL from base in at least 9/12 concepts, has mean fractional KL reduction at least 25%, improves neutral likelihood against full FLAS in at least 9/12 concepts, and has positive mean neutral-likelihood difference.
- FS.4: the candidate beats equal-time early support `{0,1}` on held-out worst-view contrast in at least 9/12 concepts with positive mean paired difference.
- FS.5: all raw subset effects and Shapley allocations are reported even if the candidate fails.
- FS.6: a pass authorizes only one frozen test on the still-unmaterialized 24-concept successor allocation. Generation, judging, held-out, 9B, cross-family, novelty, and SOTA remain closed.
