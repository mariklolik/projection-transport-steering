# CacheBack method hypotheses, version 1

Status: independent Pilot B candidate, frozen before implementation or model output.

## Local future-distribution metric

Fix a base state at generation time `t`, a residual intervention `u`, a base rollout distribution `P0`, and teacher-forced continuation `tau`. Let `z_{t+k,tau}(u)` be the logits produced under the same base-sampled tokens and let `J_{t->t+k,tau}` be the Jacobian of those logits with respect to `u` at zero. For `p_{t+k,tau}` under the unsteered model, let `F = diag(p) - p p^T`. Define

`G_H = E_{tau ~ P0} sum_{k=0}^H J_{t->t+k,tau}^T F_{t+k,tau} J_{t->t+k,tau}`.

The expectation is approximated by a frozen common-random set of base rollouts. The primary metric uses the full vocabulary and float64 accumulation. Top-k Fisher is a named approximation diagnostic only.

For teacher-forced forward KL

`D_H(u) = E_tau sum_{k=0}^H KL(P0(.|tau_<k) || P_u(.|tau_<k))`,

differentiability gives `D_H(0)=0`, zero first derivative, and

`D_H(u) = 0.5 u^T G_H u + o(||u||^2)`.

Thus `G_H` is a local second-order metric for expected truncated sequence KL under the fixed base-trajectory distribution. It is not exact finite-action sequence optimality. With the same trajectories nested across horizons, `G_{H+1}-G_H` is positive semidefinite up to numerical tolerance.

## Action

Let `q` be the immediate semantic log-odds covector at `t`, `M_H = G_H + lambda I`, and `rho` the common local target. The exact action is

`u_H = rho M_H^{-1} q / (q^T M_H^{-1} q)`.

All horizons share the same state, covector, regularization rule, rollout samples, and realized immediate-effect calibration. `H=0` is exact FishBack under the new full-vocabulary implementation.

## Cache-path diagnostic

The standard condition injects at one residual state and allows downstream layers to write the altered token into their key/value cache. The cache-reset condition computes the immediate logits under intervention but replaces every affected current-token cache entry with its unsteered value before `t+1`. Under teacher forcing, a one-shot decoder intervention has no recurrent path to later tokens except cached states; future Jacobians should therefore collapse in this condition. Collapse alone is only a wiring sanity check.

The cache-only condition holds immediate logits at their base value while applying the first-order affected-cache displacement to future decoding. A cache-mediated mechanism requires both substantial collapse under reset and recovery under cache-only. This paired criterion is more discriminating than requiring a one-shot future effect to survive cache removal, which would contradict the decoder causal graph.

For `H>0`, define the temporal-externality ratio

`TER_H(u) = u^T (G_H_full - G_H_reset) u / u^T (G_H_full - G_0) u`

when the denominator exceeds the frozen numerical floor. Values, numerator, denominator, cache-only recovery, and bootstrap uncertainty are reported without clipping.

## Hypotheses

- CB.1: exact synthetic and GPT-2 replay tests recover the KL Hessian, PSD horizon nesting, the closed-form matched-target action, and cache reset/cache-only causal semantics within frozen tolerances.
- CB.2: `H in {2,4,8}` changes the action relative to `H=0` by more than numerical noise on at least two horizons, with finite condition numbers and no PSD-nesting violation beyond tolerance.
- CB.3: at the same realized immediate semantic target, CacheBack reduces cumulative teacher-forced forward KL by at least 15% versus exact `H=0` FishBack, with a one-sided 95% lower interval above 15%, on at least two horizons.
- CB.4: immediate target-effect retention is at least 98%, matched reachability is at least 80%, and no single future offset contributes 80% or more of the total paired KL reduction.
- CB.5: predicted quadratic cost ranks realized small-dose KL with a cluster-bootstrap Spearman lower interval above `0.50`, and the median relative Taylor error at the smallest dose is below 20%.
- CB.6: cache reset removes at least 80% of future quadratic cost and cache-only recovers at least 50%; `TER_H` is positive on at least two horizons. Failure narrows the result to generic autoregressive propagation or closes it.
- CB.7: exact `H=8` construction uses at most 12 times `H=0` wall time, at most 40 GB peak allocated memory, and complete exact Jacobian/JVP/VJP counts. A slower result is mechanism-only and cannot be promoted as a practical controller.

## Claim boundary

Passing the pilot supports local truncated-sequence geometry only for GPT-2 Small, the frozen morphology targets, horizons, layer, and rollout distribution. It does not establish free-running generation optimality, instruction-tuned-model transfer, long-context control, safety, or SOTA. PathwayBack is not implemented unless CB.1-CB.7 all pass.
