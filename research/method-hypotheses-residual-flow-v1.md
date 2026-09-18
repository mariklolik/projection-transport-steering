# ResidualFlowBack method hypotheses, version 1

Status: post-sentinel candidate frozen before ResidualFlowBack model output. The name is internal and not authorized for manuscript branding.

## Residual correction around a released flow

Let `F_c` be the released FLAS transformation at layer 20 with flow time 2 and three Euler steps. For frozen matched witness `v`, let

`d_v = s_v(F_c) - s_v(0)`

be its finite length-normalized positive-minus-negative continuation-score change. Let

`q_v^F = grad_u s_v(F_c + u) at u=0`

be the covector of a constant additive correction installed after the FLAS hook at the same layer and applied at every causal position.

For target quantile `a` in `{0.50,0.75,1.00}`, define `rho_a = max(0, quantile_a({d_v}))` and residual deficits `b_v(a) = max(0,rho_a-d_v)`. The candidate correction solves

`min_u 0.5 u^T M_F u subject to (q_v^F)^T u >= b_v(a) for every v`,

where `M_F` is the same positive diagonal sequence-score metric as version 1, re-estimated from the frozen neutral pool with FLAS active for the current concept. Raw covectors are used because right-hand sides are in score units. No witness deletion, slack penalty, product cap, certificate relaxation, or post-output target is allowed.

The pooled residual control uses the deficit-weighted mean covector `sum_v b_v q_v^F / sum_v b_v` and its minimum-`M_F` direction. The Euclidean control solves the same vector-target inequalities under the identity metric. The shuffled-deficit control permutes `b_v` across fixed covectors with NumPy `SeedSequence([20260907, concept_id])`. Each control direction is rescaled to the candidate's exact `M_F` correction cost. The retained pooled sequence-metric direction is loaded from the immutable SemanticOutcomeBack packet and is rescaled to the same local correction cost. These rescalings are evaluation controls, not certificates that the rescaled control still realizes its construction target.

For positive-definite `M_F`, a feasible system has a unique primal correction. Its dual is the standard nonnegative quadratic program with a vector linear term `b(a)`. The result is only the minimum local quadratic correction for the declared linear inequalities. Finite FLAS score changes enter the right-hand side, but the correction remains first-order; no global score guarantee, causal abstraction, or exact population Fisher claim follows.

## Required controls

- `flas`: unchanged released flow;
- `flas_residual_metric`: the candidate;
- `flas_pooled_residual_metric`: one pooled trajectory-local covector, rescaled to the candidate correction metric cost;
- `flas_residual_euclidean`: identical nonuniform deficits under the identity metric, rescaled to the candidate metric cost;
- `flas_shuffled_deficits`: frozen permutation of deficits across witness covectors;
- `pooled_sequence_metric`: the strongest retained version-1 action without FLAS;
- released AxBench prompt, DiffMean, FLAS, CES, UniSteer where compatible, and strongest published frontier methods before any SOTA claim.

The candidate component requires improvement over unchanged FLAS and pooled residual correction. Beating only the Euclidean control supports the metric component already observed in version 1.

## Hypotheses

- RF.1: zero correction exactly reproduces released FLAS; all vector-target solves pass feasibility, KKT, factor reconstruction, finite output, hook ordering, and serialization checks.
- RF.2: all 12 post-hoc design concepts complete without witness removal, and at least one target quantile improves held-out worst-view score over both unchanged FLAS and pooled residual correction in at least 9/12 concepts with positive mean paired changes.
- RF.3: the same quantile has median neutral likelihood change no worse than `-0.02` nats per token relative to unchanged FLAS and does not increase median neutral forward KL by more than `0.002`.
- RF.4: one quantile is frozen from the design packet before any fresh 24-concept successor-development source is materialized.
- RF.5: on the fresh 24 concepts, the frozen candidate improves tuning-prompt AxBench HMean over unchanged FLAS and the strongest complete residual control with a Holm-adjusted one-sided 95% concept-clustered interval above zero while instruction, fluency, neutral, severe-collapse, failure, and compute gates pass.
- RF.6: held-out 2B, 9B, and cross-family studies remain closed until the fresh-development gate and exact judge receipt pass.

## Claim boundary

The 12 reused concepts are a post-hoc design set and cannot confirm the method. A design pass authorizes one frozen fresh-development test only. SOTA still requires common-protocol 2B and 9B results above the registered published markers and strongest direct comparators with complete uncertainty and failures.
