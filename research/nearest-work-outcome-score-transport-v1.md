# Nearest-work boundary for outcome-score transport, version 1

Coverage date: 2026-09-07. Status: prospective novelty boundary, not a contribution claim.

## Occupied ideas

| Neighbor | Occupied scientific object | Consequence |
|---|---|---|
| Projection-Transport Steering | One-dimensional monotone transport followed by a minimum-metric lift along a declared linear projection | Scalar OT plus a minimum-cost lift is not new by itself. |
| FishBack | Output-functional minimum pullback-Fisher action for next-token morphology | A Fisher minimum-distortion activation update is not new by itself. |
| A-LQR | Local transformer dynamics, Jacobians, Riccati feedback, and adaptive semantic setpoints over depth | Feedback or local linear control is not new by itself. |
| Latent Reward Steering | Terminal-outcome supervision, a learned latent reward, token-local normalized reward gradients, and reward-confidence gating | Reward-gradient steering and selective intervention are not new by themselves. |
| Policy Gradient Steering | Rollout-return-weighted activation policy gradients averaged into a fixed vector and scalar KL/Fisher calibration | Natural-gradient language or rollout-weighted gradients are not new by themselves. |

The new route cannot be framed as merely reward steering, optimal transport, Fisher geometry, gating, or feedback. Its empirical comparison must isolate each of those components.

## Unoccupied conjunction to test

The working object is a prefix-consistent outcome value, trained and queried on the same information set, whose score distribution is transported toward the successful-prefix distribution. The transported score defines a per-state desired increment. A separate local constrained solve realizes only that scalar increment at minimum cost under one fixed pooled metric.

The candidate differs from its closest neighbors only if evidence supports all of the following:

1. prefix-consistent value estimation is calibrated on held-out questions and predicts paired action value rather than merely terminal correctness;
2. score transport supplies useful state-dependent dose beyond fixed-step reward ascent;
3. the declared metric reduces output distortion at matched achieved score increment;
4. the coupled observer and action outperform their observer-only, action-only, Euclidean, fixed-dose, and normalized-gradient ablations;
5. the effect survives a temporal split and at least one broader benchmark without evaluation-cell tuning.

Until those gates pass, `Outcome Score Transport` and `ValueTransport` are working labels only. There is no novelty, behavior, architecture, or SOTA claim.

## Honest theorem boundary

The intended proof composes two conditional statements:

- under finite second moments and an atomless scalar source law, monotone quantile transport minimizes one-dimensional quadratic transport cost;
- for a differentiable score, a positive-definite local metric, a nonzero score gradient, and a first-order target increment, the metric-projected update is the unique minimum-cost update satisfying that local scalar constraint.

Their composition does not establish globally optimal hidden-state transport, finite-rollout correctness improvement, causal abstraction, or safety. Approximation error, atoms, clipping, value-model misspecification, changing future dynamics, and repeated interventions remain empirical failure modes.
