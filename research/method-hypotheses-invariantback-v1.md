# InvariantBack method hypotheses, version 1

Status: independent Pilot A candidate, frozen before implementation or model output.

## Robust semantic action

At one fixed residual layer, let `q_g in R^d` be the semantic target covector for construction encoding view `g`. Let `G_bar` be the equal-view pooled off-target Fisher metric and let `M = G_bar + lambda I`, with `lambda > 0`. Each covector is normalized as

`q_tilde_g = q_g / sqrt(q_g^T M^{-1} q_g)`

so every isolated view has the same local minimum cost at a common target `rho`. With `Q = [q_tilde_1,...,q_tilde_m]`, the hard candidate is

`min_u 0.5 u^T M u  subject to  Q^T u >= rho 1`.

This objective protects the pooled off-target output functional while requiring local semantic progress in every construction view. It does not guarantee finite-action behavior or unseen-view invariance.

## Dual and KKT system

Writing constraints as `rho 1 - Q^T u <= 0` with multipliers `alpha >= 0`, the Lagrangian is

`L(u, alpha) = 0.5 u^T M u + alpha^T (rho 1 - Q^T u)`.

Stationarity gives `M u - Q alpha = 0`, hence `u = M^{-1} Q alpha`. Substitution gives the concave dual

`max_{alpha >= 0} rho 1^T alpha - 0.5 alpha^T Q^T M^{-1} Q alpha`.

At an optimum, primal feasibility, dual feasibility, stationarity, and complementarity all hold:

- `Q^T u >= rho 1`;
- `alpha >= 0`;
- `M u = Q alpha`;
- `alpha_g (q_tilde_g^T u - rho) = 0` for every view.

The primal action is unique because `M` is positive definite. Dual coefficients need not be unique when view covectors are redundant. A positive coefficient identifies an active KKT multiplier under the solved program; it is not called a hardest encoding.

## Infeasibility and contingent method

For `rho > 0`, the hard constraints are infeasible exactly when there exists a nonzero `alpha >= 0` with `Q alpha = 0`. This is checked numerically with a scale-aware feasibility tolerance and recorded per group.

The only contingent method is fixed before data inspection:

`min_{u, xi >= 0} 0.5 u^T M u + C 1^T xi  subject to  Q^T u + xi >= rho 1`.

Its dual is the same quadratic objective under `0 <= alpha <= C`; it is always feasible. The candidate uses the hard action when all development construction groups are feasible. If hard feasibility is at least 90% but below 100%, it uses the slack action with `C` selected from `{1, 10, 100}` on development only. If feasibility is below 90%, the route closes. Pilot-test slack, margins, and active constraints are always reported. No CVaR, per-view weights, additional slack family, or post-result constraint deletion is allowed.

## Competing explanations

The pilot distinguishes five explanations:

1. averaging is sufficient, so mean-`q` FishBack equals the robust QP;
2. identifier nuisance can be removed linearly, so QuotientBack or SteerFair-style removal equals the robust QP;
3. Fisher geometry matters but worst-view constraints do not, so mean-`q` Fisher wins;
4. worst-view constraints matter but Fisher geometry does not, so robust Euclidean control wins;
5. both are required for held-out semantic control.

QuotientBack estimates a nuisance subspace only from construction identifier and mapping contrasts and solves `min 0.5 u^T G_bar u` subject to a semantic target and `N^T u = 0`. It is an ablation, not part of the primary method.

## Hypotheses

- IA.1: every returned hard or slack solution satisfies its KKT system within the frozen tolerance, and synthetic feasible, redundant, contradictory, and ill-conditioned systems behave as predicted.
- IA.2: at least 90% of development construction groups admit the hard action; otherwise the single contingent rule applies or the route closes.
- IA.3: at the development-selected off-target JS budget, InvariantBack improves the worst held-out semantic margin over the development-selected strongest feasible baseline by more than `0.10` pooled unsteered standard deviations, with a one-sided 95% lower interval above `0.10`.
- IA.4: canonical semantic performance is non-inferior within a 5% relative margin, mapping comprehension has a one-sided 95% lower interval at least `0.80`, and matched operating-point coverage is at least 80% overall and within every dataset and view family.
- IA.5: relative to CAA and FishBack, extraction-index advantage decreases with a one-sided 95% upper interval below zero while semantic-label following does not decrease. A uniform degradation cannot pass this gate.
- IA.6: the primary direction holds separately on NormBank, MNLI, and SC101, on new verbalizers, unseen row permutations, a new completion format, paraphrases, and open-ended semantic scoring without an unexplained sign reversal.
- IA.7: InvariantBack's construction, calibration, and deployment costs are fully measured and it remains on the target-effect versus JS versus compute Pareto frontier among feasible methods.

## Claim boundary

Passing this single-model pilot would support a Gemma-2-9B result under the frozen controlled datasets and views. It would not establish architecture breadth, broad behavioral safety, global robustness, causal abstraction, or SOTA. Failing IA.2-IA.6 closes the route without changing view definitions, slack family, Fisher rank, layer set, or threshold.
