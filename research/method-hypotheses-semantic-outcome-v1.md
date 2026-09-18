# SemanticOutcomeBack method hypotheses, version 1

Status: working candidate frozen before implementation or model output. The name is internal and is not authorized for manuscript branding.

## Matched semantic witnesses

For concept `c`, construction view `v` is a frozen tuple `(x_v, y_v^+, y_v^-)` with one instruction, a concept-bearing continuation, and an unsteered continuation generated from the same instruction. Let `u in R^d` be one shared vector added after decoder block `l` at every causal token position. Define the length-normalized continuation score

`s_v(u) = mean_t log p_u(y^+_{v,t} | x_v, y^+_{v,<t}) - mean_t log p_u(y^-_{v,t} | x_v, y^-_{v,<t})`

and its zero-action covector `q_v = grad_u s_v(0)`. The positive and negative continuations use the same prompt but different teacher-forced causal histories. This is a local construction objective, not the behavioral endpoint.

## Declared metric

On a disjoint neutral construction pool, let `r_j` be the gradient of one length-normalized sequence log likelihood under the same repeated action. The first candidate uses the diagonal sequence-score metric

`M = diag(lambda + mean_j r_j odot r_j)`, with `lambda = max(1e-12, 0.01 mean_k mean_j r_{j,k}^2)`.

This is a positive-definite empirical surrogate. It is not called the exact output Fisher or exact sequence KL. Every view is normalized independently of the metric:

`q_tilde_v = q_v / ||q_v||_2`.

Keeping the normalized constraints fixed makes `robust_euclidean` versus `semantic_outcome` an isolated metric ablation. Raw gradient norms and metric reachabilities remain reported as difficulty diagnostics.

## Robust minimum action

For `Q = [q_tilde_1,...,q_tilde_m]`, the candidate direction is the unique solution

`min_u 0.5 u^T M u  subject to  Q^T u >= 1`.

Hard infeasibility, nonfinite gradients, invalid reachability, or failed KKT checks are serialized and never repaired by deleting a witness. No slack variant is authorized in version 1. Deployment factors `{0.25, 0.5, 1.0, 1.5}` multiply the frozen direction. Comparator directions are evaluated at the same measured metric cost whenever their action class permits it.

The Lagrange dual is

`max_{alpha >= 0} 1^T alpha - 0.5 alpha^T Q^T M^{-1} Q alpha`,

with `u = M^{-1} Q alpha`. Positive definiteness gives a unique primal action. If `q` is any convex combination of the normalized construction covectors, feasibility implies `q^T u >= 1`. This convex-hull statement does not cover an unseen witness outside the frozen hull.

If each score has a local remainder bound

`|s_v(u) - s_v(0) - q_v^T u| <= 0.5 L_v ||u||_2^2`,

then each normalized construction view has a finite-action lower bound obtained by subtracting its declared remainder term. The bound is conditional; no global smoothness or empirical value of `L_v` is assumed.

## Required ablations and competing explanations

- `pooled_euclidean`: mean witness covector under the identity metric;
- `pooled_sequence_metric`: mean witness covector under `M`;
- `robust_euclidean`: all witness constraints under the identity metric;
- `semantic_outcome`: all witness constraints under `M`;
- `last_token_semantic_outcome`: replaces sequence scores by the final candidate-token score;
- `DiffMean`: released AxBench mean-activation baseline;
- `prompt`: AxBench prompt steering;
- `FLAS`: released learned-flow checkpoint and exact generation path;
- `CES`: the strongest reproducible implementation of the indexed contrastive-energy objective, or an explicit unavailable-comparator receipt that blocks broad SOTA wording.

The candidate contribution requires both `semantic_outcome > pooled_sequence_metric` and `semantic_outcome > robust_euclidean`; otherwise it narrows to the supported component.

## Hypotheses

- SO.1: every returned action passes feasibility, stationarity, complementarity, diagonal/full consistency on small systems, factor-rescaling, and zero-action replay tests.
- SO.2: at least 90% of the 12 exposed sentinel concepts produce complete hard solutions without witness deletion, and no failure is hidden from coverage.
- SO.3: on disjoint teacher-forced witness pairs, the candidate improves the concept-level worst-view score over the strongest complete action ablation in at least 9 of 12 sentinel concepts, with positive mean paired change and no median neutral sequence-likelihood loss beyond the frozen margin.
- SO.4: on exposed tuning prompts, complete open generations show a positive concept-score change without an instruction or fluency sign reversal relative to the strongest feasible action ablation. This is auxiliary until the exact AxBench judge is available.
- SO.5: on 24 development concepts, the candidate exceeds the strongest fair reproduced baseline in concept-clustered HMean with a one-sided 95% interval above zero, while instruction, fluency, neutral likelihood, severe collapse, pass, and compute gates all pass.
- SO.6: the 64-concept internal pilot and 100-concept held-out AxBench panel remain unopened until one method, factor rule, judge, and comparator set are frozen from development.
- SO.7: a 2B result does not authorize 9B or cross-family wording. The frozen 9B protocol opens only after the 2B pilot passes; Qwen-family transfer is a later separate gate.

## Claim boundary

A sentinel pass supports only feasibility and exposed mechanism evidence. A development pass supports only a development comparison. SOTA requires the exact common AxBench held-out protocol, direct fair comparators, concept-clustered uncertainty, complete failures, and the registered 2B and 9B frontier gates.
