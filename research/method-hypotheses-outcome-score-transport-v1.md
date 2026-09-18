# Outcome-score transport hypotheses, version 1

Freeze date: 2026-09-07. Status: working method hypothesis before result-generating implementation.

## Estimand and method

Let `R` be executable final-answer correctness and let `z_t` be the residual-stream state at the frozen intervention layer after generation step `t`. A causal prefix model estimates

`v_t = P(R = 1 | z_1, ..., z_t, progress_t)`.

Training examples are prefixes of complete fit-split rollouts. Query-time inputs use the same causal prefix contract. Progress bins are quartiles of the generation token budget, fixed before fitting.

Within each progress bin, fit an empirical monotone map `g_b` from low-value fit prefixes toward the successful-prefix value distribution. At a gated state, the desired local score change is

`d_t = clip(g_b(v_t) - v_t, 0, d_max)`.

For score gradient `a_t` in a frozen low-rank activation basis and positive-definite pooled metric `M`, apply

`delta_t = d_t M^-1 a_t / (a_t^T M^-1 a_t)`.

The action is skipped when `d_t = 0`, the gradient denominator is below the frozen numerical floor, the value is outside the calibration support, or the intervention budget is exhausted. There is no dataset-specific layer, dose, step count, or threshold.

## Primary hypotheses

- OST.1: the prefix-value observer passes held-out calibration with finite complete predictions, AUROC above `0.65`, Brier improvement above the constant-rate predictor with a one-sided paired bootstrap lower endpoint above zero, and no progress quartile below `0.55` AUROC.
- OST.2: on AIME 2024 development, full outcome-score transport improves paired exact-answer correctness over the strongest feasible matched baseline, with a one-sided exact McNemar `p < 0.05` and a one-sided question-bootstrap 95% lower endpoint above zero.
- OST.3: candidate degradation among baseline-correct questions is at most one question and its one-sided 95% upper bound is below `0.10`; severe parse failures, empty completions, and numerical failures are zero.
- OST.4: the full candidate has lower mean and 95th-percentile output KL than Euclidean score transport at matched achieved score increment, without lower intervention coverage or correctness.
- OST.5: the full candidate beats fixed-dose Fisher action, normalized Euclidean reward gradient, and transport-dose Euclidean action; no progress, baseline-correctness, difficulty, response-length, or gate-score stratum has an unexplained sign reversal.
- OST.6: after a development pass, the frozen candidate is non-inferior to the strongest matched baseline on AIME 2025 and superior on at least one of AIME 2025, AIME 2026, or MATH-500 under multiplicity-adjusted uncertainty.

OST.1 is required before any steering development. OST.2-OST.5 are jointly required before temporal or breadth evaluation. OST.6 is necessary but not sufficient for a SOTA claim.

## Mechanism decomposition

The common packet contains:

1. base greedy decoding;
2. fixed chain-of-thought prompt;
3. contrastive activation addition;
4. rollout-weighted fixed policy-gradient direction with scalar Fisher calibration;
5. LRS-like normalized Euclidean value-gradient ascent;
6. fixed-dose minimum-metric value-gradient action;
7. transport dose with Euclidean action;
8. full transport dose with minimum-metric action;
9. outcome-value gate without action;
10. norm-matched random and shuffled-score controls.

All methods share questions, outputs limits, layer, fit information, and one frozen intervention budget. Output KL, forward and backward products, generated tokens, wall time, peak memory, parameter bytes, and interventions are reported rather than collapsed into one score.

## Stop rules

Close the route without rescue if the sentinel lacks both success classes, the observer fails OST.1, exact baseline replay fails, the candidate is numerically invalid on any required row, AIME 2024 OST.2 fails, a matched ablation explains the whole gain, or protected degradation fails OST.3. Do not change the layer, rank, progress bins, transport clip, metric ridge, gate, parser, prompt, or evaluation allocation after observing the corresponding stage.
