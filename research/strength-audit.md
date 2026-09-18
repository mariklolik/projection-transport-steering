# Research strength audit

Scale: 0 absent, 1 weak, 2 partial, 3 strong. This audit concerns the source evidence inherited at project start, not future target-repository results.

## Empirical readiness

| Dimension | Score | Evidence |
|---|---:|---|
| Task breadth | 2 | Four QA benchmarks, but one behavior construct dominates. |
| Model breadth | 1 | Two sizes from one older architecture family; secondary branches do not form one sealed protocol. |
| Strong baseline coverage | 1 | CAA, CAST-style gating, MiMiC, and Linear-AcT appear; prompt, ReFT, SADI, COAST, A-LQR, COBRAS, and PCHI are absent. |
| Confirmatory separation | 0 | Gate and action selection occur on evaluated records, and fit/evaluation disjointness is incomplete. |
| Uncertainty | 1 | Subset variability and some intervals are reported, but overlapping deterministic subsets are treated like replications. |
| Robustness and shift | 1 | Multiple benchmarks help, but there is no sealed OOD, paraphrase, adversarial, or temporal shift protocol. |
| Failure decomposition | 1 | Gate quality and selectivity are present, but oracle observer/action and causal treatment-effect decomposition are absent. |
| Construct validity | 1 | Multiple confidence readouts exist, yet wrong-confidence labels and downstream behavioral benefit are not fully separated. |
| Reproducibility and receipts | 2 | Substantial code and main raw rollouts exist; multi-subset raw lineage and environment receipts are incomplete. |

Empirical total: 10/27. State: `EVIDENCE_THIN`.

## Formal readiness

| Dimension | Score | Evidence |
|---|---:|---|
| Problem definition | 2 | Detection and action are separated, but desired causal utility is not fully identified. |
| Assumption clarity | 1 | Several assumptions are implicit or broader than the theorem permits. |
| Population result correctness | 1 | The one-dimensional core is recoverable; higher-dimensional wording is incorrect. |
| Finite-sample connection | 0 | No bridge from population OT to the fitted knot map. |
| Conditional-utility theory | 1 | Detector optimality is established under a classification model, not steering utility. |
| Geometry and cost | 2 | Projection-constrained minimum-cost action has a clean exact formulation. |
| Dynamics and causal mechanism | 1 | Static hooks are studied without a validated state-dynamics or causal abstraction. |
| Computational statement | 1 | Hook arithmetic is characterized, but end-to-end cost and comparison class are not. |
| Falsifiability | 2 | The decomposition suggests clear oracle and stress tests. |

Formal total: 11/27. State: `EVIDENCE_THIN`.

## Promotion gates

The project cannot enter paper-polish mode until all of the following are true:

- every fit, tuning, development, and confirmatory ID set is disjoint and hashed;
- one-pass and two-pass methods are instrumented and reported separately;
- three modern model families and at least three distinct behavior constructs pass the same protocol;
- prompt, additive, conditional, affine-transport, nonlinear-transport, finetuning, and task-specific closest baselines receive matched budgets;
- every encoded task passes cross-encoding and every primary cell passes matched-KL construction-null specificity tests;
- observer policies improve held-out action loss over always-act, never-act, and observational score-threshold policies;
- the primary paired metric has stratified confidence intervals and a declared capability non-inferiority margin;
- the frozen collateral panel passes mean and protected-tail safety gates with severe events disclosed;
- every headline cell has raw outputs, environment, command, model revision, and analysis receipts;
- formal statements are restricted to their actual estimator, action class, and assumptions;
- an independent review finds no unresolved fatal flaw in novelty, validity, leakage, or claim support.

## Target-repository checkpoint, development v1

The target implementation passed 33 dependency-light tests locally and 40 tests, including Torch hooks and tensor transport, on `avi-gn-fsk40`. A real Qwen-2.5 checkpoint smoke test exercised the same-forward hook path. These receipts promote implementation readiness only; they do not promote empirical readiness.

The first disjoint 1,000-group MMLU-Pro development run failed the observer and joint-method gates. The best action-value policy had Brier 1.08863 versus 1.08859 for fixed `additive_2`, while the per-item oracle reached 0.88907. The result leaves the empirical state `EVIDENCE_THIN` and blocks confirmatory execution. Full receipts and intervals are in `research/development-result-audit-v1.md`.

## Target-repository checkpoint, TruthfulQA sequence

Seven versioned observer-stage candidates were evaluated without opening the frozen development allocation. Empirical spherical quantile transport, misaligned and deployment-aligned posterior gates, and a final cutoff correction all failed their registered screens against expanded matched Spherical Steering. Fair baseline expansion improved observer MC2 from 0.72480 to 0.73688.

A final parameter-free envelope then received a one-shot 120-question development comparison. It matched MC1 and improved calibrated MC2 by 0.00133, but the paired 95% BCa interval was [-0.00861,+0.01116]. The superiority gate failed, no remaining development or confirmatory runs were authorized, and the project remains `EVIDENCE_THIN`. Full receipts are in `research/development-result-audit-v8.md`.
