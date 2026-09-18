# Source audit

Audit target: source manuscript PDF and repository commit recorded in [`evidence-manifest.md`](evidence-manifest.md).

## What already exists

The source project implements additive steering, projection ablation, projection clamping, a Gaussian scalar transport, a piecewise-linear empirical quantile transport, coordinate-wise Linear-AcT, full-space MiMiC, trace-derived gates, and a gate/action sweep. It evaluates Gemma-2 2B and 9B variants on deterministic subsets of MMLU, ARC, GSM8K-MCQ, and GPQA. Its central decomposition is useful: estimate a behavior direction, detect when intervention is needed, and apply a scalar action along that direction.

The reusable contract is therefore not another family of ad hoc hooks. The new work must preserve a small controller interface:

1. an observer maps an activation prefix to a gate state;
2. an action maps a hidden state and gate state to a constrained residual update;
3. a runner records model passes, hook locations, tuning decisions, and paired outputs;
4. an evaluator decomposes detection, action, capability, and distribution-shift failures.

## Blocking empirical findings

| Finding | Evidence in source | Consequence |
|---|---|---|
| Representation leakage across reported subsets | `features_caa.extraction_records` excludes seeds 7, 11, and 23 with `eval_n=1000`, but the reported subset seeds also include 31 and 47. | Directions are not proven disjoint from every headline evaluation subset. |
| Statistics leakage across reported subsets | `extract_projection_stats.py` excludes only `--eval-seed 7`, `--eval-n 300` by default, while its directions, maps, and gates are reused for seeds 11, 23, 31, and 47. | Headline robustness estimates are contaminated unless every fitted artifact is rebuilt from a globally disjoint fit split. |
| Selection on evaluation data | `steer_v5_sweep.py` evaluates two gates and three actions on the same records used in the result analysis. | The selected headline condition is exploratory, not confirmatory. |
| Seeds are not independent runs | `paper/make_multiseed.py` explicitly calls them data-subset seeds under deterministic greedy decoding. Subsets are drawn from one benchmark and can overlap. | Mean plus sample standard deviation across five subsets is not an uncertainty estimate over independent model runs. |
| Conditional steering is two-pass | The sweep first reconstructs baseline traces, forwards them again to compute trace projections, then regenerates flagged examples under the hook. | The conditional method cannot be described as requiring no extra forward pass. One-pass and two-pass variants must be separated. |
| Tail support claim conflicts with code | `general/steering.py` linearly extrapolates outside its source quantile knots. | The implemented empirical map can leave the observed target support. |
| Architecture path is hard-coded | Hook sites use `model.model.layers[layer]`. | Portability to modern architectures is not established. |
| AUROC tie handling is incomplete | The local ranking implementation does not apply the standard half-credit treatment for tied positive-negative scores. | Detector estimates can be biased when gate scores are quantized or repeated. |
| Non-inferiority was not tested | Accuracy preservation is argued from non-significance around zero. | “No accuracy cost” is unsupported without a declared margin and one-sided paired interval/test. |
| Raw lineage is incomplete | Raw rollouts were located for the principal result only; other subset reports are aggregates. | Independent reconstruction of every headline cell is currently impossible. |
| Readout identifiers can masquerade as semantics | Directions are extracted from traces containing multiple-choice answer identifiers and confidence tokens. | Every result needs a cross-encoding control that remaps answer labels while preserving task semantics. |
| Source positions mix decision and realization | Direction extraction pools generated-token activations rather than isolating prompt, decision-boundary, and post-answer states. | Source position becomes a preregistered factor; execution-boundary and tail-subtracted sources must be compared on held-out development data. |

## Blocking formal findings

| Source statement | Required correction |
|---|---|
| Atomlessness gives a unique Brenier map in arbitrary dimension. | In one dimension, monotone rearrangement needs an atomless source for a deterministic map. In higher dimensions, the standard Brenier uniqueness result requires absolute continuity of the source with respect to Lebesgue measure. |
| The population transport theorem justifies the empirical 41-knot implementation. | A separate finite-sample statement is required for the fitted interpolant, including ties, truncation or extrapolation, and approximation error. |
| Detector AUROC or a Youden threshold bounds behavioral selectivity. | Detection optimality does not imply downstream utility. A link requires explicit causal-response, monotonicity, or regret assumptions and must be tested with an oracle-gate decomposition. |
| A Fisher or Gauss-Newton quadratic bounds KL. | The quadratic is a local approximation unless additional regularity and trust-region bounds are proved. |
| Reading and updating a dense residual vector proves end-to-end compute optimality. | It proves at most a restricted dense-vector access lower bound. Model forwards, gate computation, controller fitting, and comparison methods must be counted separately. |

## Correct formal core available now

- For scalar distributions with finite second moment and an atomless source, the monotone rearrangement is the almost-everywhere unique monotone quadratic-cost transport.
- For a unit direction `v` and desired scalar projection `z`, the unique Euclidean minimum-norm update satisfying `v^T h' = z` is `(z - v^T h)v`.
- For a symmetric positive-definite metric `Q`, the exact minimum-`Q`-norm update satisfying `v^T delta = a` is `a Q^{-1}v / (v^T Q^{-1}v)`.
- These statements establish optimality inside a declared action constraint. They do not establish that the direction is causal, the target distribution is desirable, or the downstream behavior improves.

## Clean-room wiring decision

The target implementation will reuse the observer/action/runner/evaluator contracts conceptually, not copy source modules. It will first add tests for split disjointness, architecture resolution, pass accounting, transport tie/tail behavior, and paired uncertainty. Only the smallest implementation needed to satisfy those contracts will follow.
