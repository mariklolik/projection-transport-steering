# InvariantBack pilot freeze, version 1

Freeze date: 2026-09-06. Frozen before implementation, dataset materialization, or model output.

## Model and independent units

The only pilot model is the pinned local Gemma-2-9B instruction-tuned checkpoint at `.external/hub/gemma-2-9b-it`. The independent unit is the source dataset group: NormBank setting-behavior group, MNLI premise group, or SC101 action group. No group crosses construction, development, sentinel, or untouched pilot-test allocations.

The exposed sentinel contains 8 groups per dataset. Development contains 24 different groups per dataset. The untouched pilot contains 48 further groups per dataset. Dataset extraction, source revision, exclusions, group hashes, and allocation seed `20260906` are materialized before any steering-method score is read.

## Encodings and endpoints

Construction uses identifier vocabularies `A/B/C`, `X/Y/Z`, and `1/2/3`, all six semantic-label-to-identifier mappings, and three fixed row permutations. Every prompt includes a mapping-comprehension check scored before steering. Covectors and `G_bar` use construction groups and views only.

Held-out evaluation uses the remaining three row permutations, the new identifier vocabularies `I/II/III` and `one/two/three`, a completion-only format, semantically equivalent prompt paraphrases, and an open-ended response format. Held-out groups are disjoint from construction groups, and the encoded likelihood endpoint is separate from the open-ended semantic endpoint. No held-out view contributes to covectors, Fisher pooling, nuisance estimation, layer choice, regularization, rank, slack choice, or strength.

## Candidate and baselines

- unsteered;
- CAA;
- exact same-state FishBack per view;
- mean-`q` Fisher action;
- nuisance-subspace projection/removal;
- SteerFair-style construction-bias removal;
- robust action with an identity metric;
- QuotientBack ablation;
- norm-matched random and construction-covariance-matched controls calibrated to the same JS budget;
- InvariantBack hard or the single prospectively contingent L1-slack action.

Released implementations are used when compatible with the identical residual site and data budget. Any adaptation is named and compared by equation, fit examples, trainable parameters, model passes, and search budget.

## Development selection

Candidate layers are `{12, 18, 24}`. Fisher ranks are `{32, 64, 128}`. Regularization multipliers are `{0.01, 0.1, 1.0}` times the pooled positive Fisher-eigenvalue median. The only slack penalties are `{1, 10, 100}`. Off-target JS budgets are `{1e-4, 3e-4, 1e-3, 3e-3}`.

Development first removes configurations with hard feasibility below 90%, mapping-comprehension lower interval below 0.80, or matched coverage below 80%. It selects the largest JS budget with valid coverage and then the configuration with the largest equal-dataset mean of per-group worst held-out-development semantic margin. Ties within `1e-6` choose lower rank, earlier layer, larger regularization, hard over slack, and lower compute in that order. The selected configuration and strongest feasible baseline are frozen before pilot-test access.

## Outcomes

The encoded semantic margin is the log-probability of the target semantic label minus the strongest non-target semantic label after mapping identifiers back to labels. Each view margin is standardized by its unsteered development-group standard deviation. The primary per-group statistic is the minimum standardized margin over all held-out encoded views at the frozen off-target Jensen-Shannon budget.

Secondary outcomes are semantic-label following, extraction-index following, extraction-row following, canonical-encoding margin, mapping comprehension, open-ended semantic score, hard feasibility, total slack, active constraints, action norm, off-target JS, wall time, model passes, peak memory, fit bytes, and search compute.

Operating points use monotone interpolation inside the observed strength bracket only. No extrapolation or nearest favorable point is allowed. A method-group without a bracket is missing and lowers coverage. All methods share the same target-token exclusions and off-target vocabulary.

## Uncertainty and strata

The primary statistic equal-weights datasets and groups. BCa bootstrap resamples independent groups within dataset 10,000 times with seed `20260906`, recomputes the per-group worst view and equal-dataset mean, and compares InvariantBack with the single strongest feasible baseline selected on development. One-sided alpha is 0.05. Secondary baseline contrasts use Holm correction.

Required strata are dataset, semantic label, identifier vocabulary, row permutation, completion format, paraphrase family, canonical correctness, base confidence quartile, action-norm quartile, and feasibility status. Attrition and severe group failures are reported before effect estimates.

## Gates

- IA.1: algebra, KKT, hashing, allocation, serialization, and replay tests pass; no pilot-test output exists during selection.
- IA.2: hard feasibility is at least 90%; the predeclared contingent rule is followed exactly when needed.
- IA.3: the primary one-sided 95% lower interval for improvement over the frozen strongest baseline exceeds `0.10` standardized units.
- IA.4: canonical performance ratio lower interval is at least `0.95`; mapping-comprehension lower interval is at least `0.80`; coverage is at least 80% overall and in each dataset and held-out view family.
- IA.5: extraction-index advantage decreases relative to both CAA and FishBack with one-sided upper intervals below zero, while semantic-label margin is non-inferior to each within `0.05` standardized units.
- IA.6: each dataset and held-out view family has a positive point estimate and none has a one-sided 95% upper interval below zero.
- IA.7: all required receipts and costs are complete; the candidate is not dominated simultaneously on primary margin, JS, and compute by a feasible baseline.

All gates pass or Pilot A closes. Multi-model scaling and a joint method are not authorized by this freeze.
