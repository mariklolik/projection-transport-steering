# ResidualFlowBack design audit, version 1

Decision date: 2026-09-07. Status: valid post-hoc design result; ResidualFlowBack version 1 closed before fresh development or generation.

## Contract and provenance

The source packet is `artifacts/source/semantic_outcome_v1_sentinel.json`, SHA-256 `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`. The reused matched-continuation packet retains its registered listing SHA-256 `d644c6d3132db749a008afaa7db17c89a733e8689b31e3bf055c78341649fa77`. Released FLAS uses commit `720ef8a67697d9b94130b374b5b3a1522a782566`, checkpoint configuration SHA-256 `d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a`, and weights SHA-256 `bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e`.

The admitted engineering smoke is `artifacts/development/residual_flow_v1_smoke_c1_attempt1`, listing SHA-256 `d753bc2f03fec31d1511daf30973d09372bb1ef1a0ec6eea3362838ed1dd2b53`. The six-shard design packet is `artifacts/development/residual_flow_v1_design_attempt1`, listing SHA-256 `82743d4f2f36c3717b2990cbb0f221751d998ef0326f6d4351722c861fe14564`. Each listing digest hashes the sorted `shasum -a 256` output for every regular file relative to its packet root. The prospective analysis is `artifacts/development/residual_flow_v1_design_analysis_attempt1/analysis.json`, SHA-256 `a7e73a03f00964c0940843a51dce96b4fd3cb501969772bcd9fea5881250c74c`.

## Integrity and compute

All 12 planned concepts, 36 quantile systems, and 180 method-quantile evaluation cells completed. Zero-residual replay and action serialization errors are exactly zero in every shard. Every candidate system is feasible, all declared construction constraints are covered, and the largest reported candidate KKT residual remains within the solver tolerance. Local and remote hashes match for all 12 action and receipt files.

Six H100 workers each loaded Gemma-2-2B-IT and FLAS once for two concepts. The packet uses 288 backward products and 720 explicitly counted teacher-forced model forwards. Maximum parallel shard wall time is 17.40 seconds and maximum allocated memory is 32.67 GB. Two H100s remained unused. No matched continuation was regenerated and no fresh-development, pilot, held-out, generation, judge, or 9B row was opened.

## Frozen gate

The endpoint is concept-level worst-view score change relative to unchanged FLAS. Intervals are fixed-seed concept bootstrap intervals and remain descriptive because the 12 concepts are post-hoc design evidence.

| Target quantile | Candidate mean worst-view change vs FLAS, 95% interval | Wins vs FLAS | Candidate minus pooled mean worst-view change, 95% interval | Wins vs pooled | Median neutral likelihood change | Added median neutral KL | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | +0.06596 [0.04580, 0.08589] | 12/12 | -0.00855 [-0.01673, 0.00122] | 2/12 | -0.05338 | 0.00890 | fail |
| 0.75 | +0.23999 [0.12692, 0.41432] | 12/12 | -0.03856 [-0.05982, -0.01862] | 2/12 | -0.15080 | 0.03807 | fail |
| 1.00 | +0.60074 [0.43019, 0.77273] | 12/12 | -0.13551 [-0.20803, -0.06004] | 2/12 | -0.42313 | 0.27147 | fail |

No quantile reaches the required 9/12 wins over pooled residual steering. All quantiles violate both the `-0.02` neutral-likelihood and `0.002` added-KL limits. The smallest quantile is therefore not a near pass: only one of 12 concepts meets both safety limits, and the candidate loses to pooled residual steering on ten concepts.

## Mechanism diagnosis

Released FLAS already improves the construction contrast for every one of the 96 witnesses. Mean finite construction change is `+1.90458` nats per token and the negative-change fraction is exactly zero. The residual targets therefore do not repair failed FLAS outcomes. They spend correction cost equalizing the weaker members of an already successful construction set.

At quantile 0.50, the candidate and deficit-weighted pooled action have mean cosine `0.9596`, yet pooled has larger mean worst-view change, `0.07450` versus `0.06596`. The candidate beats Euclidean residual steering in 9/12 concepts by a mean `+0.00691`, again isolating the useful trajectory-local metric component. It beats shuffled deficits in only 5/12 and trails them by mean `-0.00567`; witness-specific deficit assignment is not predictive of held-out benefit. Mean active constraints are `2.92` of eight and dual concentration is `0.584`, so failure is not caused by an overconstrained or numerically unresolved system.

Increasing the target raises both efficacy and damage. Across concepts, candidate metric cost has descriptive Pearson correlation `0.585` with worst-view gain, `-0.799` with neutral likelihood change, and `0.948` with added neutral KL at quantile 0.50. The cost-KL correlations remain `0.978` and `0.939` at quantiles 0.75 and 1.00. The higher-quantile apparent efficacy is therefore a dose-damage tradeoff, not evidence for the residual assignment.

The smoke and full packet independently repeat concept 1. Finite base and FLAS construction contrasts are exactly equal across the two executions. Gradient norms vary by at most `4.42e-6`, matched action cosines are at least `0.999992`, and individual teacher-forced changes vary by at most `0.00980`. Every gate decision for that concept is unchanged. This is bounded mixed-precision repeat sensitivity, not permission to average, rerun, or select a favorable result.

## Decision

Claim C55 is contradicted under its frozen design gate. ResidualFlowBack version 1 is closed without changing the quantile set, cost, targets, witness mapping, safety limits, comparator, flow time, Euler steps, metric, layer, or source rows. The next 24 concept IDs remain unmaterialized. A successor must change the controlled object rather than tune dose or rebalance construction witnesses. The retained evidence is narrow: a small trajectory-local metric correction can improve teacher-forced worst-view score over unchanged FLAS, but pooled correction is stronger and the added correction is unsafe under the declared limits.
