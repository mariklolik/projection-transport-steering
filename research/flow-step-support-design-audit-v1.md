# Flow-step support design audit, version 1

Decision date: 2026-09-07. Status: valid post-hoc design result; Flow-step support version 1 closed before fresh development or generation.

## Contract and provenance

The source packet is `artifacts/source/semantic_outcome_v1_sentinel.json`, SHA-256 `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`. The byte-identical Stage A matched continuations retain listing SHA-256 `d644c6d3132db749a008afaa7db17c89a733e8689b31e3bf055c78341649fa77`. Released FLAS uses commit `720ef8a67697d9b94130b374b5b3a1522a782566`, configuration SHA-256 `d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a`, and weights SHA-256 `bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e`.

The first smoke packet, `artifacts/development/flow_step_support_v1_smoke_c1_attempt1`, listing SHA-256 `ac2153c55efa4ebdc5768813419937d1596ddd93e9e6727b79f730ad489229ef`, failed before any condition output because an explicit `scale=1.0` introduced an unnecessary bfloat16 net-displacement recomposition in the replay control. The accepted smoke is `artifacts/development/flow_step_support_v1_smoke_c1_attempt2`, listing SHA-256 `dd59ecdc4e27b3c100fcacea6d80c555b2627d3a99b9cc50787574fcd47ea5e5`, with exact replay, 47 expected forwards, 9.47 seconds, and 34.05 GB peak allocated memory. The repair removed scaling from raw subset interventions; it did not change the equal-time candidate or any scientific threshold.

The accepted six-shard packet is `artifacts/development/flow_step_support_v1_design_attempt1`, listing SHA-256 `c1ae541f22a0deafef3a5215bea02113fd0a64b755354e603e7e7618d48187b5`. The frozen analysis is `artifacts/development/flow_step_support_v1_design_analysis_attempt2/analysis.json`, SHA-256 `4e0b7297271834cbb38fd69ce2acfe69116e0538e6c01ee49d86959449821520`. The first analyzer invocation failed at import time because a pure analyzer imported a GPU runner and inherited its optional `flas` dependency; it created no analysis artifact. The accepted analyzer is self-contained and does not alter receipts, endpoints, candidate identity, or gates.

## Integrity and compute

All 12 planned concepts and all 168 concept-condition cells completed: seven raw subsets, six equal-time supports, and the released `N=2` condition per concept. Every shard has exact full-path replay, exact JSON serialization, finite outputs, the registered source hashes, and the expected 92 teacher-forced base-model calls. Local and remote hashes match for all six receipts. Exact three-player Shapley reconstruction has maximum absolute error `4.44e-16`.

Six H100 workers each loaded Gemma-2-2B-IT and FLAS once for two concepts. The complete packet uses 552 counted teacher-forced forwards, no gradients, no generations, and no judge calls. Maximum parallel shard wall time is 18.13 seconds, summed shard time is 103.89 seconds, and maximum allocated memory is 35.75 GB. GPUs 0--5 ran concurrently at 98--99% observed utilization while GPUs 6--7 remained free. The next 24 concepts, all pilot and held-out concepts, 9B models, generation, and judging remain unopened.

## Frozen candidate gate

The fixed candidate is equal-time late support `{1,2}`. Intervals resample concepts with the frozen 10,000-draw seed and remain descriptive because these 12 concepts are post-hoc design evidence.

| Gate component | Result | Decision |
|---|---:|---|
| Candidate minus full-path held-out worst view | `+0.07779` [`+0.00725`, `+0.14556`], 9/12 wins | FS.2 pass |
| Candidate neutral-KL fractional reduction from full | `-0.35045`, 0/12 reductions | FS.3 fail |
| Candidate minus full neutral likelihood | `-0.31706` [`-0.34384`, `-0.29037`], 0/12 wins | FS.3 fail |
| Candidate minus equal-time early `{0,1}` held-out worst view | `-0.41045` [`-0.54385`, `-0.27560`], 1/12 wins | FS.4 fail |

The candidate clears the efficacy comparison with full FLAS, including a positive descriptive interval, but fails both safety components on every concept and loses the matched support control on 11 of 12 concepts. The joint gate therefore fails decisively.

## Complete condition summary

All values are concept means. Neutral columns use the per-concept median across the 16 neutral rows.

| Condition | Construction contrast change | Held-out mean change | Held-out worst change | Neutral likelihood change | Neutral forward KL |
|---|---:|---:|---:|---:|---:|
| `raw_0` | 1.9761 | 1.8264 | 1.1404 | -1.0668 | 1.2922 |
| `raw_1` | 1.2107 | 1.0861 | 0.6764 | -0.2492 | 0.3132 |
| `raw_2` | 0.4089 | 0.3591 | 0.2149 | -0.0300 | 0.0333 |
| `raw_01` | 1.8693 | 1.6880 | 1.0569 | -0.6157 | 0.7868 |
| `raw_02` | 1.8090 | 1.6283 | 1.0110 | -0.5588 | 0.7037 |
| `raw_12` | 1.4197 | 1.2655 | 0.7822 | -0.3767 | 0.4446 |
| `full` | 1.9046 | 1.7040 | 1.0635 | -0.6314 | 0.7969 |
| `equal_0` | 6.7524 | 6.7889 | 3.3349 | -12.0124 | 12.4315 |
| `equal_1` | 2.8372 | 2.7426 | 1.7351 | -2.6953 | 2.7968 |
| `equal_2` | 1.0915 | 0.9982 | 0.6234 | -0.3472 | 0.3501 |
| `equal_01` | 2.7391 | 2.5404 | 1.5518 | -1.5494 | 1.8743 |
| `equal_02` | 2.5897 | 2.4058 | 1.4877 | -1.4162 | 1.7044 |
| `equal_12` | 1.9621 | 1.8046 | 1.1413 | -0.9484 | 1.0736 |
| `official_n2` | 2.1261 | 1.9520 | 1.1926 | -0.9442 | 1.1771 |

## Mechanism diagnosis

Raw late support `{1,2}` exposes the underlying tradeoff. Relative to the full path it lowers neutral KL in 12/12 concepts by mean fractional reduction `0.44064` and improves neutral likelihood in 12/12 by mean `+0.25468`, but it loses held-out worst-view efficacy in 12/12 by mean `-0.28133`. Applying the frozen `1.5` equal-time rescale recovers `+0.35912` held-out worst-view change relative to raw `{1,2}`, while adding `+0.62898` neutral KL and decreasing neutral likelihood by `-0.57174`. The safety failure is therefore tied to nominal-mass restoration, not to a failure of late-step removal to reduce raw damage.

Exact raw-subset Shapley allocation gives step 0 about 63.0% of full held-out worst-view efficacy, 92.7% of full neutral KL, and 93.4% of the full neutral-likelihood loss. Step 1 contributes about 30.4% of efficacy and 15.0% of KL. Step 2 contributes about 6.6% of efficacy and has negative average marginal allocations for KL and likelihood damage because of interactions with earlier steps. These are algebraic allocations of the implemented set function, not independent causal effects of Euler increments.

Candidate efficacy gains over full occur for concepts `1,16,24,29,33,58,59,60,73`; losses occur for `5,34,48`. The sole win over equal-time early support is concept `33`. In an explicitly post-hoc median split, candidate-minus-full efficacy is `+0.14815` in the half with stronger full-path efficacy and only `+0.00743` in the weaker half; the corresponding split by full-path neutral KL is `+0.13635` versus `+0.01923`. This exploratory concentration says the fixed rescale mainly amplifies already strong, already damaging cells. It cannot define a routing rule or reopen the candidate.

The released `N=2` condition is stronger than the `N=3` full path on the mean endpoints but also more damaging. Because `N=2` changes both the Euler grid and state trajectory, this is a numerical-schedule diagnostic rather than evidence that two particular `N=3` steps are sufficient.

## Decision

Claim C56 is contradicted under its frozen gate. Flow-step support version 1 is closed without selecting another subset, tuning the `3/|S|` factor, using per-concept support, changing flow time, weakening safety, or adding a residual correction. Fresh development, generation, judge, pilot, held-out, 9B, novelty, behavior, and SOTA stages remain blocked. The retained result is a narrow mechanism finding: early FLAS support dominates both teacher-forced efficacy and collateral distribution shift, while removing it creates a real efficacy-safety tradeoff that fixed equal-time scaling does not solve.
