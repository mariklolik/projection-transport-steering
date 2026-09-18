# ResidualFlowBack benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before any ResidualFlowBack output.

## Provenance and allocation

The 12 SemanticOutcomeBack sentinel concepts are now an explicitly post-hoc design set. Their source packet SHA-256 is `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`; their immutable matched-continuation Stage A packet listing SHA-256 is `d644c6d3132db749a008afaa7db17c89a733e8689b31e3bf055c78341649fa77`. They may select one residual target quantile but cannot support confirmation.

The next 24 published held-in IDs remain unmaterialized and unread for this route: `74,76,82,84,85,87,91,114,118,123,131,132,134,140,141,142,143,144,157,169,174,182,194,204`. They form the fresh successor-development allocation only if the design gate passes. The remaining 64 held-in concepts, all 100 held-out concepts, and all 9B allocations remain unopened.

FLAS source is commit `720ef8a67697d9b94130b374b5b3a1522a782566`; `generate.py` SHA-256 is `14e1bde99c119e6b970cf7601569b6ab05bedfbd44d65286044af52936f529f7`; `model.py` SHA-256 is `9058008c85835fadeb0ac9737ebd711520fecbf09b92ab8ba3e6ba1bc775dcd3`. The 2B checkpoint configuration SHA-256 is `d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a`; weights SHA-256 is `bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e`. The base model and tokenizer use the nine hashes in `semantic-outcome-source-receipt-v1.md`.

## Frozen design execution

Use Gemma-2-2B-IT, layer 20, released FLAS flow time 2, three Euler steps, and the released concept encoder. Reuse every version-1 matched positive and negative continuation byte-for-byte. For each concept, compute finite FLAS construction changes, trajectory-local correction covectors, and 16 trajectory-local neutral score gradients. Inspect target quantiles only at `{0.50,0.75,1.00}`.

Evaluate unchanged FLAS, candidate residual metric, deficit-weighted pooled residual metric, Euclidean residual, shuffled deficits with `SeedSequence([20260907, concept_id])`, and the retained pooled sequence-metric action. Candidate and correction controls are compared at identical trajectory-local correction metric cost. Unchanged FLAS has zero added correction cost and remains the primary practical baseline. Report both each control's unscaled solver receipt and its realized rescaled margins; only the candidate's hard construction constraints carry a certificate.

The design endpoint is held-out positive-minus-negative length-normalized continuation-score change relative to unchanged FLAS, including the worst of eight views per concept. Safety endpoints are neutral likelihood and forward tokenwise KL relative to unchanged FLAS and base, finite outputs, action norm and metric cost, active constraints, residual coverage, hook replay, wall time, peak memory, and all model/backward passes.

## Analysis and gate

All 12 design concepts and three target quantiles are reported. The independent unit is a concept. Intervals resample concepts 10,000 times with seed `20260907` and are labeled post-hoc exploratory. No factor, row, or comparator can be removed.

The frozen diagnostic packet also reports construction-change location and dispersion, negative-witness fraction, residual-deficit distribution, active-constraint count, dual concentration, realized linearized coverage after cost matching, action cosines, and candidate effects in lower versus upper halves split by the median pre-correction FLAS construction mean and construction range. These median-split summaries are descriptive heterogeneity checks only and cannot enter the gate or select a quantile.

One target quantile advances only if at least 11/12 concepts are complete; the candidate beats unchanged FLAS and pooled residual metric on held-out worst-view change in at least 9/12 concepts; both mean paired changes are positive; median neutral likelihood relative to FLAS is at least `-0.02` nats per token; added median neutral forward KL is at most `0.002`; all invariant and resource receipts pass; and no hidden failure is present. If several pass, choose the smallest quantile. This tie-break is frozen before output.

Failure closes ResidualFlowBack version 1 without changing target quantiles, witness count, source rows, FLAS time or steps, metric ridge, layer, margins, comparators, or gate. A pass authorizes materializing the fresh 24-concept source and freezing its generation/judge analysis; it does not authorize behavior or novelty wording.

## SOTA boundary

The registered AxBench markers remain `1.015` at 2B and `1.120` at 9B, with a complete common-protocol comparison against the strongest reproduced method. Direct FLAS, CES, UniSteer compatibility, prompt, DiffMean, retained pooled sequence metric, and released learned baselines are required. No teacher-forced design or local judge result substitutes for the exact released GPT-4o-mini behavior evaluation.
