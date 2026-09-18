# InvariantBack sentinel protocol, version 1

Freeze date: 2026-09-06. Frozen after data allocation and replay smoke, before any nonzero intervention or view-covector result.

## Purpose and inputs

The sentinel checks only whether the pre-answer Gemma residual site, semantic view gradients, and robust constraints are operationally coherent. It cannot select a layer, Fisher rank, regularization, slack penalty, strength, JS budget, method, or baseline and cannot support a performance claim.

The model is the pinned local Gemma-2-9B-IT checkpoint. The public data packet SHA-256 is 472cc968c8bce46e8e2a59c7cdd8bb99cae6b8c674bd6995d2be402dca0ce259. The sealed pilot packet is not copied to the GPU host.

For each dataset, the q-sentinel subset is the first eight already ordered construction groups in the immutable public packet. This is separate from the eight evaluation sentinel groups. No evaluation-sentinel group contributes to a covector.

## Cells and views

The 27 cells cross three datasets, the three ordered label contrasts, and layers 12, 18, and 24. Each cell estimates 54 semantic target covectors by crossing all six label-to-identifier mappings, construction vocabularies A/B/C, X/Y/Z, and 1/2/3, and the three frozen construction row permutations. The prompt wording is construction paraphrase 0. Each covector is the gradient of the mean target-minus-source candidate-sequence log likelihood on the source endpoint of the eight q-sentinel groups.

Candidate sequences are teacher-forced and scored by mean token log likelihood. The action is added once to the block-output residual at the final token of the prefix ending in Answer:. Numeric identifiers are not forced to be one token. Model parameters are frozen.

## Sentinel geometry and gates

Every covector is normalized by its identity-metric norm. The exact hard robust QP uses target rho=1 and an identity metric, solved in the QR/SVD span of the 54 covectors by the already tested dual/KKT primitive. This is a feasibility diagnostic, not the Fisher candidate.

- zero-action replay max absolute logit error must be exactly zero;
- every covector must be finite and nonzero;
- every returned feasible solution must have stationarity and complementarity residual at most 1e-5 and minimum normalized-view margin at least 1-1e-5;
- each dataset and ordered contrast must have at least one feasible layer;
- shard hashes, view counts, forward counts, wall time, and peak memory must be complete.

Any cell may be infeasible without closing the route because layer selection is reserved for development. A dataset-contrast with no feasible layer, a replay failure, a nonfinite covector, or an incomplete shard closes Pilot A. Passing opens a separate mapping-comprehension sentinel; it does not open development by itself.

