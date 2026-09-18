# Outcome-score interface audit, version 3

Audit date: 2026-09-07. Decision: close version 3 before class-mix generation.

## Frozen question

Does the version-3 Qwen3-8B non-thinking interface produce eight complete, objectively parseable AIME outcomes within 2,048 tokens? The gate required 8/8 terminal integer parses, no cap hits, complete finite traces, exact zero-action replays, complete hash-bound packets, and peak memory below 70 GiB.

The model, mode, cap, prompt, scorer grammar, row IDs, seeds, sampling, and stop rules were frozen in `benchmark-freeze-outcome-score-interface-v3.md` before implementation or output.

## Accepted packet

- Host: `avi-gn-fsk40`.
- Hardware: eight NVIDIA H100 80GB HBM3 GPUs, one resident model and one row per GPU.
- Raw packet: `artifacts/development/outcome_score_v3_termination_attempt1`.
- Accepted analysis: `artifacts/development/outcome_score_v3_termination_analysis_attempt1/analysis.json`.
- Packet-listing SHA-256: `3371c13b40aa170a09d46d33aac697a88a5b5db8cc43508fd4ab8ba961bc06be`.
- Analysis SHA-256: `012f6514d9d55933f747189c24d21ed2d88dbae94a34051a68ed3cd82c80fb01`.
- The accepted remote analysis was reproduced byte-for-byte locally.

All eight shards completed with pass receipts. All eight traces are finite, align one-to-one with generated tokens, serialize exactly, and have unique keys. All eight zero-action replays are exact. No engineering retry or excluded scientific row exists.

## Result

| Shard | Year | Tokens | Terminal content | Frozen parse | Exact outcome |
|---:|---:|---:|---|---|---|
| 0 | 2015 | 1,452 | `\\boxed{307}` | pass | correct |
| 1 | 2008 | 1,924 | `\\boxed{1786}` | fail: outside `[0,999]` | undefined under frozen scorer; integer differs from gold 504 |
| 2 | 1986 | 2,048 | unfinished | fail: cap | undefined |
| 3 | 1996 | 2,048 | unfinished | fail: cap | undefined |
| 4 | 1983 | 588 | `\\boxed{15}` | pass | correct |
| 5 | 2018 | 1,021 | `\\boxed{150}` | pass | incorrect; gold 600 |
| 6 | 2008 | 2,048 | unfinished | fail: cap | undefined |
| 7 | 1985 | 1,526 | `\\boxed{1985}` | fail: outside `[0,999]` | undefined under frozen scorer; integer differs from gold 32 |

The primary result is 3/8 valid parses and 3/8 cap hits, so both mandatory interface gates fail. Of the three valid rows, two are correct and one is incorrect. Five rows terminate with an unambiguous boxed integer, but the two values above 999 are intentionally rejected by the frozen parser. Their disagreement with gold is a post-result diagnostic, not a version-3 label.

Generated lengths are `[588, 1021, 1452, 1526, 1924, 2048, 2048, 2048]`; median is 1,725, mean is 1,581.875, and total main-generation tokens are 12,655. All three cap hits occur in different years; both sampled 2008 rows fail for different reasons, one out-of-range terminal integer and one cap hit. Only rollout index 0 was authorized, so no within-question or seed heterogeneity is estimated.

## Cost and GPU efficiency

Workers completed in 38.083 to 110.004 seconds. Summed worker time is 683.578 seconds, or 0.190 H100-hours; the parallel wall upper bound is 110.004 seconds. The serial-to-parallel wall ratio is 6.21 and worker occupancy over the eight-GPU envelope is 77.7%. Unequal generation lengths cause the lower occupancy.

Compared with version 2, non-thinking inference reduces parallel wall time by 47.3%, summed worker time by 58.0%, and instrumented forward calls by 60.7%. Exact replays account for half of the 25,310 forward calls. Maximum allocated memory is 16,768,290,816 bytes, or 15.62 GiB. All GPUs were released before analysis.

## Interpretation

Version 3 weakens neither the terminal-integer measurement principle nor the Outcome Score Transport efficacy hypothesis, because it never supplies a complete label packet. It falsifies the narrower claim that this exact Qwen3-8B non-thinking interface is complete at 2,048 tokens.

The new evidence separates three outcomes:

1. valid executable labels for three rows;
2. unambiguous terminal integers outside the gold-answer range for two rows;
3. right-censored reasoning for three rows.

For binary exact-match reward, an out-of-range integer is scientifically an incorrect prediction rather than a missing measurement. The version-3 range restriction therefore creates avoidable label attrition. Changing it after observing output would be a retrospective repair, so version 3 stays closed. A successor should define the reward over any syntactically bounded terminal integer and change the primary model rather than extending Qwen3's cap again.

## Decision boundary

- Close C63 as contradicted for the frozen version-3 interface.
- Do not open Stage C, fit, calibration, validation, AIME 2024 development, AIME 2025 pilot, AIME 2026 confirmation, MATH-500, or model scaling.
- Preserve the two out-of-range answers only as scorer-domain diagnostics.
- Keep Outcome Score Transport efficacy claim C60 unverified.
- Permit a new gate only with a separately frozen model and scorer-domain contract; no Qwen3 cap extension or output relabeling is permitted.

## Skill receipt

- Router state: `EVIDENCE_AVAILABLE + REVIEW_MODE + SUBMISSION_MODE`.
- `scientific-results-and-figures`: `PASS` for packet reconstruction, denominator audit, failure taxonomy, row/year stratification, v2-v3 cost comparison, and exact reproduction; `BLOCKED` for behavioral effect.
- `scientific-research-strengthening`: `EVIDENCE_THIN`; defect `SCIENTIFIC_WEAKNESS`; next admissible increment changes the model and exact reward domain rather than presentation.
- `experimental-setup-writer`: `PASS` for reproducible Stage-T reporting; downstream experiment setup remains blocked.
- Fair-strong status: blocked by complete labels, class mix, observer calibration, matched steering comparisons, mechanism controls, protected degradation, temporal confirmation, and multi-model evidence.
