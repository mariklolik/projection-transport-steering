# Outcome-score transport sentinel audit, version 1

Analysis date: 2026-09-07. Status: `CLOSE_BEFORE_FIT`.

## Experiment contract

The immutable basis packet contains 32 historical AIME questions selected without model output from years through 2023. Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218` generated two stochastic rollouts per question at layer 18, temperature `0.7`, top-p `0.95`, and a maximum of 1,024 new tokens. The independent unit for later behavior inference would be a question; the 64 rollouts here are nested engineering observations and cannot be treated as 64 independent behavioral units.

The frozen gate required 64 unique rows, zero parse failures, at least eight correct and eight incorrect rollouts, finite traces, exact zero-action replay, peak memory below 70 GiB, and extrapolated fit wall time below six hours. AIME 2024, AIME 2025, AIME 2026, MATH-500, fit, calibration, and validation were not opened.

## Attempt lineage

| Attempt | Result | Scientific output | Disposition |
|---|---|---:|---|
| 1 | Import-time dependency on the legacy FLAS runner | 0 rows | Invalid engineering attempt retained; no model loaded |
| 2 | Transformers 5 `BatchEncoding` was passed as a tensor | 0 rows | Invalid engineering attempt retained; regression test added |
| 3 | Eight complete shards and 64 rollouts | 64 rows | Structurally valid packet; scientific gate failed |
| Analysis 1 | Remote module lacked the newly frozen summarizer | none | Invalid import-only analysis attempt |
| Analysis 2 | Complete hash-bound merge, reproduced byte-for-byte locally | 64 rows | Accepted decision artifact |

Attempt-1 listing SHA-256 is `f6ff4921725db6299f2db483a036b78577d3974c010d53d5c099f9a047e4ba7d`; attempt-2 listing SHA-256 is `291e139b871565823eeb27fcc7a59747e0d189d099a8145ed4a70b749259f4fb`. The accepted attempt-3 packet listing SHA-256 is `32b21f9a8a5398b1217fbd20776a112e6a4fd150246346ac5f3bc88c19948d62`. The accepted analysis SHA-256 is `321b89ef85668ca02d523f2225fb226663771342ec3491a728653a32836332c6`.

## Frozen gate result

| Gate | Result | Evidence |
|---|---:|---|
| Packet completeness | PASS | 8/8 shards, 32/32 questions, 64/64 unique rollouts |
| Finite activation traces | PASS | 64/64 traces, each 1,024 states at hidden width 4,096 |
| Zero-action replay | PASS | 8/8 shard replays exactly reproduced the sampled completion |
| Peak memory | PASS | maximum 16,624,853,504 bytes, below 70 GiB |
| Runtime | PASS | maximum shard wall time 239.495 s; frozen fit extrapolation 3,406.15 s |
| Parse completeness | FAIL | 0/64 completions contain exactly one declared terminal answer |
| Outcome class mix | FAIL | no valid correctness labels; the serialized zero count is induced by parse failure |

The decision is `close_before_fit`. No observer, transport map, metric, steering action, baseline comparison, uncertainty estimate, or method effect exists.

## Post-hoc failure diagnosis

All 64 completions reached exactly 1,024 generated tokens and all 64 traces have length 1,024. None terminated within the budget. Only one completion contains the phrase `final answer` anywhere, one contains `answer is`, none contains a boxed answer, and the strict terminal form occurs zero times. The failure covers all 32 questions, both rollout indices, and all 22 represented years. The observed pattern is therefore a uniform termination-budget failure, not a question subset, shard, GPU, or answer-parser ambiguity.

This diagnosis is post-hoc and cannot rescue version 1. It supports only a new protocol hypothesis: before collecting a class-mix sentinel from a reasoning model, separately verify that the frozen generation budget and thinking mode produce complete terminal answers. It does not authorize scoring partial reasoning or extracting latent guesses.

## Cost and utilization

Each worker loaded one resident 8B model and executed 9,216 forward steps: eight primary 1,024-token rollouts plus one 1,024-token replay. Summed worker time was 1,851.97 s; parallel wall upper bound was 239.495 s. Peak allocation ranged from 16,603,783,680 to 16,624,853,504 bytes. Main-rollout throughput was 65,536 tokens over eight concurrent workers. All GPUs returned to zero allocated memory after completion.

The model-resident sharding strategy was operationally efficient, but every generated token was scientifically unusable for outcome supervision. The next version must spend a much smaller prompt-termination packet before a 64-rollout class-mix packet.

## Results-skill receipt

- Skill: `scientific-results-and-figures`; result: `PASS` for failure reconstruction, `BLOCKED` for behavioral effect estimation.
- Inputs: accepted analysis and all attempt-3 raw rows, traces, and receipts under `artifacts/development/outcome_score_v1_sentinel_*`.
- Checked contract: denominators, nesting, selection status, missingness, compute, adverse evidence, and promotion boundary.
- Figure: not generated; a plot cannot add information to uniform 64/64 truncation, and no behavioral effect is estimable.
- Invalidated downstream receipts: observer, method, result, global-paper, review, and venue-compliance gates remain unopened.
