# Outcome-score transport termination audit, version 2

Audit date: 2026-09-07. Decision: close version 2 before class-mix generation.

## Frozen question

Does Qwen3-8B with thinking enabled produce eight complete, executable AIME terminal answers within 4,096 new tokens under the version-2 protocol? The prospective gate required 8/8 strict `Final Answer: <integer>` parses, no cap hits, finite traces, exact no-op replays, complete hash-bound packets, and peak memory below 70 GiB.

The model snapshot, basis rows, rollout indices, seeds, prompt, parser, layer, sampling, and cap were frozen in `benchmark-freeze-outcome-score-transport-v2.md`. No version-2 output existed at that freeze.

## Accepted packet

- Host: `avi-gn-fsk40`.
- Hardware: eight NVIDIA H100 80GB HBM3 GPUs, one resident model and one question per GPU.
- Raw packet: `artifacts/development/outcome_score_v2_termination_attempt1`.
- Accepted analysis: `artifacts/development/outcome_score_v2_termination_analysis_attempt1/analysis.json`.
- Packet-listing SHA-256: `a87142670ad65f773a3ae3b187da3dec4a171dfcb3e4c892c3f4a833753e51bf`.
- Analysis SHA-256: `e37c346a246794978dd411c7880e594c8528aaaba0ac7d725eca685028944ddc`.
- The accepted remote analysis was reproduced byte-for-byte locally.

All eight shard receipts passed. All eight traces are finite, have one state per generated token, serialize exactly, and have unique generation keys. All eight zero-action replays are exact. No engineering retry or excluded scientific row exists.

## Result

| Shard | Year | Tokens | Frozen parse | Termination diagnosis | Descriptive final content |
|---:|---:|---:|---|---|---|
| 0 | 2015 | 3,925 | fail | EOS before cap | final `\\boxed{307}`, equal to gold |
| 1 | 2008 | 4,096 | fail | cap | unfinished |
| 2 | 1986 | 4,096 | fail | cap | unfinished |
| 3 | 1996 | 4,096 | fail | cap | unfinished |
| 4 | 1983 | 3,739 | fail | EOS before cap | final `\\boxed{15}`, equal to gold |
| 5 | 2018 | 4,096 | fail | cap | unfinished |
| 6 | 2008 | 4,096 | fail | cap | unfinished |
| 7 | 1985 | 4,096 | fail | cap | unfinished |

The primary result is 0/8 valid frozen parses and 6/8 cap hits. Both mandatory termination gates fail. The two early-EOS outputs contain unambiguous correct boxed integers, but this is a post-result format diagnostic and cannot repair the frozen parser or turn either row into a version-2 label. The serialized `correct_rows=0` is therefore not a model-accuracy estimate: correctness is undefined for all eight rows under the frozen executable scorer.

Generated lengths are `[3739, 3925, 4096, 4096, 4096, 4096, 4096, 4096]`; median is 4,096, mean is 4,030, and total main-generation tokens are 32,240. The failure spans seven represented years, with both sampled 2008 questions capped. Only rollout index 0 was authorized, so no seed or within-question heterogeneity can be estimated.

## Cost and efficiency

The workers completed in 195.166 to 208.690 seconds. Summed worker time is 1,628.176 seconds, or 0.452 H100-hours; the parallel wall upper bound is 208.690 seconds. Relative to serial execution, the observed wall-time ratio is 7.80, and worker occupancy over the eight-GPU envelope is 97.5%.

Exact replays doubled the main-generation workload, giving 64,480 instrumented forward calls. Maximum allocated memory was 17,087,380,992 bytes, or 15.92 GiB, below the 70-GiB gate. All GPUs were released after the gate; class-mix generation was not speculatively launched.

## Interpretation

Version 2 falsifies the claim that a 4,096-token thinking budget plus the strict line parser supplies complete outcome labels. It exposes two distinct interface defects:

1. termination failure: six responses remain unfinished at 4,096 tokens;
2. measurement mismatch: two complete, gold-matching boxed answers are rejected by the custom line-only parser.

Increasing the cap again would address only the first defect and would repeat the same unvalidated measurement contract at higher cost. Relaxing the parser after observing these outputs would invalidate the prospective version-2 result. A successor must therefore be a separately frozen inference-and-scoring protocol, validated first on a small termination packet, rather than a rescue of version 2.

## Decision boundary

- Close claim C62 as contradicted for version 2.
- Do not open Stage C, observer fit, calibration, validation, AIME 2024 development, AIME 2025 pilot, AIME 2026 confirmation, MATH-500, or model scaling.
- Preserve both early-EOS boxed answers only as diagnostic evidence about evaluator mismatch.
- Keep the central outcome-score transport efficacy claim C60 unverified.
- Permit a version-3 gate only if it changes the scientific interface, freezes an objective release-aligned mathematical scorer before output, and does not extend the thinking-token cap.

## Skill receipt

- Router state: `EVIDENCE_AVAILABLE + REVIEW_MODE + SUBMISSION_MODE`.
- `scientific-results-and-figures`: `PASS` for packet reconstruction, nesting, failure modes, stratification, and end-to-end cost; `BLOCKED` for behavioral effect because no valid outcome-label packet exists.
- `scientific-research-strengthening`: `EVIDENCE_THIN`; defect `SCIENTIFIC_WEAKNESS`; primary next increment is a separately frozen outcome-interface gate, not paper prose.
- Fair-strong status: blocked by outcome measurement validity, class mix, observer validity, matched steering effects, mechanism controls, protected degradation, temporal confirmation, and multi-model scope.
