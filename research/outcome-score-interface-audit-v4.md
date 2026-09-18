# Outcome-score interface audit, version 4

Audit date: 2026-09-07. Decision: close version 4 before class-mix generation.

## Frozen question

Does Qwen2.5-7B-Instruct with a total terminal-integer scorer produce eight complete AIME outcome labels within 2,048 tokens? The prospective gate required 8/8 parses and pre-cap terminations, complete finite layer-14 traces, exact replays, exact model identity, complete hash-bound packets, and peak memory below 70 GiB.

The model, config hashes, layer, prompt, scorer grammar, rows, seeds, sampling, and stopping rules were frozen in `benchmark-freeze-outcome-score-interface-v4.md` before implementation or output.

## Accepted packet

- Host: `avi-gn-fsk40`.
- Hardware: eight NVIDIA H100 80GB HBM3 GPUs, one resident model and row per GPU.
- Raw packet: `artifacts/development/outcome_score_v4_termination_attempt1`.
- Accepted analysis: `artifacts/development/outcome_score_v4_termination_analysis_attempt1/analysis.json`.
- Packet-listing SHA-256: `c451159ad6ec213484e20304170b53c109267d7cef8abe93b70cb616d7ff4a43`.
- Analysis SHA-256: `4859c268623d1d700c2a86db56beb560355882aff8c720ff99a814504bc47dea`.
- The remote analysis was reproduced byte-for-byte locally.

All eight receipts bind the frozen Qwen2.5 config hash and layer 14. All traces are finite and align with generated-token counts. All zero-action replays are exact. No engineering retry or excluded scientific row exists.

## Result

| Shard | Year | Tokens | Terminal output | Frozen parse | Diagnostic exact outcome |
|---:|---:|---:|---|---|---|
| 0 | 2015 | 956 | `\\boxed{307} \\]` | pass | correct |
| 1 | 2008 | 698 | `\\boxed{1504}\\).` | fail | incorrect; gold 504 |
| 2 | 1986 | 625 | `\\boxed{1}\\)` | pass | incorrect; gold 560 |
| 3 | 1996 | 574 | `\\boxed{799}\\]` | pass | correct |
| 4 | 1983 | 521 | `\\boxed{15}\\).` | fail | correct |
| 5 | 2018 | 582 | `\\boxed{50}. \\]` | fail | incorrect; gold 600 |
| 6 | 2008 | 981 | `\\boxed{44}` | pass | incorrect; gold 251 |
| 7 | 1985 | 559 | `\\boxed{88}\\).` | fail | incorrect; gold 32 |

The primary result is 4/8 accepted parses, so Stage T fails. No row reaches the cap. All eight outputs end with an unambiguous boxed integer; the four failures differ only by conventional terminal punctuation that the frozen regular expression excluded. Under exact integer comparison, the diagnostic outcomes would be three correct and five incorrect, but these four rows are not relabeled in version 4.

Generated lengths are `[521, 559, 574, 582, 625, 698, 956, 981]`; median is 603.5, mean is 687, and total main-generation tokens are 5,496. Termination succeeds across all seven represented years. Only rollout index 0 was authorized, so this gate supplies no performance estimate or within-question uncertainty.

## Cost and GPU efficiency

Workers complete in 24.622 to 38.723 seconds. Summed worker time is 235.405 seconds, or 0.065 H100-hours; parallel wall is 38.723 seconds. The serial-to-parallel wall ratio is 6.08 and worker occupancy is 76.0%. Exact replays account for half of 10,992 forward calls. Maximum allocated memory is 15,373,015,040 bytes, or 14.32 GiB.

Relative to Qwen3 version 3, Qwen2.5 reduces parallel wall by 64.8%, worker time by 65.6%, and forward calls by 56.6%. This is a verified efficiency improvement for the interface gate, not a reasoning-quality or steering result.

## Interpretation

The model change resolves the termination defect: every answer finishes well before the same 2,048-token cap. The remaining failure is scorer completeness. A sequence of hand-expanded regular expressions has now rejected semantically unambiguous standard math forms in two consecutive protocol versions. Adding the observed punctuation after output would be another retrospective repair and would not establish coverage of unseen valid forms.

The closest LRS release already delegates math evaluation to `math_verify` over the full model output. The smallest scientifically defensible successor is therefore a separately frozen standard-evaluator gate, pinned by package version and source hash, rather than another custom grammar extension.

## Decision boundary

- Close C64 for the exact version-4 regex interface.
- Do not open Stage C, fit, calibration, validation, AIME 2024 development, AIME 2025 pilot, AIME 2026 confirmation, MATH-500, or model scaling.
- Keep the eight terminal integers as post-result scorer diagnostics only.
- Preserve Qwen2.5, layer 14, cap 2,048, prompt, seeds, and rows as the strongest termination configuration; do not infer accuracy from eight preflight rows.
- Permit a new gate only with a pinned standard mathematical evaluator frozen before rescoring or new generation.

## Skill receipt

- Router state: `EVIDENCE_AVAILABLE + REVIEW_MODE + SUBMISSION_MODE`.
- `scientific-results-and-figures`: `PASS` for packet reconstruction, row/year failure taxonomy, v3-v4 cost comparison, and exact reproduction; `BLOCKED` for method effect.
- `scientific-research-strengthening`: `EVIDENCE_THIN`; defect `SCIENTIFIC_WEAKNESS` in measurement completeness; next increment is a pinned evaluator audit and gate.
- `experimental-setup-writer`: `PASS` for the version-4 instrument report; downstream experiment setup remains blocked.
- Fair-strong status: blocked by standard-evaluator validation, class mix, observer calibration, matched steering comparisons, mechanism controls, protected degradation, temporal confirmation, and multi-model evidence.
