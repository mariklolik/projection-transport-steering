# Outcome-score interface benchmark freeze, version 5

Freeze date: 2026-09-07. Status: frozen before dependency integration and version-5 output.

## Decision

Version 5 tests whether Qwen2.5-7B-Instruct plus the pinned strict Math-Verify configuration supplies a complete outcome-label interface. It replaces the failed custom grammar with a standard audited evaluator while preserving the successful version-4 model and generation protocol.

The strongest alternative is that new generations expose evaluator failures, incomplete answers, or cap censoring despite the v4 conformance result. Any missing label closes version 5.

## Frozen protocol

- Model snapshot: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`.
- Model config SHA-256: `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`.
- bfloat16, layer 14 of 28, temperature `0.7`, top-p `0.95`, 2,048 new tokens, standard chat template, and inherited deterministic seeds.
- Prompt: concise reasoning ending in one boxed integer.
- Evaluator: `math-verify==0.9.0`, wheel SHA-256 `3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194`.
- Prediction extraction: only `LatexExtractionConfig(try_extract_without_anchor=False)`, no fallback, first match, five-second timeout, exceptions enabled.
- Verification: strict, five-second timeout, exceptions enabled, against the parsed canonical dataset integer.
- Stage-T basis positions: `0,4,8,12,16,20,24,28`; rollout index 0; one row and exact replay per H100.

Version-4 outputs are excluded from the version-5 gate. Fit, calibration, validation, AIME 2024, AIME 2025, AIME 2026, and MATH-500 remain unopened.

## Stage T

Stage T passes only if all eight new rows parse and terminate before 2,048 tokens; all rows, traces, receipts, hashes, and keys are complete; traces are finite and length-aligned; replays are exact; receipts bind the model, layer, evaluator version, and extraction configuration; and peak memory stays below 70 GiB.

Correctness is descriptive. Any parse exception, timeout, empty parse, cap hit, model mismatch, replay mismatch, trace failure, or missing artifact closes version 5 without evaluator, prompt, cap, model, row, seed, or sampling changes.

## Stage C

Only after Stage T passes, generate a separate two-rollout packet for all 32 basis rows on eight four-question workers. Stage-T generations are excluded.

Stage C requires 64/64 complete parses and pre-cap terminations, finite aligned traces, exact replays, peak memory below 70 GiB, parallel wall below 20 minutes, and at least eight correct and eight incorrect outcomes. Failure closes version 5 before fit.

## Reporting boundary

Report evaluator errors separately from incorrect predictions. Report complete denominators, extracted final candidate, exact correctness, length, year, rollout, shard, trace, replay, model and evaluator identity, memory, forward calls, wall time, occupancy, and H100-hours. Stage T receives no performance confidence interval.

A Stage-T pass establishes only interface validity. A Stage-C pass establishes only complete class support. Observer, steering, mechanism, robustness, temporal, multi-model, SOTA, and paper claims remain blocked.
