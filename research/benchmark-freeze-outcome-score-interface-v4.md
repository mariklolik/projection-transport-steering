# Outcome-score interface benchmark freeze, version 4

Freeze date: 2026-09-07. Status: frozen before version-4 implementation and output.

## Decision

Version 4 tests whether a conventional instruction model plus a total exact-match reward interface can supply complete AIME outcome labels at bounded cost. It changes the primary model and scorer domain rather than extending Qwen3's token budget.

The strongest alternative is that Qwen2.5-7B-Instruct still produces incomplete or nonterminal answers within 2,048 tokens. Any missing label closes version 4 before class-mix generation.

## Model and inherited inputs

- Model: `Qwen/Qwen2.5-7B-Instruct`, snapshot `a09a35458c702b33eeacc393d103063234e8bc28`.
- Model config SHA-256: `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`.
- Tokenizer config SHA-256: `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`.
- Architecture: 28 decoder layers, hidden width 3,584, bfloat16; trace layer 14.
- Frozen historical AIME basis packet SHA-256: `8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7`.
- Temperature `0.7`, top-p `0.95`, 2,048 new-token cap, deterministic inherited seeds, concise boxed-answer prompt, and one exact zero-action replay per worker.
- Stage-T basis positions `0,4,8,12,16,20,24,28`, rollout index 0, one row per GPU.

The Qwen2.5 tokenizer and exact chat-template rendering were inspected before freeze. All earlier model outputs stay outside version 4. Fit, calibration, validation, AIME 2024, AIME 2025, AIME 2026, and MATH-500 remain unopened.

## Total exact-match reward

The scorer accepts a terminal decimal integer in either the existing final-answer line or final LaTeX box. It allows an optional minus sign and one to twelve digits, with only whitespace and an optional closing math delimiter after the integer. It rejects free-text numbers, expressions, incomplete boxes, and nonterminal answers.

Every syntactically valid integer receives a binary outcome: exact equality to the gold AIME integer is correct; every other value is incorrect. The parser does not restrict predictions to `[0,999]`. This prevents wrong out-of-range predictions from becoming missing labels while keeping parsing deterministic and judge-free.

## Stage T

Run the eight fixed preflight rows on eight H100s. Stage T passes only if:

- 8/8 rows, traces, receipts, and keys are complete and hash-valid;
- 8/8 terminal integers parse and no row reaches 2,048 tokens;
- every trace is finite and matches generated-token count;
- 8/8 zero-action replays are exact;
- every receipt binds the exact model config, layer 14, scorer, cap, sampling, and inference mode;
- maximum peak memory is below 70 GiB.

Correctness is reported but is not a Stage-T gate. Any failure closes version 4 without model, prompt, scorer, cap, layer, row, or sampling changes.

## Stage C

Only after Stage T passes, generate two new rollouts for every one of the 32 basis questions, split into eight four-question workers. Stage-T outputs are excluded.

Stage C requires 64/64 parses and pre-cap terminations, finite complete traces, exact replays, memory below 70 GiB, parallel wall below 20 minutes, and at least eight correct and eight incorrect outcomes. Failure closes version 4 before observer fit.

## Analysis and scope

Report parse form, terminality, cap censoring, exact correctness, length, year, rollout, shard, trace validity, replay, model identity, memory, forward calls, worker time, parallel wall, occupancy, and H100-hours. Do not attach a performance confidence interval to Stage T. Tokens remain nested in rollouts and rollouts in questions.

A Stage-T pass validates only the outcome interface. A Stage-C pass validates only label completeness and class support. Neither supports an observer, steering, mechanism, robustness, generalization, SOTA, or submission claim.
