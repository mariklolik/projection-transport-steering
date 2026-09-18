# Outcome-score transport benchmark freeze, version 2

Freeze date: 2026-09-07. Status: frozen before version-2 model output.

## Inherited contracts

Version 2 inherits without change the method hypothesis, Qwen3-8B snapshot, layer 18, historical AIME source, `basis/fit/calibration/validation/development/pilot/reserve` allocation, strict `Final Answer: <integer>` parser, temperature `0.7`, top-p `0.95`, deterministic rollout seeds, executable correctness, closest baselines, downstream temporal order, and claim boundaries from version 1.

Version 1 is closed and is not pooled with version 2. Its 64 capped completions and traces remain failure evidence only. AIME 2024, AIME 2025, AIME 2026, MATH-500, fit, calibration, and validation remain unopened.

## Version-2 change

The sole scientific-protocol change is the generation cap: 4,096 new tokens instead of 1,024, with Qwen3 thinking still enabled. The cap is frozen from the closest LRS AIME collection budget of approximately 4,000 tokens and the version-1 observation that every rollout was right-censored at 1,024. The prompt and scorer do not change.

## Stage T: termination gate

Select basis rows at zero-based positions `0,4,8,12,16,20,24,28` in the already frozen `basis.json` order. Generate rollout index 0 for each, one question per H100 and one resident model per worker. Each worker also performs one exact zero-action replay of its row.

Stage T passes only if:

- all 8 rows, traces, receipts, and unique keys are complete and hash-valid;
- all 8 completions parse under the unchanged terminal form;
- no completion is empty or reaches 4,096 tokens;
- all traces are finite and match generated-token counts;
- 8/8 zero-action replays are exact;
- maximum peak memory is below 70 GiB.

Correctness is reported but is not a Stage-T gate. A failure closes version 2 before the class-mix packet. No parser repair, prompt change, thinking-mode change, partial-answer extraction, or cap extension is allowed.

## Stage C: class-mix gate

Only after Stage T passes, generate a new complete packet of two rollouts for all 32 basis questions under the identical 4,096-token configuration. Stage-T rows are not reused or pooled. Use eight disjoint four-question shards.

Stage C inherits the version-1 structure, parse, trace, replay, and memory gates and additionally requires at least eight correct and eight incorrect rollouts. Maximum parallel wall time is 30 minutes. A failure closes version 2 before fit. A pass opens only the fit-rollout stage; it does not establish observer validity, steering efficacy, novelty, or SOTA.

## Analysis

The frozen analysis reports planned and observed rows, unique questions, generated-token distribution, termination mode, parse form, exact correctness, class mix, trace lengths, replay, per-shard wall time, peak memory, and total forward steps. Outcomes are stratified by question year, rollout index, shard, termination status, and completion length. Tokens remain nested within rollouts and rollouts within questions; no token-level uncertainty is permitted.
