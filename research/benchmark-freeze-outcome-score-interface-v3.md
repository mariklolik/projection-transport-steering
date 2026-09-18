# Outcome-score interface benchmark freeze, version 3

Freeze date: 2026-09-07. Status: frozen before version-3 implementation and model output.

## Decision

Version 3 tests whether a coherent non-thinking inference and objective AIME-scoring interface can produce complete outcome labels before any class-mix, observer, or steering expenditure. It is a new protocol, not a repair or continuation of version 2.

The strongest alternative is that Qwen3-8B still fails to terminate reliably or cannot produce eight objectively parseable AIME answers under the shorter non-thinking budget. Any such failure closes version 3.

## Inherited immutable inputs

- Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218` in bfloat16.
- Residual layer 18 of 36.
- Frozen historical AIME basis packet SHA-256 `8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7`.
- Temperature `0.7`, top-p `0.95`, and deterministic seeds derived from `SHA-256("ost-v1|question_id|rollout_index")`.
- Stage-T basis positions `0,4,8,12,16,20,24,28`, rollout index 0, one row and one exact zero-action replay per GPU.
- The split order, downstream matched baselines, uncertainty, protected degradation, temporal confirmation, and claim boundaries from version 1.

Version-1 and version-2 outputs are not pooled with version 3. Fit, calibration, validation, AIME 2024, AIME 2025, AIME 2026, and MATH-500 remain unopened.

## Version-3 interface

Thinking is disabled through the Qwen chat-template switch. The maximum is 2,048 new tokens. The prompt requests concise reasoning and a final boxed integer.

The objective scorer accepts exactly either:

- one terminal line `Final Answer: <integer>` under the existing strict rule; or
- a final terminal LaTeX `\\boxed{<integer>}`, allowing only trailing whitespace and a closing inline or display-math delimiter.

The parsed integer must lie in `[0,999]`. Free-text numbers, incomplete boxes, nonterminal boxes, signs, expressions, and values outside the AIME range fail. Correctness is exact integer equality to the dataset answer. No LLM judge, symbolic equivalence, gold-text search, or post-result exception is allowed.

This narrow scorer is benchmark-aligned and dependency-free. The closest LRS release evaluates full math outputs through `math_verify` and includes boxed answer exemplars in its steering source. The pinned anchors are `utils/rule_eval.py` SHA-256 `889bf5ff08afbb20dd98ae449e2cf4e5f4508affd896d5852d2d9fd252724ef8` and steering source SHA-256 `1b73c95d73c3af038f700fa45429028d0acff96aa4c7a537f0c6883b7370344a` at commit `5becfd7b2fa9ae80bab713d4a93dbcb36d146fe6`.

## Stage T: outcome-interface gate

Run the same eight preflight rows used by version 2, with new version-3 outputs and no reuse of prior generations. Use eight H100s concurrently.

Stage T passes only if:

- all 8 rows, traces, receipts, and generation keys are complete and hash-valid;
- all 8 completions parse under the frozen version-3 scorer;
- no completion is empty or reaches 2,048 tokens;
- all trace lengths equal generated-token counts and all states are finite;
- all 8 zero-action replays are exact;
- maximum peak memory is below 70 GiB.

Correctness and output form are reported but correctness is not a Stage-T pass condition. A failure closes version 3 without prompt, scorer, cap, sampling, row, or thinking-mode changes.

## Stage C: class-mix gate

Only after Stage T passes, generate a new packet of two rollouts for all 32 basis questions under the identical version-3 interface. Stage-T outputs are excluded. Use eight disjoint four-question shards.

Stage C requires all 64 rows to parse and terminate before the cap, finite complete traces, exact shard replays, peak memory below 70 GiB, parallel wall time below 20 minutes, and at least eight correct and eight incorrect outcomes. Any failure closes version 3 before observer fit.

## Analysis and stopping

Report complete denominators, generation length, termination form, correctness, year, rollout, shard, trace, replay, memory, forward calls, worker time, parallel wall time, and GPU-hours. Tokens remain nested in rollouts and rollouts in questions; no token-level uncertainty is permitted.

No downstream process runs speculatively. A Stage-T pass opens only Stage C. A Stage-C pass opens only source rollout collection for the already frozen fit split. Neither gate supports steering efficacy, mechanism, novelty, robustness, generalization, SOTA, or paper-readiness claims.
