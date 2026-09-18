# Outcome-score interface research plan, version 3

Freeze date: 2026-09-07. Status: active, frozen before implementation and output.

## Reviewer-relevant question

Can the outcome label used by Outcome Score Transport be measured completely, objectively, and cheaply on Qwen3-8B before fitting a prefix-value observer?

The current strongest verified statement is negative: thinking-mode versions 1 and 2 do not supply a valid label packet. Version 2 separates six cap-censored rows from two correct boxed answers rejected by a custom line-only parser. The next test must resolve both interface defects without another thinking-cap extension.

## Primary increment

Validate one coherent version-3 interface: Qwen non-thinking generation, a 2,048-token cap, concise boxed-answer prompt, and an exact terminal integer scorer. Reuse the existing model, frozen preflight rows, seeds, layer hook, trace serialization, no-op replay, sharding, and analyzer.

The failure outcome is explicit: one parse failure, one cap hit, one incomplete or nonfinite trace, one replay mismatch, an incomplete packet, or excessive memory closes version 3. The test does not select among prompts, caps, parsers, or decoding modes.

## Execution order

1. Freeze the interface, exact scorer grammar, eight row IDs, cap, inference mode, and gates.
2. Add failing unit tests for terminal boxed parsing, non-thinking chat-template use, version-3 protocol fields, and version-aware analysis.
3. Make the smallest extension to the existing scorer, runtime, protocol registry, runner, and analyzer; create no parallel implementation.
4. Run the full local suite and focused remote tests.
5. Verify eight idle GPUs and launch eight one-row Stage-T workers concurrently.
6. Merge only after 8/8 receipts, reproduce the analysis locally, and update the claim and evidence ledgers.
7. If Stage T passes, freeze no new choices and launch the separate 64-row Stage-C packet.
8. If Stage C passes, return to observer fitting and calibration; otherwise close the route.

## GPU efficiency

Stage T has at most 16,384 main-generation tokens plus exact replays. It spends one model load per GPU and should complete near one 2,048-token generation pair. It prevents a failed Stage-C packet of up to 131,072 main-generation tokens.

Stage C uses one resident model per GPU for eight main rollouts and one replay, up to 18,432 generated steps per worker. CPU hashing and analysis occur after GPU release. No fit, development, pilot, confirmation, MATH-500, or second-model job is queued before its gate.

## Deep analysis contract

Separate interface validity from model accuracy. Report accepted parse form, terminality, cap censoring, exact correctness, lengths, year, rollout, shard, replay, memory, wall time, GPU-hours, and effective parallel occupancy. If the parser fails, do not count its serialized false value as an incorrect model answer. If the sample is only a gate, do not attach a performance confidence interval.

## Strength assessment

- Evidence class: `EVIDENCE_THIN`.
- Defect: `SCIENTIFIC_WEAKNESS` in outcome measurement validity.
- Primary increment: version-3 Stage T.
- Contingent increment: Stage C only after a complete Stage-T pass.
- Novelty: blocked.
- Behavioral effect: unestimated.
- Fair-strong: blocked by class mix, observer calibration, matched baselines, mechanism tests, protected degradation, temporal confirmation, and multi-model evidence.

The version-3 result changes the paper only if it validates or rejects the measurement interface. It cannot strengthen the method claim through framing.
