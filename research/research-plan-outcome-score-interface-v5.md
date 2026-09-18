# Outcome-score interface research plan, version 5

Freeze date: 2026-09-07. Status: active, frozen before dependency integration and output.

## Primary increment

Replace the repeatedly incomplete custom regular expressions with one pinned, strict, source-audited Math-Verify wrapper. Preserve Qwen2.5, layer 14, prompt, cap, rows, seeds, sampling, sharding, and fail-closed execution.

## Execution order

1. Freeze the package version, wheel and source hashes, extraction and verification configuration, model identity, and gates.
2. Add the exact dependency lock.
3. Add failing tests for all v4 terminal forms, adversarial intermediate numbers and boxes, empty parses, wrong outputs, evaluator exceptions, and version-5 receipt identity.
4. Implement one wrapper and route it through the existing runtime and protocol registry.
5. Run the full local suite, reproduce versions 2 through 4 byte-for-byte, sync, and run focused remote tests.
6. Launch the eight-row Stage-T packet on eight idle H100s.
7. If Stage T passes, immediately launch the frozen 64-row Stage-C packet; otherwise close version 5.
8. If Stage C passes, implement the already planned source-rollout and observer-fit stage without opening later evaluation splits.

## Efficiency

Expected Stage-T cost is bounded by the observed version-4 envelope: eight concurrent model loads, at most 16,384 main tokens plus replays, and CPU evaluation after generation. The v4 parallel wall was 38.7 seconds. Stage C uses one model load per GPU for eight main rollouts plus one replay.

No speculative downstream GPU work is allowed. Parser conformance uses preserved outputs and synthetic adversarial strings on CPU.

## Strength status

- Evidence: `EVIDENCE_THIN`.
- Defect: outcome-instrument completeness.
- Primary increment: version-5 Stage T.
- Contingent increment: Stage C.
- Method efficacy, novelty, and SOTA: unestimated and blocked.
- Fair-strong: blocked by class support, observer calibration, matched steering effects, mechanism controls, protected degradation, temporal confirmation, and multi-model evidence.
