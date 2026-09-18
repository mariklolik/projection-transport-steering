# Outcome-score interface research plan, version 4

Freeze date: 2026-09-07. Status: active, frozen before implementation and output.

## Decision question

Can Qwen2.5-7B-Instruct produce a complete, total binary AIME reward packet within 2,048 tokens, enabling outcome-observer research without more Qwen3 cap tuning?

Versions 1 through 3 established that Qwen3's tested interfaces remain incomplete. Version 3 also showed that an out-of-range terminal integer is a wrong answer, not an unmeasurable answer. The smallest decisive next step changes the primary model and makes the exact-match reward total over bounded terminal integers.

## Execution order

1. Freeze model snapshot, config hashes, layer, scorer grammar, prompt, cap, rows, seeds, and gates.
2. Add failing tests for signed and out-of-range terminal integers, version-4 protocol identity, layer enforcement, model hash enforcement, and version-aware analysis.
3. Extend the existing scorer, runtime dispatch, protocol registry, runner, and receipts; create no duplicate pipeline.
4. Run the full local suite, byte-reproduce version-2 and version-3 analyses, sync exact hashes, and run focused remote tests.
5. Verify eight idle H100s and launch the eight-row Stage-T packet.
6. Merge only complete receipts, reproduce analysis locally, and update claims and manifests.
7. If Stage T passes, immediately run the already frozen separate Stage-C packet on all eight GPUs.
8. If Stage C passes, begin source rollout collection and observer fitting; otherwise close version 4.

## GPU efficiency

Stage T spends at most 16,384 main-generation tokens plus replays and prevents an invalid 131,072-token Stage-C packet. Each worker loads one model once. Stage C performs eight main rollouts plus one replay per resident model. CPU analysis begins only after all GPUs are released.

The eight-worker schedule is intentionally imbalanced only by natural response length. Report both wall speedup and occupancy. No downstream data or second model is opened speculatively.

## Strength status

- Evidence: `EVIDENCE_THIN`.
- Defect: `SCIENTIFIC_WEAKNESS` in complete outcome measurement.
- Primary increment: version-4 Stage T.
- Contingent increment: Stage C.
- Outcome Score Transport efficacy: unestimated.
- Novelty and SOTA: blocked.
- Fair-strong: blocked by complete class support, observer calibration, matched steering comparisons, mechanism controls, protected degradation, temporal replication, and multi-model evidence.

The result may validate an instrument or close another route. It cannot strengthen the paper through prose.
