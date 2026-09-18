# Outcome-score transport research plan, version 2

Freeze date: 2026-09-07. Status: active termination-first successor to closed version 1.

## Decision and alternative

The immediate question is whether Qwen3-8B can produce executable AIME terminal answers under a fixed reasoning budget. The strongest alternative is that extended thinking remains right-censored even at 4,096 tokens, making the current model/protocol unsuitable for outcome-value supervision. Eight-of-eight strict parses advance; any failure closes this version.

This test does not ask whether outcome-score transport works. It prevents spending the much larger fit and steering budget when the label-generating process is invalid.

## Execution order

1. Freeze version-2 cap, row IDs, seeds, parser, output contract, and gates.
2. Extend the existing versioned runner through an observed red-green test; do not fork the runtime or scorer.
3. Run eight one-row model-resident Stage-T workers concurrently.
4. Merge only after 8/8 receipts and classify termination without changing the protocol.
5. If Stage T passes, run the 64-row Stage-C class-mix packet across eight GPUs.
6. If Stage C passes, resume the version-1 observer, transport, baseline, and temporal plan using only version-2 source rollouts.
7. If either stage fails, close version 2 and reconsider the model/protocol as a new scientific route rather than extending the cap again.

## GPU efficiency

Stage T spends at most 32,768 generated tokens in parallel and should finish near the time of one long rollout. It prevents a failed 262,144-token class-mix packet. Stage C uses one model load per GPU and eight rollouts per worker. CPU validation and packet hashing overlap no GPU work and do not delay the next authorized stage.

No development, pilot, confirmation, multi-model, or MATH-500 process is launched speculatively. All eight H100s are released immediately after each gate.

## Strength assessment

- Evidence class: `EVIDENCE_THIN`; the first outcome-label gate failed.
- Defect: scientific protocol validity, not presentation.
- Primary increment: version-2 Stage T.
- Contingent increment: Stage C only after a complete Stage-T pass.
- Novelty boundary: unchanged and blocked.
- Fair-strong status: blocked until observer validity, matched development superiority, mechanism decomposition, protected degradation, and untouched temporal replication pass.

The expected information gain is binary and high: a pass validates the label-generation interface; a failure prevents downstream GPU waste. Neither outcome improves the paper through wording.
