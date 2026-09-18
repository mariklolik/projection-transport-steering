# Semantic-outcome sentinel execution, version 1

Freeze date: 2026-09-07. Status: frozen after the one-concept smoke and before any multi-concept sentinel output.

## Boundary

The admitted smoke is `artifacts/development/semantic_outcome_v1_smoke_c1_attempt3`. It passed zero-action replay, hard feasibility, KKT, finite output, generation, serialization, and resource checks. Attempts 1 and 2 remain invalid engineering receipts. None of the three attempts selects a method, factor, witness, metric, layer, or concept.

The 12-concept sentinel keeps every scientific constant in `benchmark-freeze-semantic-outcome-v1.md`. This execution freeze only short-circuits expensive open generation and judging when the teacher-forced premise already fails.

## Shared metric and shards

The global neutral metric is reused byte-for-byte from the admitted smoke `actions.pt`, SHA-256 `fb6236abebc91010e57355e7f9ea93079863d8f21108f3eba374f15eca448c20`. Its source is the same frozen 16-row neutral pool, model, tokenizer, layer, ridge, and score definition used by every sentinel concept. Reuse removes 80 redundant backward products relative to six independent two-concept workers.

Immediately before launch, the scheduler rechecks ownership, memory, utilization, and free memory. Six idle H100s receive two concepts each in published sentinel order: `(1,5)`, `(16,24)`, `(29,33)`, `(34,48)`, `(58,59)`, and `(60,73)`. Each process loads one model once, uses at most two CPU threads, generates matched negatives concept-by-concept, and serializes concept failures without reallocating or deleting a witness.

## Stage A: construction and mechanism

For every concept, Stage A constructs `pooled_euclidean`, `pooled_sequence_metric`, `robust_euclidean`, `semantic_outcome`, `last_token_semantic_outcome`, and a DiffMean operator. The DiffMean operator is the released mean-positive-minus-mean-negative construction applied to the same frozen matched continuation tokens at the same layer. It is a component control; a common-protocol released AxBench DiffMean receipt remains required before a development performance claim.

All noncandidate action directions are rescaled to the candidate's declared diagonal-metric cost before applying candidate factors `{0.25,0.5,1.0,1.5}`. Raw directions, scale multipliers, Euclidean norms, metric costs, construction margins, duals, ranks, and KKT residuals are retained.

Stage A batches all held-out positive and negative continuations for a concept into one forward per method-factor cell. It separately batches all 16 neutral continuations and reports per-row likelihood change and forward tokenwise KL against no-op. The independent unit remains the concept; continuation rows are nested diagnostics.

All four factors and all concepts are reported. No factor is selected from Stage A. Open generation is authorized only if at least 11/12 concepts have complete hard actions and at least one frozen factor satisfies the SO.3 premise against both primary ablations: candidate worst-view improvement wins in at least 9/12 concepts, each candidate-minus-ablation mean paired change is positive, and median neutral likelihood loss is within `0.02` nats per token of the strongest action ablation.

## Stage B: tuning generation

Only after Stage A passes, generate all five exposed tuning prompts for every required method-factor cell, no-op, and exact AxBench prompt steering. Decoding policy, seeds, batch composition, raw generations, parse status, severe-collapse rules, and judge receipt are frozen before the first Stage B generation. SO.4 and the complete sentinel decision are evaluated only from that immutable packet.

Failure at either stage closes SemanticOutcomeBack version 1. It cannot be rescued by changing witness count, source allocation, metric ridge, layer, factor grid, margin, endpoint, or comparator identity.

## Analysis contract

The Stage A audit begins with planned/completed concepts and paired-cell missingness. It shows all concept-level worst-view and mean-view changes, candidate-minus-ablation pairs, neutral likelihood and KL, active-set size and dual concentration, witness cosine/disagreement, positive and negative token lengths, action cost, wall time, peak memory, forward calls, backward products, and generated tokens. Any interval is labeled exploratory and resamples concepts only. No teacher-forced result is called behavioral performance or SOTA.
