# Baseline-v2 complete-result analysis contract

Analysis implementation starts while all eight frozen workers are running. Runtime progress and some zero raw-file metadata have been inspected; no candidate selection outcome has been inspected. This is an exploratory selection analysis, not a new confirmatory preregistration. The primary endpoint, candidate product, winner tie-break and response-entry rule remain those frozen before dispatch.

## Discovery before code

The producer already persists raw reference chunks before scoring, scored chunks and their receipt hashes, per-update fit JSONL, full raw adapter state, and a terminal receipt. `irc_baseline_support.checked_packet` and `prepare_inputs` validate allocations and native targets. `reft_training.epoch_batches` reconstructs exact optimizer membership. `outcome_score_runtime.rollout_seed`, `score_math_outputs`, and the pinned author extractor define identity and grading. `statistics.paired_bca_interval` already implements question-paired BCa intervals. Existing analyzers use incompatible historical schemas; those closed routes remain unchanged.

Only missing analysis wiring is needed: verify the existing baseline output contract, then aggregate all eight candidates against the common zero. Add analysis-only files and tests; do not modify the 52-source running snapshot, frozen config, grader, generator, training recipe, or allocations. The caller supplies complete source-bound records, never unequal available-case marginal means. No new bootstrap, labeler, trainer or sampler is needed.

## Integrity gate

Require all eight final receipts and corresponding terminal container states; a running container's ExitCode=0 is not completion. Record missing, failed and extra output files without interpreting missing candidates as defeated. Match config/source/candidate/checkpoint hashes and frozen-base status. Every candidate must contain exactly 564 updates covering all 1,500 questions in each of 12 epochs, the frozen within-batch order, final partial batch, target-token counts, learning-rate schedule and raw checkpoint. The JSONL gradient norm is pre-clipping: it may exceed one and does not certify post-clipping norm.

For zero, require the exact 32-question share from each worker and their disjoint 256-question union. For each candidate require all 256 selection identities in their original order. Match receipt chunk hashes, raw/scored tokens and text, prompt hashes, immutable support membership, row IDs and per-question seed. EOS must be the first and final EOS token; nonterminated completions retain the full 8,192-token cap and score zero. Evaluator errors remain technical failures, not exclusions or imputed wrong answers. Hash every consumed artifact. Never load sealed-test solutions.

The complete-packet audit invokes both pinned graders once more on CPU: one original scoring pass plus one audit replay, not literally one lifetime invocation. Stored scores remain authoritative; disagreement or replay timeout blocks the audit without replacement, exclusion or generation retry. Preserve every audit attempt, including CPU duration, input/code hashes and candidate/condition/chunk/cluster/error locators, without answer text in failure messages. A replay timeout is not a method-efficacy failure. Generation-token bounds use the verified model vocabulary (151,936 for this Qwen), not the shorter tokenizer vocabulary (151,669); undefined tokenizer entries retain the original decoder behavior.

## Estimands and selection

The primary endpoint is author-normalized MATH correctness with a nonempty extracted box and EOS. The measurement-and-allocation amendment supersedes the older benchmark paragraph naming Math-Verify as primary. Report all eight candidates and zero, not only winners. Select full and prompt winners separately by primary count, then lower rank, then lower site. Only a strict full-over-zero point gain satisfies the frozen baseline entry criterion; this is not statistical confirmation or automatic response-launch authorization.

The independent resampling unit is the question, paired across the nine conditions. One rollout seed per question and one fit seed do not measure between-training-seed variability. Use the existing BCa implementation with 10,000 resamples, seed 20260907 and two-sided 95% intervals. All candidate and stratum intervals are descriptive exploratory intervals, without simultaneous coverage or winner-selection correction. Report zero variance/degenerate intervals explicitly; fewer than two questions has no interval. Do not infer preservation from nonsignificance.

On the prospectively fixed supported subset, report both the primary author score and Math-Verify score, their paired candidate-minus-zero effects, coverage and scorer-disagreement counts. Parse failures stay in the same denominator. This subset cannot replace the full-panel primary endpoint.

## Error and cost analysis

For each candidate show the full base-correct/candidate-correct 2x2 table, including rescue and harm counts. Report EOS, cap failure, malformed extraction, generated-token distribution and paired token-count change. Stratify by source subject and difficulty, unmodified prompt-token bins (<128, 128–255, >=256), zero correctness, zero EOS/cap status, and zero output length (<=512, 513–2048, >2048). Candidate output-length strata are post-treatment descriptions only. The baseline has a fixed prompt/full intervention policy, not a learned firing rate; no action-frequency selection is possible in this stage.

Report all eight container lifetimes, including compilation, loading, fitting, grading, failed work and shutdown, summed as H100-hours. Separately report measured generation-batch seconds, grading seconds and fit seconds. Do not sum nested component timers into container cost or infer per-question latency/tail latency from batch timings. A point-in-time memory/utilization snapshot is not peak memory or mean utilization. Compare measured cost to the prospective 48-H100-hour ceiling; historical engineering spend remains in its separate ledger and in total research cost, never reclassified as free.

The sum of disjoint fit, zero-generation/grading/check and candidate-generation/grading/check timers must not exceed the enclosing worker timer by more than 0.001 seconds. Update timers are nested inside fit and are not added again. The complete analyzer does not audit an administratively aborted partial panel: preserve its individual files and cancellations in a separate abort inventory, with no selected winner or positive entry admission.

The smallest useful display is the complete nine-condition table plus paired rescue/harm and subject/difficulty tables. No figure or submission prose is generated before complete results and their integrity audit. Broader SOTA, response-controller efficacy, protected utility, cross-model scope, fair-strong and submission gates remain blocked.
