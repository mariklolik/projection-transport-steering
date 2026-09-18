# Outcome-score observer research plan, version 1

Freeze date: 2026-09-07. Status: active before basis-completion or fit output.

## Strength classification

- Scientific maturity: `EVIDENCE_AVAILABLE + REVIEW_MODE + SUBMISSION_MODE`.
- Evidence class: `EVIDENCE_THIN` for outcome-steering efficacy; `fit-eligible` for the instrument.
- Defect: `SCIENTIFIC_WEAKNESS` in prefix-observer validity and causal action readiness.
- Strongest verified claim: C65, complete strict outcome labels and both classes on the exposed basis packet.
- Central unverified claim: C66, held-out causal-prefix discrimination and calibration under OST.1.

The strongest alternative is that final correctness is separable only at late or terminal states, or only within exposed questions, while online early-prefix predictions remain uninformative. A fit or validation failure resolves against the route and stops steering work.

## Ordered execution

1. Freeze source collection, basis weighting, observer candidates, grouped selection, controls, calibration boundary, validation statistics, and stop rules.
2. Extend the existing v5 protocol and runner only for `basis_completion` and `fit`; add fail-closed source aggregation.
3. Observe RED tests, implement the minimum source-stage surface, run the full suite, and reproduce v2-v5 accepted analyses.
4. Launch basis completion across eight idle H100s. Aggregate and freeze 128 basis rollouts.
5. Construct and freeze the maximum-rank basis before fit observer training.
6. Launch 1,024 fit rollouts across eight H100s. While GPUs run, implement and test only the predeclared basis and observer fitter.
7. Aggregate fit source completeness and class/progress support before model selection.
8. Run all six candidates under the frozen three-fold grouped protocol, publish every candidate result, and refit only the selected eligible observer.
9. If selection passes, freeze the calibration implementation and only then collect calibration outputs. Otherwise close the route.
10. If calibration passes, freeze the complete observer artifact and open validation once. OST.1 alone decides whether steering development opens.

## Efficient execution

Question shards keep one Qwen2.5 instance resident per GPU. Basis completion costs eight main rollouts per worker. Fit costs 128 main rollouts per worker and is expected to remain below the 29.5-minute Stage-C estimate scaled from observed throughput, excluding conservative six-hour gate slack. Source traces stream directly to one shard artifact; no cross-GPU synchronization occurs.

Basis algebra and packet aggregation run on CPU while GPUs are idle or collecting the next authorized source stage. Observer candidates use the same precomputed basis coordinates. Independent folds or candidates may occupy separate GPUs after source collection; no model replication is needed for the small observer. Calibration and validation are not generated speculatively.

## Analysis depth

Completeness precedes performance. Source analysis reports exact question and rollout coverage, label balance, progress-bin support, length by class and year, paired within-question outcome patterns, parser/cap failures, trace alignment, replay, wall-time distribution, token throughput, H100-hours, memory, and all individual severe cases.

Observer selection reports group-fold Brier and AUROC for every candidate, constant and progress controls, terminal-only diagnostics, progress and year strata, question-cluster uncertainty, optimization convergence, parameter count, training time, inference cost, and failure cases. A high terminal diagnostic cannot rescue an uninformative deployable prefix observer.

## Stop and promotion boundary

No source-support failure is repaired by changing the model, cap, parser, bins, layer, rank menu, sampling, or shards. No fit-selection failure is repaired by adding candidates or selecting on calibration. No validation failure is repaired on AIME 2024.

A fit pass opens only calibration. A calibration pass opens only one-shot observer validation. An OST.1 pass opens only matched steering development. Fair-strong, temporal, multi-model, multi-behavior, SOTA, paper-writing, and submission gates remain blocked.
