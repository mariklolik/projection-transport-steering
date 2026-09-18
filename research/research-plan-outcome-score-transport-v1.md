# Outcome-score transport research plan, version 1

Freeze date: 2026-09-07. Status: active successor plan after v20 and later proxy-first routes were closed.

## Scientific sequence

1. Preserve all v20 artifacts as closed evidence; do not change `product cap`, certification, or partial shards.
2. Treat LRS and PGS as nearest-work constraints and keep their unmatched published numbers separate from common-protocol baselines.
3. Freeze executable reward, sources, splits, scorer, observer, candidate, ablations, analysis, compute, and stop rules.
4. Pass CPU scorer, split, algebra, serialization, and no-op replay tests.
5. Run the 32-question, 64-rollout sentinel across eight model-resident H100 workers.
6. Fit the prefix observer only if class mix, parser, runtime, memory, and trace gates pass.
7. Open validation once; close the route if observer calibration fails OST.1.
8. Run all matched methods once on AIME 2024 development; close on primary, degradation, mechanism, or compute failure.
9. Seal one artifact for AIME 2025, then AIME 2026; open MATH-500 only after a temporal pass and a separate scorer audit.
10. Add architectures and behaviors only after the single-model route is fair-strong. Rewrite the paper only after the evidence-strength gate advances.

## Efficient GPU schedule

The unit of parallelism is the question shard, not the model or matrix operation. Each of eight workers loads Qwen3-8B once, pins one GPU, processes its sorted disjoint question list, and atomically writes one raw packet plus receipt. No worker loads test data before its stage is authorized.

The sentinel assigns four questions and eight rollouts to each GPU. Full fit assigns contiguous balanced shards after deterministic hash ordering. CPU processes validate completed packets, fit small observers, compute intervals, and prepare the next unopened manifest while GPUs finish. A merge begins only after all eight receipts exist and their source, model, split, config, row-count, and key-set hashes match.

This removes per-question model reloads, idle analysis gaps, cross-GPU communication, and speculative full sweeps. It does not fill GPUs with unaudited work. A stage that fails stops downstream allocation immediately.

## Faster iteration without adaptive leakage

The funnel exposes information in increasing scientific cost:

| Stage | New information | Maximum work | Decision |
|---|---|---:|---|
| CPU preflight | scorer and algebra correctness | seconds | repair implementation only |
| GPU sentinel | class mix, parser, trace, replay, runtime, memory | 64 generations | stop or open fit |
| Observer fit/validation | calibration and progress robustness | 1,536 fit plus 256 calibration rollouts | stop or open steering development |
| AIME 2024 development | method and ablation ordering | 30 paired questions | select once or close |
| AIME 2025 pilot | temporal transfer | 30 paired questions | stop or open confirmation |
| AIME 2026 confirmation | future temporal transfer | 30 paired questions | scope claim |
| MATH-500 breadth | larger independent benchmark | 500 paired questions | behavior-breadth evidence |

The implementation may be repaired after an engineering failure only when no scientific output exists and the estimand, constants, splits, and methods are unchanged. Scientific gate failure creates a closed version; it cannot be rescued by threshold, layer, rank, dose, parser, or subset changes.

## Deep analysis contract

Every stage reports completeness before effect. Observer analysis includes AUROC, Brier, calibration curve, constant and terminal-only comparators, progress quartiles, correctness balance, and leave-year-out sensitivity. Steering analysis includes paired transitions, recoveries and degradations, exact and bootstrap uncertainty, multiplicity, gate coverage, action reachability, intervention timing, score increments, achieved-versus-target score change, action norm, output KL mean/tail, runtime, memory, generated tokens, and derivative counts.

The mechanism audit crosses dose construction with metric construction, adds random and shuffled controls, and reports oracle-observer and oracle-action ceilings. Individual severe failures remain visible. Missingness, parser ambiguity, numerical invalidity, and worker failures are outcomes, not silently dropped rows.

## Promotion boundary

A fair-strong result requires an observer pass, AIME 2024 superiority to the strongest matched baseline, protected degradation control, a nontrivial component decomposition, and one untouched temporal replication. A registered SOTA result additionally requires the broader multi-model and multi-behavior conditions in `sota-definition.md`. Until then, the project reports a candidate mechanism or a bounded negative result, never SOTA.
