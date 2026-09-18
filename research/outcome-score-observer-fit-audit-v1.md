# Outcome-observer source audit after the research reset

Date: 2026-09-07. Decision: close the frozen observer-v1 route before fitting. This is a source-eligibility failure, not a measured failure of an observer or steering policy.

## Provenance and exposure

The user requested a scientific reset while fit-source collection was already running. That collection finished unchanged; no observer was trained and no calibration, validation, development, pilot, or confirmation outcomes were opened. The incomplete observer-basis test is preserved in `tmp/paused-observer-v1/`; it is not an implemented method.

The earlier replay repair changes only selection of the first declared rollout index. Basis completion uses indices 2 and 3, not 0. Its failed attempt 1 remains recorded in `outcome-score-observer-basis-attempt1-failure.md`; attempt 2 is separate.

| Object | Verified locator or SHA-256 |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct, snapshot `a09a35458c702b33eeacc393d103063234e8bc28` |
| Frozen runner | `ed8816b90ef47b71d3e04182a953add9d0865648de8e368a0e1114401431c443` |
| Basis-completion analysis | `artifacts/development/outcome_score_observer_v1_basis_completion_analysis_attempt1/analysis.json`; `d700969ea5c6826e070a567f2b4233100534d64d878481d508a27e06cab99f3e` |
| Basis-completion packet listing | `6ecec960b3629d05957f08109948d26f2760d8d4a05cff99a9b95e40c3c1b6de` |
| Fit analysis | `artifacts/development/research_reset_fit_archive_analysis_v1/analysis.json`; `e328d5e821841f8286eff9692b4af709528e91bde0610b75c77b8d67595062ed` |
| Fit packet listing | `6d80335cff1dd4ab61bc6017077205f96be7a6ecf4c40485894fa7917e00c204` |

Basis-completion analysis directory name refers to the first analysis invocation, not the failed generation attempt. Its inputs are attempt-2 shards; an underscore-named symlink view adapts the existing analyzer to the actual hyphen-named shard directories.

Fit raw outputs and traces remain on `avi-gn-fsk40` under `/home/mekashirskiy/projection-transport-steering/artifacts/development/outcome_score_observer_v1_fit_attempt1/shard_00` through `shard_07`. The unchanged analyzer verified row/receipt/trace hashes and finite traces remotely. Small row files, receipts, logs and analysis were copied locally. The approximately 6.06 GB of dense traces were not downloaded. Local verification does not independently re-read those remote tensors.

## Results under the unchanged contract

Basis completion passes: 64 rows, 32 questions, 15 correct, 49 incorrect, exact replay, complete parses and pre-cap termination. Correct outcomes occur in 9 questions; incorrect outcomes in 26. Maximum worker time is 123.010 seconds. This authorizes only the old basis stage, not a performance claim.

Fit collection contains exactly 1,024 rows from 256 questions and four rollouts per question. All eight receipts pass artifact/replay checks. The source summary fails `parse_complete`, `terminated_before_cap`, and `progress_support`; its decision is `close_before_observer_fit`.

| At-risk token position | Remaining rows | Questions | Eventually correct | Eventually incorrect |
|---|---:|---:|---:|---:|
| 0 | 1,024 | 256 | 201 | 823 |
| 512 | 840 | 240 | 153 | 687 |
| 1,024 | 249 | 122 | 28 | 221 |
| 1,536 | 46 | 35 | 0 | 46 |

There are 999 accepted parses and 25 parse failures. All 23 cap-hit rows are among the parse failures; two additional parse failures terminate earlier. The descriptive exact-correctness rate over all 1,024 rows is 19.629%. An exploratory question-cluster percentile bootstrap gives [15.625%, 23.730%] (10,000 resamples; seed 20260907). This interval describes the exposed fit questions and is not a confirmatory estimate of a new method.

The counts of questions with 0, 1, 2, 3, or 4 correct rollouts are respectively 170, 31, 18, 14, and 23. Thus 63 questions have mixed outcomes, 23 always succeed in these samples, and 170 never succeed in these samples. The presence of at least one success in 86/256 questions is an observed multi-sample availability statistic, not a deployable selector result or a causal intervention effect.

Correct completions have median length 645 tokens and maximum 1,464. Incorrect completions have median 806 and maximum 2,048. A late-token observer therefore sees a selected population that excludes all successfully terminated trajectories in this packet. Final completion length would be a future-information diagnostic, not an admissible online feature.

## Cost

Eight H100 80 GB workers use a summed 12,900.004 seconds (3.5833 H100-hours), with a longest worker of 1,765.129 seconds (29.419 minutes). Main generations contain 845,606 tokens; instrumented forwards including replay total 852,639. Maximum allocated memory is 15,438,928,384 bytes (14.38 GiB).

Dividing main tokens by summed worker time gives 65.55 tokens per worker-second. Dividing summed time by eight times the longest worker gives 91.35% shard balance, not GPU compute utilization. These costs include the worker's loading and bookkeeping. Low allocated memory alone does not prove idle tensor cores; the next runtime benchmark must measure utilization, batching, and end-to-end throughput separately.

## Interpretation and disposition

The previous all-bins/both-classes eligibility rule makes this particular observer design ineligible. It does not establish that late states are useless, that early correctness is unpredictable, or that interventions cannot help. No such predictor or intervention was evaluated.

Future protocols must distinguish missing scientific artifacts from valid failed model answers. An unparseable or budget-exhausted answer can be assigned zero task utility while remaining in the evaluation denominator. Terminated trajectories form an absorbing state and require no later intervention. This is a prospective design change for a different experiment, not a retrospective repair of observer-v1.

Keep C66's scientific effect unverified and close its frozen execution route. Preserve C65's narrower Stage-T/Stage-C result. Do not fit on this packet as a successor, tune the old progress bins, extend the old token cap, or open its later splits. This packet is now exploratory historical evidence only.
