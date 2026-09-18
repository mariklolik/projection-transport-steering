# Baseline-v3: complete fit-component audit

## Decision and evidence boundary

All eight original candidates completed their full training component. This resolves complete fit coverage and raw-checkpoint integrity, not held-out improvement, terminal worker success, final base immutability, baseline-entry admission or SOTA. The original eight-way experiment continues unchanged; no candidate is selected from training loss.

At 2026-09-08 06:38:01 UTC, every worker had 564 closed fit records and a saved raw checkpoint. The selection directory is created only after the producer checks the static callable, restores the inference callable, loads the raw checkpoint strictly and checks its equality to the trained action. All eight workers were still running, with selection progress [8,16,0,8,0,48,0,48]/256. These are completion counts, not correctness scores. No terminal producer receipt existed in that capture.

The [predeclared component protocol](irc-baseline-fit-audit-v3.json), [exact CPU execution record](irc-baseline-fit-cpu-execution-v3.json), [native result](../artifacts/selection/irc_baseline_v3/fit-audit-v1/result.json) and [independent review](irc-baseline-fit-independent-review-v3.json) retain the relevant scope and provenance. Native result SHA256 is `79f9fed0f91da13de053a8e74caf7d287d8d6b79506aac6f7564f5c69deca267`; unchanged scientific config is `61529de3e7cf2e865d147ca11e54388590bb50e12d8702c388adb2ed7702f7be`.

## What passed

The existing native CPU `audit_fit` function was reused once per new complete fit. It reconstructed the original examples, membership, within-batch order, twelve epoch hashes, learning-rate schedule, target-token and active-position counts, finite loss/gradient/timing fields and changed-parameter contract. Every stdout fit event joined the saved JSONL record exactly.

Each candidate has 12 epochs × 1,500 questions, 564 optimizer updates, 9,000 microbatches and 4,280,916 supervised target-token exposures. Each epoch has 47 logical updates, 356,743 target tokens and a final 28-example batch. Across the eight candidates there are 4,512 updates, 72,000 microbatches and 34,247,328 target-token exposures. The unique fitting set is still **1,500 questions**, reused across candidates and epochs; neither tokens nor 144,000 question-epoch-candidate exposures are independent replications.

All eight raw checkpoints contain exactly the six source-backed tensor keys, with the proper rank-4/rank-8 shapes and dtypes. The hidden-size buffers equal 4096; all tensors are finite. Only raw CPU tensors were loaded, with `weights_only=True`; no base-model weights, model forward, generation, grader, generic test suite or GPU qualification ran. Checkpoint hashes match before and after capture and CPU audit. This does not independently replay optimizer state or prove a final checkpoint's task performance.

## Training trajectories, not evaluation outcomes

The table shows target-token-weighted **online training loss** aggregated across updates within an epoch. Parameters change during the epoch, so these are not post-epoch evaluations of a fixed checkpoint, nor losses measured on selection or sealed-test questions.

| Candidate | Policy | Rank | Site | Epoch1 loss | Epoch12 loss | Epoch12−11 | Sum of update timers, s |
|---|---|---:|---:|---:|---:|---:|---:|
| 00 | prompt | 4 | 17 | 0.759791 | 0.502730 | -0.001889 | 2562.464 |
| 01 | prompt | 4 | 23 | 0.799403 | 0.538206 | -0.001255 | 2126.729 |
| 02 | prompt | 8 | 17 | 0.756108 | 0.493714 | -0.002604 | 2595.900 |
| 03 | prompt | 8 | 23 | 0.787716 | 0.528579 | -0.002365 | 2147.246 |
| 04 | full | 4 | 17 | 0.756879 | 0.487978 | -0.001804 | 2570.794 |
| 05 | full | 4 | 23 | 0.792311 | 0.515016 | -0.001239 | 2105.266 |
| 06 | full | 8 | 17 | 0.746379 | 0.466264 | -0.003088 | 2588.511 |
| 07 | full | 8 | 23 | 0.771146 | 0.491629 | -0.002557 | 2193.679 |

All eight online epoch-loss trajectories decrease monotonically. For matching rank/site pairs, final-epoch full-minus-prompt loss is −0.014752, −0.023190, −0.027450 and −0.036950 in the table's rank/site order. This describes optimization of the training objective; it does not establish that applying the adapter during generated reasoning improves correct, completed answers. Rank8 also has lower final online loss than rank4 in the four matching policy/site pairs; no layer, rank or policy is promoted on that basis.

The pre-clipping gradient norm exceeds 1 in 183/4,512 updates, with maximum 5.883746. This is not a clipping failure: the recorded value is explicitly pre-clip, and these observations do not measure post-clip norms. Each candidate's first update has learning rate 0 and no changed parameter, as frozen by the warmup schedule. The remaining updates change two or three parameter tensors; the contract does not demand that all three change at every step.

Late-epoch losses still decrease by about 0.00124–0.00309 between epochs 11 and 12. This does not establish convergence, justify extra epochs, or support early stopping; the original final 12-epoch checkpoint rule is unchanged. There is one fit seed and no uncertainty estimate for training-seed variation.

## Time, resource accounting and missing producer receipts

The sum of measured update timers is 18,890.589395914227 seconds (5.247385943309507 H100 component-hours). These timers are nested inside fitting and ongoing worker lifetimes; they are not additional charges, actual full-fit duration, end-to-end training cost or serving latency. Cross-worker timer differences are not a controlled hardware-speed comparison.

The helper's receipt-like input was constructed from frozen expected counts/orders and captured checkpoint hashes. Its `seconds` field is the same-FSK capture timestamp minus original Docker `StartedAt`: a conservative elapsed wall-clock envelope containing loading, zero generation and potentially selection. The helper's returned `fit_seconds` was removed; the result names the envelope `fit_duration_upper_bound_seconds` and records `measured_fit_duration_seconds:null`. It is **not a producer fit receipt** and must never enter `audit_component_cost` or substitute for measured terminal cost.

Native CPU audit took 18.527369648218155 seconds; its controller took 19.649426262825727 seconds. The local transfer completed before verification. The 541,716,480-byte fit archive, SHA256 `0237ddfdbc119af2a942f1debc48c33ca9a41807ba8f7c1433c3c3c09175e3da`, includes 42 payloads: 16 fit/checkpoint files plus 26 capture/dispatch/log files. The inventory SHA256 is `4fde669e02cad87609df0fd52b735e322e0448c6d5f7e68001ba794af29f32d5`.

## Unchanged next decision

Complete all eight original 256-question selection panels. Join the real producer fit receipts to these exact checkpoint/order/count/target hashes and verify actual fit duration, final base versions, terminal status and every worker's original inclusive deadline. Reuse this input-matched coverage/checkpoint audit; do not rerun its unchanged checks merely for another green receipt. Reuse the earlier complete zero audit only after its source/config/raw hashes and final producer joins match.

Only then apply the frozen primary-count/rank/site rule, paired rescue/harm and EOS/cap analyses, fixed-support sensitivity, source strata and exploratory BCa intervals to all eight candidates. Training loss and output length do not replace the primary endpoint. A full candidate exceeding zero 198/256 is an exploratory initialization condition, not sufficient evidence of SOTA or automatic response-launch authority.

Scientific strength remains 15/36, with S5/S7 at 0. Strongest comparators, three architecture families, three constructs, protected utility, uncertainty/mechanism evidence and an empirical ICLR-ready paper remain open. No scientific stopping rule, allocation, seed, metric, cap, candidate or compute ceiling changed.
