# Outcome-score observer benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before basis-completion, fit-split output, or observer implementation.

## Decision

Can a deployable causal-prefix observer predict eventual exact AIME correctness from frozen Qwen2.5 layer-14 states well enough to justify outcome-score steering? This stage tests measurement and generalization, not steering efficacy.

Claim C66 is eligible only if the final frozen observer passes OST.1 once on the untouched validation allocation. A fit-selection failure closes the route before calibration. A calibration failure closes the route before validation. A validation failure closes the route before AIME 2024 steering development.

## Source hierarchy and allocations

The allocation and source hashes remain those in `benchmark-freeze-outcome-score-transport-v1.md`. The version-5 interface supersedes only the original model and decoding interface: Qwen2.5-7B-Instruct snapshot `a09a35458c702b33eeacc393d103063234e8bc28`, config SHA-256 `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`, layer 14, bfloat16, non-thinking chat decoding, temperature 0.7, top-p 0.95, 2,048-token cap, and the pinned strict Math-Verify evaluator.

| Allocation | Packet SHA-256 | Questions | Rollouts | Authorized use |
|---|---|---:|---:|---|
| basis | `8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7` | 32 | four, indices 0-3 | representation basis only |
| fit | `abee6c7c85376e085756d2f60a427969105ce908abbe00a83027a55931d0fd1b` | 256 | four, indices 0-3 | observer architecture selection and refit only |
| calibration | `615d2a0294c7d0ee602ce0f600b2e53af4e10a7e22958b31033c380b629a2596` | 128 | two, indices 0-1 | probability, support, transport, metric, gate, and budget calibration only |
| validation | `0e9eb7b0749488a670254d6ac5fba49f85f32c040e77029e2b8a4aec4a2f5fec` | 128 | two, indices 0-1 | one-shot OST.1 test only |

Stage C already supplies basis rollout indices 0 and 1. `basis_completion` adds only indices 2 and 3. The basis allocation cannot fit or select observer parameters. The fit worker cannot load calibration, validation, reserve, AIME 2024, AIME 2025, AIME 2026, or MATH-500.

The exact benchmark prose requiring four basis and fit rollouts and two calibration rollouts takes precedence over the inconsistent `1,536 fit` summary cell in `research-plan-outcome-score-transport-v1.md`. No allocation or output count changes.

## GPU source collection

Each stage uses eight sorted contiguous question shards and one resident model per H100. `basis_completion` assigns four questions and eight main rollouts per worker. `fit` assigns 32 questions and 128 main rollouts per worker. Each worker repeats the first rollout under a zero action, stores raw completion and score fields, stores every generated-token layer-14 state, reloads the trace artifact, and writes a hash-bound receipt.

Source promotion requires all eight receipts; exact allocation, packet hash, model, layer, evaluator, group, rollout, row, and trace identities; exact zero-action replay; finite traces aligned one-to-one with generated tokens; no parse failure; no cap hit; peak memory below 70 GiB; and parallel wall below 20 minutes for basis completion and six hours for fit.

Fit additionally requires at least 128 correct and 128 incorrect rollouts, at least 64 question groups with a correct rollout, and at least 64 with an incorrect rollout. Each absolute progress bin `[0,512)`, `[512,1024)`, `[1024,1536)`, and `[1536,2048)` must contain prefixes from at least 32 question groups and at least 16 rollouts in each outcome class. Failure closes the observer route without changing cap, bins, model, layer, prompt, parser, seeds, or subset.

## Representation basis

The basis uses all 128 basis rollouts. Within a rollout, each nonempty absolute progress bin receives equal total weight and each state within a bin receives equal weight; every rollout receives equal total weight. Compute the weighted center. Form a supervised anchor as the equal-rollout difference between correct- and incorrect-outcome mean states. Reject a zero or nonfinite anchor.

The first basis vector is the normalized anchor. Project weighted centered states orthogonally to it and compute the next 127 left-to-right ordered principal directions. Canonicalize each vector sign by making its largest-magnitude coordinate positive. The resulting 3,584 by 128 orthonormal matrix and center are frozen before loading fit outputs into an observer trainer. Candidate ranks are nested prefixes 32, 64, and 128.

## Prefix observer selection

All observers receive centered basis coordinates and normalized absolute progress `t/2048`. Candidate predictions are causal and available at the controlled state.

The frozen candidate set is:

1. current-state L2 logistic regression at ranks 32, 64, and 128;
2. a one-layer GRU with hidden width 64, input LayerNorm, no dropout, and a scalar sigmoid head at ranks 32, 64, and 128.

The GRU carries its hidden state over the full observed prefix at training and deployment. It is not trained on a full sequence and queried with a length-one sequence. Training uses AdamW for 20 epochs, learning rate `1e-3`, weight decay `1e-4`, gradient-norm cap 1.0, and deterministic seeds derived from `ost-observer-v1`, candidate ID, and fold. Logistic regression uses L2, `C=1`, LBFGS, tolerance `1e-6`, and at most 1,000 iterations.

Loss and evaluation checkpoints are eight deterministic evenly spaced states in each nonempty absolute progress bin. Each question, rollout, and nonempty bin receives equal weight. Folds never split a question. Three folds are assigned by SHA-256 of `ost-observer-v1|group_id` modulo three.

Select by lowest mean held-out question-aggregated Brier score across the three folds. A candidate is eligible only with finite complete predictions, global AUROC above 0.60, positive Brier improvement over the fold-training constant-rate predictor, and AUROC above 0.52 in every progress bin meeting the frozen source-support gate. Within Brier difference 0.002, prefer logistic over GRU and then lower rank. If no candidate is eligible, close before calibration. Refit only the selected architecture on all fit groups.

## Controls and calibration

Fit reports the constant-rate predictor, a deployable progress-only logistic predictor, all six frozen observer candidates, and a terminal-state logistic diagnostic evaluated only at the last state. The terminal diagnostic is not treated as a deployable prefix baseline.

Only after fit selection passes may calibration rollouts be generated. Calibration fits a nonnegative-slope Platt map on frozen observer logits, score support, four empirical monotone transport maps, metric ridge, gate threshold, and one intervention-count budget. These objects and their exact selection rules must be frozen in a separate calibration implementation record before calibration output is loaded.

## One-shot validation and OST.1

Only after the complete calibrated artifact is frozen may validation rollouts be generated. The independent unit is the question group. OST.1 requires finite complete predictions, global AUROC strictly above 0.65, positive paired question-level Brier improvement over the fit-rate constant predictor with a one-sided 95% BCa lower endpoint above zero, and AUROC strictly above 0.55 in every supported progress bin. Use 10,000 group resamples with seed `20260907`. Report calibration curves, rollout and progress strata, outcome balance, year strata, leave-one-year-out metric sensitivity, response length, missingness, wall time, peak memory, generated tokens, and parameter bytes.

No observer architecture, rank, epoch, loss weight, checkpoint rule, calibration family, progress bin, threshold, or support rule changes after the corresponding output exists.
