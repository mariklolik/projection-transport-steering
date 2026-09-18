# Operational definition of steering state of the art, version 0.1

Status: preregistration input. This document defines what must be beaten before the target method is designed or evaluated on confirmatory data.

## Why there is no single inherited leaderboard

Activation-steering papers use different models, behaviors, judges, source examples, layer and strength searches, inference passes, and capability constraints. A raw gain from one paper is therefore not a transferable state-of-the-art threshold. Three claims must remain separate:

1. task-local SOTA: the best score under one paper's original task and protocol;
2. method-class SOTA: the best comparable method within a family such as additive, affine, nonlinear transport, conditional routing, or finetuned control;
3. registered constrained SOTA: the best method on the new common panel under identical data, tuning, pass, specificity, uncertainty, and safety rules.

This project targets the third claim. Original-paper reproduction is reported separately and can support only task-local claims.

## Existing reference frontier

| Frontier | Reference methods | Evidence that makes it a strong comparator | What cannot be imported directly |
|---|---|---|---|
| Prompt and trained control | system prompting, LoReFT, ReFT-r1, AxBench controls | Prompting is exceptionally strong in AxBench; LoReFT leads its representation methods on average | AxBench concept prompts, ten-prompt steering samples, and Gemma-only cells do not define broad behavioral SOTA |
| Fixed linear steering | CAA or DiffMean, ITI | Low-cost one-pass standards with broad use and strong detection baselines | Best-layer and multiplier sweeps vary across papers |
| Affine transport | AcT, MiMiC, LinEAS | Coordinate, full-covariance, and jointly fitted multi-layer distribution matching | Different target distributions, fit sizes, and offline compute |
| Conditional steering | CAST, SADI, PCHI, DSAS, FASB, IDEEA, GAPS, REINS-Gate, W2S-Multi, Latent Reward Steering | Condition gates, learned strength, downstream confidence control, backtracking, per-input direction or support routing, adaptive stopping, per-dimension posterior masks, and state-specific outcome-reward gradients | FASB, REINS, W2S-Multi, and LRS use extra trajectory, reward, or gate work; PCHI is task-specific; GAPS selects operating points on evaluation curves; LRS uses per-cell hyperparameter sweeps and its pinned release lacks an exact runnable artifact chain |
| Sparse coordinate control | AUSteer, GAPS, SADI | Fine-grained adaptive intervention can outperform dense block updates with a smaller footprint | Coordinates are architecture- and basis-dependent; published selection budgets are not matched |
| Geometry, depth, and feedback | StTP, StMP, Spherical Steering, Selective Steering, Gaussian depth schedule, COAST, FishBack, A-LQR, ASC, Policy Gradient Steering | Projection-aware and norm-preserving actions, depth allocation, output-functional minimum action, KL-bounded strength, explicit closed-loop layer dynamics, and rollout-gradient trust-region calibration | Several headline configurations come from full layer/strength/angle sweeps; FishBack is narrow and expensive but directly occupies Fisher minimum-distortion steering; PGS is not evaluated on language models; ASC is reasoning-specific; geometry, Jacobian cost, and objectives differ from transport |
| Nonlinear transport | FLAS, INNSTEER, COBRAS, Manifold Steering, Riemannian-Manifold Steering | Concept-conditioned flows, invertible latent translation, density bridges, and learned geometric paths | Map training, online integration, full-state replacement, or optimization costs are much larger and action supports differ |
| Learned action generation | FLAS, HyperSteer, Steer Like the LLM | Input-conditioned flows, hypernetworks, and prompt-mimicking actions can beat prompting on AxBench cells | Much larger fit budgets, teacher/judge coupling, and different parameter classes prevent direct transfer |
| Source construction | execution-boundary and tail-subtracted mean/PCA directions | Recent evidence shows source context can matter more than estimator choice | Published best-over-grid values are development references, not confirmatory thresholds |
| Specificity and safety audit | cross-encoding, SteerCheck nulls, broad cross-behavior effects | Detects identifier control, generic opposition, and collateral regressions | These are validity gates rather than competing controllers |
| Sign and safety correction | ITI-RRF, constrained safety-component ablation | Fixes detection-control inversion and steering-induced jailbreak regressions | Corrections are post-hoc and do not choose the best action per input |
| Externality and alternative control | OPIUM, ALTSTEER | Directly optimizes downstream side effects or desired safe alternatives | Per-vector fitting, staged schedules, leakage, and target-specific scorers prevent raw headline transfer |
| Fundamental validity | Steering off Course, Non-Identifiability, Cylindrical Representation, Perfect Detection Failed Control, Steering Awareness, Rogue Scalpel | Tests cross-model reliability, uniqueness, observer-action alignment, awareness, and safety failure | These are promotion gates and threat models, not controllers |

The strongest inspected published reference points are not directly comparable but define reproduction targets: FLAS reports held-out AxBench HMean 1.015 on Gemma-2-2B and 1.113 on Gemma-2-9B at fixed flow time 2, versus prompting at 0.762/1.091 and HyperSteer at 0.608/0.934; FishBack reports median baseline-to-method off-target-KL ratios of 1.42-6.52 on GPT-2 Small, 2.20-3.50 on Llama-3-8B, and 1.77-3.61 on Qwen3-8B at matched verb-concept probability; GAPS reports Gemma-3 4B toxicity of 0.48% under its 5% perplexity and 3% MMLU-probability budgets, versus 3.52% for DSAS and 6.52% unsteered; AUSteer reports five-task means of 61.34, 83.96, and 83.21 on Llama-2-7B, Gemma-2-9B, and Qwen-3-8B, versus SADI's 59.49, 82.05, and 80.80; Steer Like the LLM reports an AxBench score of 1.120 on Gemma-2-9B versus HyperSteer 1.091 and prompting 1.075; Spherical Steering reports Llama TruthfulQA MC average 53.17 versus 38.16 unsteered and Qwen 51.59 versus 39.15; INNSTEER reports aggregate alignment probability 94.52 versus 78.61 for its ODE comparator with perplexity 12.37 versus 11.57; REINS reports GUISE HRR/SRR/CR of 26.3/63.7/2.7 versus CorrSteer-A 43.7/23.7/20.3 on Qwen3.5-4B; ITI-RRF improves over ITI in 27 of 30 cells; constrained safety-component ablation returns mean attack success to near or below the unsteered baseline across three models while retaining most of the steering effect. These values are paper-specific frontier markers, not thresholds for a cross-paper SOTA claim.

The exact released implementation, revision, original dataset result, and reproduction receipt for every comparator must be added before that comparator can define the empirical frontier.

## Registered constrained utility

For model-behavior cell `c` and method `m`, define paired target change `T_cm`, capability change `C_cm`, collateral vector `S_cm`, severe-event count `E_cm`, deployment passes `P_cm`, and total tuning compute `G_cm`.

A method is feasible in a cell only if all conditions hold:

- the one-sided 95% lower confidence bound for `C_cm` is above the frozen non-inferiority margin;
- the cross-encoding and matched-KL specificity gate passes;
- every protected collateral endpoint passes its mean and tail margin and no undisclosed severe event occurs;
- the method stays within the declared pass and compute budget;
- raw paired outputs, artifact hashes, environment, and analysis receipts are complete.

The cell utility is the paired target gain for a feasible method and infeasible otherwise. Pareto plots retain target gain, capability, collateral risk, latency, and tuning compute separately; no hidden scalar weights are used for the headline comparison.

## SOTA promotion gate

The target method earns the phrase “state of the art” only for the registered panel and budget if all of the following hold after the single sealed evaluation:

1. its hierarchical mean paired target gain exceeds the strongest feasible baseline and the one-sided multiplicity-adjusted 95% interval for the difference excludes zero;
2. it is non-inferior to the cell-wise best feasible baseline within a frozen target-gain margin in every behavior family and every architecture family;
3. it is feasible under capability, specificity, collateral, pass, compute, and receipt constraints;
4. it lies on the empirical Pareto frontier at both matched total tuning compute and matched deployment passes;
5. the joint method beats its action-only and observer-only ablations, so the claimed coupling is supported;
6. no confirmatory cell is removed, relabeled exploratory, or used to change the method.

If only conditions 2-6 hold, the supported wording is “competitive” or “Pareto-optimal,” not SOTA. If condition 1 holds only for a behavior or model subset, the claim names exactly that subset. A published headline number outside this common protocol is never described as beaten by an incomparable result.

## Development target before sealed evaluation

Development can proceed to the sealed panel only when the candidate:

- beats the strongest reproducible baseline on one development model-behavior cell with a one-sided paired interval above zero;
- passes capability, specificity, collateral, and pass gates on that cell;
- preserves the advantage across the preregistered difficulty, baseline-correctness, confidence, length, and observer-score strata without an unexplained sign reversal;
- remains Pareto-optimal after matched tuning compute;
- has a complete oracle-observer, oracle-action, source-position, and null-direction decomposition.

This gate is intentionally harder than reporting a positive mean. It is the point at which a proposed method becomes a credible SOTA candidate rather than another steering variant.

## Semantic-outcome AxBench frontier amendment, 2026-09-07

The active candidate is evaluated first on the exact AxBench open-generation task rather than on an internal likelihood proxy. The inspected published markers are FLAS HMean `1.015` for Gemma-2-2B and Steer Like the LLM HMean `1.120` for Gemma-2-9B. They are lower bounds on the frontier, not sufficient comparators.

Registered AxBench SOTA requires, for each model, a complete common-protocol rerun of FLAS and every stronger feasible released comparator under the same concept list, five/five prompt split, judge, failure rules, and compute accounting. The candidate must exceed both the published marker and the strongest common-protocol comparator, with a multiplicity-adjusted one-sided concept-clustered interval above the latter. Both 2B and 9B gates, direct sequence-energy and component ablations, specificity, instruction, fluency, neutral-distribution, severe-failure, pass, and compute gates must pass. A teacher-forced likelihood gain, tuning-prompt result, missing CES receipt, or isolated 2B win cannot support SOTA wording.
