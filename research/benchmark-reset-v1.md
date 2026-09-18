# Reset benchmark and next decisive experiment

Date: 2026-09-07. Status: research-design decision recorded before new-method outcomes; not yet an executable or sealed benchmark. Dataset manifests, comparator artifacts and runtime conformance are explicit prerequisites below. This document governs the new increment only. It does not reopen or amend any earlier frozen route.

## 1. What SOTA means

The target is the best measured task-success versus end-to-end resource frontier among admitted, strong steering and adaptation methods on the same frozen base models and questions. It is not the best language model in the world and not a comparison of unrelated paper headline scores.

Three labels are distinct:

1. **Candidate:** a prospective development comparison passes the practical gain and uncertainty gates.
2. **Registered reasoning-steering SOTA:** one locked method beats every admitted feasible comparator on the sealed reasoning panel, with simultaneous uncertainty, capability protection and measured resource limits.
3. **Broader steering SOTA:** the same procedure passes the project's broader claim-specific behavior and architecture gates. A math result alone cannot earn this label; the existing full-paper breadth target of three architecture families and three behavior constructs is not satisfied by a two-model math result.

Do not require exceeding a published numerical marker obtained with different models, splits, judges or search budgets. Instead reproduce that paper's native setting separately when feasible, and rerun admitted methods on the common setting. This supersedes cross-protocol numerical promotion wording for this new increment, without changing any old decision.

## 2. The one question to test first

Does learning the consequences of actual, finite residual interventions enable repeated online control that improves exact answer correctness beyond a strong fixed representation adapter, without extra online candidate generation?

The candidate is provisionally called **intervention-response control (IRC)** as a descriptive label, not a priority claim. A controller predicts action values from currently available state and recent history. It does not differentiate a correctness probe trained only on natural trajectories. The key evidence must isolate intervention-response supervision and repeated feedback, because RISER already trains latent routing on task reward, DSAS/CLAS already provide dynamic coefficients, and FPCG/ACTS already provide outcome-guided text control.

## 3. Controlled object and estimand

Let `s` contain the token prefix, exact causal cache and decoder position before an intervention. Let `o(s)` contain only the current selected-layer activation, a fixed recent-state summary, elapsed token count and remaining budget. The controller cannot observe a reference answer, future completion, future length, held-out reward or another action's evaluation outcome.

For allowed action `a` and fixed continuation policy `pi_ref`, estimate

`Q(s,a) = E[terminal exact correctness | apply a for 64 tokens at s, then continue with pi_ref]`.

All actions branch from the same prefix/cache. Pair continuation randomness where supported, randomize execution order, and isolate branch cache/RNG state. A difference of responses from unrelated prefixes is not this estimand. `A(s,a) = Q(s,a) - Q(s,a_ref)` is a useful regression target, but subtracting a common reference does not itself change the optimal action or create novelty.

The deployed policy is subsequently evaluated as a complete repeatedly controlled rollout. One-step response estimates under `pi_ref` do not establish the performance of repeated decisions on shifted states. The full-rollout experiment is mandatory.

## 4. Models, data and exposure

Primary reasoning model: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, thinking enabled. This is a common comparator backbone, not a claim that it is the latest or strongest open model. Independent architecture-family replication: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`; its exact revision/config/tokenizer/weight manifest must be pinned before collection. Do not silently replace unavailable models.

Initial primary inference budget: 8,192 generated tokens, temperature 0.6, top-p 0.95, with each model's documented chat template and a shared mathematical-answer instruction. A separate 32,768-token frontier point is required before claiming an advantage at the native long-reasoning budget used by ACTS. The 8,192-token study is a new task-utility protocol, not a rescue of the closed Qwen3 observer sentinels. Truncations count as failures, not grounds to extend the cap after results.

Use public mathematical problems and solutions; no new manual labels. Prefer the MATH release linked by its [authors](https://github.com/hendrycks/math), but first verify that the chosen snapshot preserves official train/test identity. The linked mirror is not automatically an authoritative split. All source archives, revisions, normalized-problem hashes, solution hashes and duplicate-cluster identities must enter a manifest.

Proposed allocation, to be materialized once using seed 20260907 after deduplication: from the official training pool, 1,500 adapter-fit questions, 512 intervention-response questions, 256 model/strength selection questions, and 256 development-evaluation questions, all disjoint by problem cluster. A 128-question response pilot is a fixed prefix of the response allocation. Remaining training questions stay reserve. If these counts cannot be met after cross-corpus exclusions, stop before outcomes and revise the design transparently, not by selecting convenient examples.

The primary sealed test is the remaining official MATH test set after exact/near-duplicate exclusion against all training, selection and previously exposed project questions. Do not use MATH-500 to tune and then count those questions again in a larger test. Require at least 3,000 eligible independent problem clusters for the intended 3-percentage-point claim; otherwise recalculate detectable effect before collection. GSM8K official test supplies a separate generalization result. AIME 2024/2025 and any later verified release are small temporal stress tests, not the powered primary endpoint. Existing historical AIME fit outputs remain exploratory and are not admitted as fresh training/validation evidence in this reset.

No split can establish absence from foundation-model pretraining. Claim adaptation/evaluation separation only; report public-benchmark contamination risk.

## 5. Comparators that may not be omitted

| Track | Mandatory comparisons | Fairness rule |
|---|---|---|
| Common-data, low-overhead | Unmodified model; fixed reasoning prompt; CAA; static and token-adaptive CLAS-like coefficients; LoReFT; parameter-matched LoRA; task-reward-trained fixed-prompt RISER-like routing | Same training questions/solutions, source collection envelope, tuning budget, inference budget and evaluator. Count all controller and adapter parameters. |
| Mechanism controls | Natural-outcome reward-gradient steering (LRS-like); current-state-only IRC; prompt-only IRC; shuffled action-response labels; random policy with matched action occupancy; unchanged fixed adapter | Same action family and, where applicable, same state features. Controls separate reward training, data volume, online information and repetition from the proposed contribution. |
| Best available resource frontier | Released ACTS; FPCG-style future-value candidate selection; reward-trained ReFT/LoRA using the candidate's total training envelope | Give competitors the full allowed resource envelope. External pretrained controllers retain an explicitly separate data-provenance track. Additional online samples are charged, not prohibited to protect the candidate. |
| Conditional inclusion | Exact LRS, RISER or DSAS releases if all required code/weights become verifiable before freeze | A rebuild is labeled `-like`; a missing release is documented, not counted as a defeated method. |

The first experiment needs only the unmodified model, prompting, the strongest feasible fixed adapter, and the directly matched mechanism controls. Other competitors are required for promotion, not all for the first pilot. Geometry methods enter if the proposed action or claim depends on geometry; arbitrary reasoning gains are not fairly compared with an unrelated concept-steering checkpoint.

## 6. Minimal candidate and fixed search space

Start from a fitted LoReFT action, not a new transport solver. Fit ranks 4 and 8 at one declared middle post-block site. Site selection is limited to two preregistered normalized-depth locations, 1/2 and 2/3; record the exact integer sites after inspecting model architecture. Compare fixed full-generation and prompt-only application. Baseline tuning uses only the 256 selection questions and the shared compute envelope.

Select one positive static strength `g_ref` before opening response outcomes. If no feasible nonzero adapter improves the selection objective, the action family has failed this entry condition; do not invent a post hoc vector library. Freeze three actions `{0, g_ref/2, g_ref}` applied for the next 64 tokens, followed by the fixed reference adapter. This makes the immediate scientific question whether selective attenuation of an already useful action buys additional correctness and protection.

Train an action-value head on the 512 response questions, grouped by question. Compare a linear head and a one-hidden-layer head; use the same three fixed regularization settings for both. Features are current action-subspace coordinates, their running mean over the previous four decision points, and normalized elapsed/remaining budget. Final trace length is prohibited. Fit preprocessing inside each training fold. No SAE, GRU, geometry estimator or generic training framework is needed for this first test.

Choose the cheapest eligible head on grouped out-of-fold action utility, not AUROC. Fit the selected head on the response-training allocation; selection thresholds use only the designated selection split. Emit every candidate's predictions, not only the winner. A matched control receives the same state features and training-token budget but learns from natural outcomes rather than action responses.

## 7. First decisive packet

1. **Runtime conformance:** 16 fixed training questions. Verify standard scoring, all-output accounting, zero-action identity within the frozen execution engine, branch-cache isolation, exact row-key joins, one evaluation per declared output, and deterministic replay. Benchmark microbatches 1, 4 and 8 on the same workload. This is engineering evidence, not a method screen.
2. **Response pilot:** 128 fixed questions, one predetermined cut per question (128 or 512 tokens, assigned by hashed identity), three actions, eight continuation seeds: at most 3,072 continuations. Already terminated trajectories take an absorbing no-action transition; keep them in the population accounting. Record the at-risk subset separately. Four seeds may construct a privileged selector; the other four evaluate it, with no outcome reuse. This diagnostic is not an upper-bound theorem or a deployable policy score.
3. **Response training:** only if the pilot shows usable action-response variation, collect the remaining 384 response questions under the same contract. Train the heads and controls. The pilot questions remain training data, never validation.
4. **Full-rollout development:** evaluate the selected policy and fixed controls on 256 untouched development questions, four rollouts each. Update the action every 64 generated tokens. This stage estimates the deployed policy, not a teacher-forced score.

The response-pilot decision is deliberately modest: if the held-out-seed privileged selector's 95% upper endpoint for gain over the fixed reference is below 2 percentage points, close this finite action family for insufficient demonstrated opportunity. Otherwise proceed only if group-held-out policy learning shows positive utility and no cache/scoring defect. An uncertain tiny pilot is not evidence of impossibility; it can justify only the already declared larger response packet, not an unlimited action search.

Development promotion requires at least +3 percentage points in mean correctness over the selected strongest fixed baseline, a one-sided question-cluster 95% lower endpoint above zero, and preservation gates below. It also requires better utility than the prompt-only and natural-outcome controls. A positive mean without the interval or attribution gates remains exploratory, not SOTA. Failure closes this candidate; no new coefficient grid on the same development questions.

## 8. Outcomes, uncertainty and failure handling

Primary outcome is exact mathematical correctness per problem, averaged over that problem's fixed rollouts. Use pinned `math-verify==0.9.0` with the public benchmark's declared extraction configuration and gold handling. The current AIME wrapper only accepts integer gold and cannot be reused unchanged for arbitrary MATH answers. Conformance must cover fractions, expressions, sets, multiple boxes, missing boxes and legitimate punctuation before new outputs are inspected.

Unparseable, empty, incorrect and budget-exhausted model outputs score zero. A corrupted artifact, missing worker, invalid gold, evaluator exception or hash drift is a technical failure, not a missing answer silently excluded from the denominator. Recovery of a technical failure retains the original record and may not change hypotheses or select only favorable rows. All original v1-v20 closure rules remain untouched.

Resample problem/duplicate clusters, never token rows or individual rollouts as if independent questions. Report absolute percentage-point differences, discordant-pair counts, intervals and complete failure rates. Primary test contrasts are frozen before test access; use simultaneous Bonferroni-adjusted paired intervals against every admitted feasible comparator. Fixed-model results and a fixed-panel mean do not estimate a population of all architectures.

Planning approximation: with paired discordance 0.25, detecting a 3-point difference at 80% power needs approximately `(1.96 + 0.84)^2 * 0.25 / 0.03^2 = 2,178` independent questions for one two-sided 5% comparison. Three comparisons require roughly 2,900 under a normal Bonferroni approximation. Actual sample size and multiplicity are recalculated from development variance without looking at test outcomes. Thirty AIME problems cannot supply this precision; more rollouts do not create more independent problems.

## 9. Capability and mechanism protection

Report subject, difficulty, prompt length, action frequency, baseline-correctness and generation-length strata. Protected estimates include GSM8K task accuracy and fixed-adapter-correct answer retention. Use a 2-percentage-point non-inferiority margin, with one-sided multiplicity-adjusted intervals; a nonsignificant loss is not protection. If strata cannot be powered, disclose them as uncertain and do not claim universal preservation.

Report all-output repetition/empty/truncation rates, mean and p95 latency, throughput at fixed concurrency, peak memory, generated and rejected tokens, forward/backward calls, and reference/controller/judge cost. Next-token KL and hidden-state norms are diagnostics, not substitutes for correctness or capability. Gate calibration must be measured on states actually visited by the policy; off-policy diagnostic accuracy alone is insufficient.

The mechanism claim requires the response-trained online head to improve over the same-capacity natural-outcome head, over prompt-only routing, and over a matched-occupancy random/shuffled control. If the adapter alone explains the result, retain the adapter as the winning baseline and reject the additional control contribution.

## 10. GPU execution policy

Use only the allocated FSK host after checking active jobs. Eight resident 7B/8B workers are preferred to repeatedly loading models; one process per GPU with two intra-op CPU threads and one inter-op thread is the validated starting point. Do not infer effective utilization from reserved memory.

The throughput smoke chooses a microbatch before scientific outputs. Freeze engine, dtype, attention implementation, padding, per-row RNG semantics and batch scheduling. Batched floating-point kernels need not reproduce an older serial sampler bitwise; require reproducibility and exact zero-action parity within the frozen new engine, and document any cross-engine difference before selection. Never change batch semantics midway to speed up a selected condition.

Branch from shared prefills and retain only states at decision points plus output/receipt hashes. Dense traces at every token are not needed for action-value fitting. Validate completed chunks on CPU while GPU workers continue; persist immutable 16-question chunks and a final completeness index. CPU analysis must not block the GPU queue. Cancel only the current candidate's future jobs after its gate closes.

Compute ceilings for this increment: runtime smoke 1 H100-hour; fixed-baseline fitting/tuning 8; response pilot 12; remaining response collection plus head fits 36; full-rollout development 16. These are hard maximum allocations, not predictions of required time. Estimate the next stage from measured token throughput before launch. If the declared packet cannot fit, stop it as budget-incomplete rather than changing the sample size or presenting partial output as a result. Multi-model/sealed evaluation receives a separate prospective cost receipt after development promotion.

Operational dashboard: completed unique question-action-seed tuples, tokens/second/GPU, useful-completed-rows/GPU-hour, GPU busy fraction, queue starvation, parse/evaluator failures, peak memory, elapsed and projected remaining cost. Notify on a new decision, failure or required intervention, not unchanged polling.

## 11. Code discovery and minimum wiring

Existing `outcome_score_runtime.py` owns generation, model seeding and traces; `torch_runtime.py` owns layer resolution and action hooks; `splits.py` owns stable grouped allocation; `statistics.py` owns paired/group bootstrap; shard/analyzer scripts own receipt joins and hashing. The prior source-stage contracts are closed and must remain unchanged.

Reuse the pinned upstream ReFT implementation for the action. The minimum missing project code is an isolated branch-continuation path, a general-math adapter to the existing evaluator, and a small action-value fitter. Extend existing modules where their contracts genuinely match; do not pretend the current batch-size-one trace collector already supports batched independent branches. No new OT solver or observer-basis package is justified.

Before code: pin upstream ReFT and exact data/model artifacts; inspect the actual callable contracts; write failing tests for scoring, branch isolation, absorbing termination, action duration and shuffled-control identity. Then implement the smallest integration. This document is discovery for the plan, not a claim that those implementation steps have been completed.

## 12. Broader-paper gate and stopping

Only after a positive reasoning result, instantiate the same response-learning procedure for open-ended concept steering on Gemma-2-2B and 9B, with released FLAS and the strongest available comparators. Use previously unexposed concept/prompt groups and complete official AxBench C/I/F evaluation. No rule evaluator or local proxy substitutes for that construct. Selective control must improve concept success while retaining instruction and fluency, with concept-clustered intervals and matched costs. This second family's exact allocation and judge receipt are prerequisites, not silently assumed parts of a math win. A third, distinct construct such as faithful use of supplied evidence still needs its own comparator, exposure and power contract. The source-start full-paper breadth and specificity gates remain unpassed; this increment does not silently lower them.

If the new candidate fails its decisive development gate, do not return to v20, relax a certificate, tune the old observer bins, or declare a null-result paper sufficient. Diagnose the specific action, information or learning bottleneck and propose a genuinely different, predeclared increment using the measured strongest baseline. The current candidate is allowed to fail; the overarching research goal remains active.

Current unpassed prerequisites: authoritative split/overlap manifest; primary/replication model receipts; pinned callable ReFT path; evaluator and batched-branch conformance; response-policy implementation; every new behavioral comparison; independent final review; current ICLR submission compliance. None is represented as already passed.
