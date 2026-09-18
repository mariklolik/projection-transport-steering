# After the baseline failure: the next research decision

Projection-Transport Steering | 8 September 2026 | Research plan, not a SOTA result

Audience: research collaborators and empirical-method reviewers. Scope: the learning-objective bottleneck after the completed eight-candidate baseline, four nearest primary methods, implementation reuse, and the next falsifiable increment. This supplements the 7 September full retrospective; it does not replace its 176-attempt inventory or reopen closed experiments.

## Executive decision

Keep intervention-response control as the research question, not as an established contribution. First obtain a useful finite action. Test outcome-pair learning against positive-only imitation on the same base-generated training corpus, with a parameter-matched LoRA control. A successful fixed action only opens the response-control experiment; it does not itself establish novelty or SOTA.

Do not launch a fresh eight-way recipe search. Reward optimization, KL anchoring, Fisher calibration and preservation of successful reasoning already appear in close work. The next contribution must come from demonstrated utility of measured intervention responses and repeated feedback beyond strong adaptive and reward-trained alternatives.

The immediate executable dependency is efficient, provenance-preserving collection of base-policy training traces, followed by long-sequence objective qualification. Neither a new scientific training protocol nor GPU dispatch is admitted by this report. Exact versions, prospective runtime limits and tests remain required. That is an internal implementation dependency, not a renewed VPN blocker.

## 1. What the failures actually teach

The complete baseline-v3 analysis records 256 paired selection questions, one training seed and one rollout per condition. All eight candidates lose to zero. Best full-generation: 133 correct versus 198 for zero; difference -25.390625 percentage points, exploratory paired 95% BCa [-32.03125, -19.921875]. It repairs 8 failures but harms 73 successes; 69 harms terminate with a parsed, primary-false answer. Shorter output is therefore not a sufficient objective: caps fall from 43 to 8 while correctness falls sharply. These intervals are not selection-corrected or simultaneous.

Evidence: `research/irc-baseline-selection-analysis-v3.md`; native aggregate SHA256 `03433c5dc63d9f8ec16308e34a58754eccfb8cc36078d0b1c54a08c584aa8467`; closure `research/irc-baseline-selection-closure-v3.json`, SHA256 `17b8ded150c916b45052412838e05a71c932e38807924205c2b73e6e686eef96`; separate independent review `research/irc-baseline-selection-independent-review-v3.json`, SHA256 `c4cac8b7da3f621957139175d150850c28bd5e875314a6e945a5041896360b1a`. These unchanged audits are reused, not re-executed.

The broader failure history yields four constraints. Numerical equivalence does not imply behavioral utility. A detector fitted to natural trajectories does not identify the effect of changing an activation. Better proxy geometry does not establish protected task performance. Lower teacher-forced loss does not establish better deployed reasoning. The new baseline failure adds evidence for a supervision mismatch, but does not isolate its cause: capacity, placement, optimization and initialization are competing explanations.

The full and prompt candidates were separately trained with different masks and weights. Their contrast is not a decoding-only causal experiment. The source solutions were short before training, but observing shortened outputs afterward does not identify mediation. A future positive result under another objective would not retrospectively prove that target mismatch caused every earlier loss.

## 2. Four closest sources and what they rule out

### RISER: reward-trained latent routing is established prior work

Ye et al., RISER, Findings ACL 2026, July, pp. 4627-4644. A frozen backbone receives a mixture from a six-primitive library; successful configurations supervise warm-up and correctness-reward GRPO trains the router with token-level KL to the base. Appendix E computes the mixture once after prefill, then reuses it during decoding. It is not repeated response-conditioned replanning. [Official publication and PDF](https://aclanthology.org/2026.findings-acl.226/)

The published appendix specifies two router epochs versus five full-model GRPO epochs, whereas the main comparison says same compute budget. Actual parity remains unresolved. This is a reason to measure common resources, not to dismiss the method.

The author repository at `acafa4a726fd33e022a280004bfe4b57ca24de8c` contains partial source, but no router model source, steered-model source, GRPO training scripts or checkpoints in its complete tree. Inference imports a missing module. Correct the historical release description to published plus partial source release; exact reproduction remains unverified. [Pinned author source](https://github.com/gooogleshanghai/RISER-Orchestrating-Latent-Reasoning-Skills-for-Adaptive-Activation-Steering/tree/acafa4a726fd33e022a280004bfe4b57ca24de8c)

### PGS: reward gradients and Fisher calibration are not a new method here

Poupart, Beynier and Maudet, Policy Gradient Steering, arXiv:2607.27574v1, 30 July 2026. Section 3 averages return-weighted activation score gradients into a reusable additive vector for a frozen policy; directional Fisher curvature sets a local action-KL budget. Appendix D covers the corresponding local natural-gradient direction, and Section 7 names backtracking and preconditioning extensions. [Primary paper](https://arxiv.org/abs/2607.27574v1)

Its experiments concern gridworld, chess-policy canonical-action likelihood and football, not LLM reasoning. Some baseline intervention sites differ; adaptation is not uniformly beaten. Per-decision importance ratios do not correct state-distribution shift. Thus, applying the recipe to reasoning would be an empirical adaptation, not priority for reward-derived steering or guaranteed finite-scale protection.

The inspected author release at `8b733320735751e21ed0f95a39ad2b62706d247f` contains implementations and configurations, but was not executed here. PGS-like reward-gradient steering belongs in the common comparator packet if the candidate uses the same information. [Pinned author code](https://github.com/Xmaster6y/policy-gradient-steering/tree/8b733320735751e21ed0f95a39ad2b62706d247f)

### OGLS-SD: successful/failed guidance and base anchoring are already combined

Yang, Wang and Zhang, OGLS-SD, arXiv:2605.12400v2, 29 May 2026. It contrasts mean teacher logits conditioned on correct and incorrect on-policy rollouts, adds that difference to unprivileged base logits, then distills into LoRA parameters. Forward-KL guidance skips already-correct rollouts; a short positive-tail loss stabilizes length. The deployed object is an adapted model, not a frozen-base activation controller. [Primary paper and appendices](https://arxiv.org/html/2605.12400v2)

The study uses Qwen3-1.7B/4B, eight rollouts, 8,192-token training generation, only the first 1,024 tokens for distillation, and 38,912-token evaluation. The behavioral-judge appendix disables thinking to align with training; this does not establish equivalence to our thinking-enabled protocol. Its four-GPU hardware report supplies no absolute training-hours total or located training-seed accounting. Guidance pooling adds teacher forwards. Reported gains are not universal across every benchmark.

The implication is caution in both directions: public-solution imitation may distort reasoning, but on-policy distillation can also be unstable. A positive/negative contrast with an identity anchor is not by itself a novel explanation or preservation guarantee. Exact author implementation and public reviews were not verified in this bounded search.

### First-token REFT: exploration is a separate intervention

Kim and No, Where Rollouts Begin: Low-Load, High-Leverage First-Token Diversification for RLVR, arXiv:2605.28295v1, 27 May 2026. REFT here means Rollout Exploration with First-Token Diversification, not Representation Fine-Tuning. It chooses distinct first semantic tokens from the policy's top set, then continues ordinary rollouts and GRPO/DAPO updates. [Primary paper](https://arxiv.org/abs/2605.28295)

Appendix B acknowledges off-policy first-token collection without importance correction. The experiments adapt model weights through LoRA, mainly on Qwen2.5/Llama instruct backbones; they do not validate a removable Qwen3 activation controller. Relative GPU-hour comparisons are reported for selected configurations, not a universal zero-cost guarantee. At least one reported pass@64 cell declines slightly.

This suggests a possible later exploration control, not an extra component to add now. More diverse responses do not guarantee more learnable correct/incorrect action pairs. The public forum required browser verification; reviews and rebuttals remain unread, and no author-code release was verified.

## 3. The discriminating experiment

The selected increment is **offline outcome-pair action qualification**. Base-generated traces are on-policy only for the source base policy. Once adapters change, fixed-pair DPO is offline preference optimization, not GRPO, and has no newly established on-policy policy-gradient guarantee.

Use a two-by-two comparison: representation intervention versus parameter-matched LoRA, each trained with positive-only sequence imitation or DPO on the same question-cluster corpus. Positive-only controls compare learning recipes on identical positive traces; they do not isolate negative information alone, because DPO also changes loss geometry and reference anchoring. Freeze the same mixed-outcome/EOS-admissible question subset and the same positive traces and question weights across all four arms. The 1,500 questions are the source pool, not guaranteed DPO coverage. This restricted control is not the strongest all-positive SFT baseline, which remains eligible in the broader resource-frontier comparison. Keep public solutions as an archived earlier condition, not a revived arm of the closed family. This new comparison will not fully identify the cause of that older failure because its initialization and corpus differ.

| Comparison | What it can establish | What it cannot establish |
|---|---|---|
| DPO versus positive-only imitation within one action family | Whether outcome-pair learning improves this frozen prospective recipe | That target mismatch caused the old failure, or that DPO is novel |
| Representation action versus matched LoRA under the same objective | Whether the declared action/deployment recipe earns utility under equal resources | Isolating parameterization alone if site/position coverage differs, or superiority to a fully tuned adaptation |
| Useful fixed action versus later response-trained controller and controls | Incremental value of feedback and intervention-response supervision | Broad steering SOTA from a single math model |

Keep backbone revision, native thinking prompt, 8,192-token evaluation cap and grader unchanged. Retain the materialized 1,500 fit, 512 response, 256 selection, 256 development and 4,359 sealed-test clusters. Fit-source generation may use only fit questions. The already exposed selection panel stays exploratory; development and sealed questions must not enter target construction or optimizer decisions.

Before optimization, freeze source request IDs, per-request seeds, exact token outputs and termination reasons; define complete correct/incorrect/tied/capped/error handling. Do not retokenize generated traces through a different chat template or truncate them to the old 4,096-token training ceiling. If preferences use only complete EOS pairs, retain all excluded source outcomes and state that conditional training population. Give each question total weight one; multiple pairs and variable lengths are not independent observations. Freeze response-only likelihood normalization and pair sampling before inspecting adapter outcomes.

Match actual trainable scalars, not ranks: one LoReFT site has r(2d+1), whereas bias-free LoRA has the sum of r(d-in+d-out) over targeted projections. At d=4096 and r=8, 65,544 and 65,536 are a disclosed near-match, not exact equality. Site support, scaling, dropout, initialization, target information and complete tuning cost must also be registered. In particular, standard PEFT LoRA acts throughout the prompt, whereas IRC's historical full policy activates only first/last-seven prompt positions plus decoding. Either qualify identical position coverage prospectively or label the comparison as different deployment recipes; equal scalar counts do not remove this confound. A zero-initialized residual delta is an engineering control; it is neither a novel algorithm nor a task-retention theorem.

The action gate remains usefulness against zero, followed by measured rescue/harm and protected utility. The old condition full>198 stays attached only to its closed family. Any successor needs its own pre-outcome selection contract; it cannot inherit a pass. One exploratory training seed may qualify implementation, not seed robustness or SOTA. Final checkpoint, learning rate, number of epochs, preference coefficient, exact action sites and parameter allocation are not yet frozen. Do not convert these missing values into implicit defaults.

## 4. Cost changes the order of work

The previous complete zero generation consumed 9,846.580574 summed single-GPU seconds for 256 questions. At that observed mean, 1,500 questions times four responses would require 64.105342 H100-hours for generation alone. This is an arithmetic scenario using a different question population, not a measured new workload or upper bound. Startup, grading, reference scoring, fitting, failures and evaluation would be additional.

Baseline spending remains 23.053432 of 48 H100-hours, leaving 24.946568. Parallelizing over eight GPUs changes elapsed time, not the summed GPU-hours. The present native collector therefore does not admit that full source plan under the remaining allowance. No budget has been increased or silently transferred.

First qualify a throughput-oriented base-only collector, such as an existing batched inference engine, without rewriting the intervention runtime. Native and optimized source policies must be explicitly identified; equal weights do not imply identical sampled trajectories. Preserve question IDs, sampling parameters, request-local seeds, raw token traces, EOS and cap accounting. No factual throughput gain is claimed until measured.

The next implementation packet should fix a small prefix of fit-only questions and the intended four source seeds, test batch-independent request identity and exact scoring joins, then measure prompt lengths, generated lengths, tokens per GPU-second, useful complete pairs per GPU-hour and peak memory. A small prefix is an engineering pilot, not certification of corpus-wide pair supply or the complete budget. Before full collection, include conservative full-source length scenarios; separately qualify full-length paired-policy forward/backward and frozen-reference forward memory (reference under no_grad) and reserve all evaluation cost. A new source-engine version, full-length backpropagation and reference/action ownership require only their changed-contract tests. Repeating the old unchanged grader, mask or fit audits provides no new admission evidence.

Only after those measurements may a complete two-by-two schedule receive a prospective cost receipt. Include source generation, failed or tied source groups, reference log-probabilities, all four fits, tuning, all outputs, zero controls, CPU grading and checkpoint reloads. Declare any resource reallocation before launch. If the complete plan cannot fit, record budget-incomplete and redesign the increment before outcomes; do not trim unfavorable arms or run an unreported partial screen.

## 5. Minimum code and exact missing contracts

The repository already has layer hooks, prompt/full masks, raw-token generation, immutable chunk receipts, seeds, frozen-base checks and paired analysis. PyReFT's DPO example supplies likelihood and preference-loss references; its reward example instead trains a sequence-classification reward model. It is not a generative policy optimizer. Standard PEFT LoRA construction is the cleaner comparator surface.

Do not copy AxBench's custom preference wrappers wholesale. The source audit found missing-attribute/list wiring, a final-partial-batch indexing risk, independently sampled chosen/rejected action strengths, and SimPO normalization that includes prompt length. PyReFT's class named ReftPreferenceDataset supplies counterfactual chosen labels on rejected histories, not an ordinary DPO pair contract.

Preserve IRC's existing position and generation policies. Add only the missing admitted-pair representation and objective integration after failing tests establish response masking, reference bypass, weighting, gradient ownership and checkpoint replay. The current SFT fit_step rejects trainable parameters inside the model and requires every adapter gradient to be nonzero. Those checks cannot be reused unchanged for standard zero-B LoRA, whose A gradient initially vanishes. Source code existing on disk is not evidence that the new trainer runs.

Code evidence: PyReFT `dafd0995a366d7b47160a337dcc388eda7431821`; AxBench `41c8332543e5a631f9a8c0a9df38799893ace758`; source-only audit, no dependency or GPU execution. Exact file locators and hashes accompany the source ledger.

## 6. What would count as progress toward SOTA

A useful fixed action opens, but does not replace, the already specified finite-response pilot. Then compare repeated control against a strong fixed action, prompt-only and current-state controls, matched response-label shuffling, natural-outcome reward-gradient control, RISER-like reward routing and strong resource-matched adaptation. The comparison must separate training information, data volume, online feedback and repeated decisions. OGLS-like distillation remains a relevant stronger resource-frontier alternative, not a claim to exact reproduction without code qualification.

The existing development promotion requires at least +3 percentage points over the strongest fixed comparator, a one-sided question-cluster 95% lower endpoint above zero, mechanism controls, and protected-utility gates. The sealed stage requires the registered comparator set and simultaneous uncertainty. Keep the independent unit as question cluster, report all seeds and conditions, and charge inherited data and external controllers. Baseline-correct retention is an offline outcome analysis, never an oracle available to the deployed gate.

If DPO improves both action families similarly, treat it as better adaptation, not evidence for a new representation mechanism. If a learned action beats zero but loses the strongest adaptation, it is not SOTA. If feedback does not beat matched controls, the controller contribution fails. If only the short-budget point wins, do not claim the long-reasoning frontier. The broader goal still requires three architecture families, three constructs and proofs corresponding to the implemented object.

## 7. Coverage, uncertainty and handoff

The source increment used one targeted alphaXiv discovery pass followed by primary PDF/HTML, author-code and browser checks. Four consequential neighbors were examined; the remaining discovery hits are not claimed as read. Two independent source/code lanes returned evidence, and the coordinating agent spot-checked the highest-impact publication, objective and missing-source claims. Public reviews for RISER, PGS and OGLS were unlocated; the first-token forum was challenge-gated. No review text or author endorsement is invented.

Stop discovery here: the objective, novelty and affordability decisions are resolved enough to implement the next qualification. More broad searching will not substitute for a measured useful action. Reopen literature only for a specific consequential gap or a newly verified release.

Scientific maturity remains EVIDENCE_AVAILABLE; intended-contribution strength remains EVIDENCE_THIN, 15/36. Fair-strong, useful controller, strongest-comparator superiority, broad generality and ICLR submission readiness remain unpassed. This report is an evidence audit and prospective plan, not submission prose. AI assistance is recorded; a responsible human author, author approval and venue-specific disclosure remain unverified.
