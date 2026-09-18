# Research reset: from predictive features to useful interventions

Projection-Transport Steering | Research decision report | 7 September 2026

## Executive decision

Restart the scientific argument, not the evidence archive. The previous experiments do not support a SOTA claim. They do establish several narrower positive results and identify repeated ways in which the research process selected attractive proxies before testing useful behavior. The next increment will test whether explicit intervention-response data can improve a repeatedly updated residual controller beyond a strong fixed adapter and existing reward-trained routing. It will not continue v20 or fit the paused natural-outcome observer.

This report is an audit and research plan, not submission prose. The ICLR paper remains blocked at the fair-strong evidence gate. A desired SOTA result cannot be promised; the benchmark, competitors, practical improvement target and stopping decisions can be fixed before new outcomes.

Three facts drive the reset:

- The strongest sealed learned-transport candidate, RCAPS v14, lost its Brier comparison in five of six model-behavior cells and tied in the sixth. LoReFT was the selected comparator in the five losses; DAPS in the tie. Positive accuracy changes relative to the unmodified model do not reverse this result.
- Fisher-protected FLAS reduced measured neutral KL, but did not establish a positive behavioral HMean effect. On 9B, a random basis improved the local judge's HMean more while worsening KL; removing the proposed rescaling also satisfied concept retention and reduced KL more. The proxy and the claimed mechanism did not align.
- The latest collection completed 1,024 trajectories, but the frozen observer source contract failed before training. This is not evidence that steering failed: no new steering policy was evaluated. It exposes an eligibility rule that confounds late-state support with successful early termination.

The goal's original 8/10, confidence-3 review is author-provided. It is not treated as a public acceptance decision or as validation of the source paper's claims.

## 1. Evidence reconstructed

The reset inventories all 176 pre-reset run-manifest records, including freezes, repairs, failed launches, analyses and actual experiments. This is not a count of 176 independent experiments. Each record has a line ordinal, source fields and SHA-256; the existing duplicate identifier `manifest-v36` is preserved and disambiguated by ordinal/hash.

The legacy archive contains 1,592 files. Ninety-seven research documents were restored into a separate temporary directory and verified against the archive's per-file hashes. Four decisive aggregate JSON files were separately restored and hash-verified: RCAPS confirmation, Fisher-FLAS 2B and 9B validation, and amortized FishBack development. Their published local summaries were checked against these aggregates. This reset did not rerun every archived GPU experiment or independently recompute every old bootstrap from per-example raw output.

The original source checkout remains read-only. Work is confined to `mariklolik/projection-transport-steering`. Original thresholds, adverse outputs and exposed splits remain unchanged. The latest fit collection was already running when the reset began; it finished as an archival packet. No old observer fit, calibration or validation was opened afterward.

### Verification levels used here

| Label | Meaning |
|---|---|
| Source-verified | Read in the original paper, code, official review or user-provided artifact; not necessarily independently reproduced. |
| Aggregate-verified | An archived analysis artifact was restored, hash-checked and its fields inspected. |
| Newly computed | Calculated in this reset from the available raw rows or exact formal construction. |
| Inference | A diagnosis consistent with evidence but not uniquely identified by it. |
| Proposed | A prospective experiment, threshold or implementation; not a measured result. |

## 2. The complete attempt history, without rewriting failures

The following table covers every scientific family and interface generation represented in the inventory. Early version 2 primarily established protocol/source infrastructure; it does not have a separate successful behavioral result to invent. Individual freezes and repairs remain available in the machine-readable inventory and append-only manifest.

| Route | Evidence and result | Correct interpretation |
|---|---|---|
| Original PTS | Strong selectivity and low operator-cost claims from Gemma/MMLU; source audit finds split, uncertainty, support and deployment-evidence gaps. | The geometric objective is not identical to behavioral control. Claims need independent reconstruction, not only a larger model. |
| v1, MMLU-Pro action-value routing | Best selected routing Brier 1.08863 versus additive 1.08859; paired difference about -0.00003, CI [-0.02077, 0.01985]. A hindsight action choice reaches about 0.88907. | An available oracle gap did not become deployable routing gain. Future-informed choice is not a controller. |
| v2-v4, TruthfulQA setup and transport | v3 gated MC2 0.60220 versus unconditional 0.62920 and spherical 0.72480. v4 quantile transport reaches 0.62969, below spherical. | Distribution-matching variants lost to the stronger geometry baseline under these settings. |
| v5, soft posterior | Near-no-op MC2 0.59462; the improved tuned spherical baseline is 0.73688. | Understeering and a weak prior comparator explained an attractive local story. |
| v6, matched extraction span | Re-extraction on 20,844 deployment-span tokens does not rescue the method: hard-posterior MC2 0.72295 versus spherical 0.73688. | Fixing observer/execution mismatch was necessary but insufficient. A calibrated-metric improvement cannot replace the primary outcome. |
| v7, threshold continuation | Lower threshold variants fail the declared gate. | Closed observer/threshold search, not permission for another grid. |
| v8, one-shot TruthfulQA | On 120 questions, MC1 ties; calibrated MC2 advantage +0.00133, CI [-0.00861, 0.01116]. | Insufficient evidence of superiority, not proof of exactly zero effect. |
| v9, CDAS and simpler DAPS | Causal-gradient variant loses to detector-only by -0.01404, adjusted CI [-0.02982, 0.00174]. DAPS beats spherical MC2 by +0.14865, CI [0.11376, 0.18247]. | A useful simple method survived; the added causal-gradient component did not. |
| v10, model replication | OLMo DAPS passes the joint comparison: MC2 +0.13719 and MC1 +0.10549. Mistral MC2 improves but MC1 non-inferiority is unresolved. Granite fails observer eligibility. | Genuine conditional positive evidence, not uniform model-family success. |
| v11, BBQ/ETHICS | Five of six observer cells fail. Mistral ETHICS improves average MC2 by +0.10730, but label-1 accuracy falls by -0.09333, CI [-0.12444, -0.07111]. | Pooled improvement concealed a strong polarity-specific harm. |
| v12, static prefix routing | Seven-action development looks favorable on BBQ, not ETHICS; family selection used upstream exposed information. | Useful development, not an independent confirmation of a selected family. |
| v13, nested 25-action routing | More complete nested selection preserves BBQ promise; ETHICS and all three FaithEval cells fall back to DAPS. Variable-option scoring required a cached-data repair. | Static state features did not reliably predict action benefit across behaviors. The scoring defect was separate from this result. |
| v14, RCAPS development | Six strong-baseline development cells select LoReFT in five and DAPS in one. | The comparison frontier changed materially when trainable baselines entered. |
| v14, sealed RCAPS | Five negative Brier contrasts, one exact tie; all six protected-stratum gates fail. Fixed-panel mean contrast -0.29341. | The proposed transport/routing method failed its registered strong-comparator claim. |
| v15, larger static action family | Of six cells, five prefer fixed fallback; the routed Granite cell has adverse regret. Hindsight gaps remain 0.08219-0.28211. | More actions and richer prefix routing did not solve action-benefit prediction. Close that family. |
| v16, Fisher-FLAS 2B | Neutral KL reduction 0.45253, CI [0.39288, 0.50994]; proxy HMean change +0.07969, CI [-0.04531, 0.22031]. | A collateral-distribution effect was measured; behavioral superiority was not. |
| v16, frozen 9B replication | KL reduction 0.15424, CI [0.13494, 0.17344]; HMean interval includes zero. Progress error 0.002614 exceeds 0.002; no-rescale control passes retention and improves KL more. | Partial endpoint replication, not replication of the complete claimed method. |
| v17, strength review | Fisher mechanism novelty and behavioral advancement do not pass. | Correctly stops SOTA promotion; does not erase the KL observation. |
| v18, amortized FishBack | Ranks 8/16/32/64 recover -15.3%/-34.1%/11.0%/53.6% of exact-reference log-KL gain, below the 80% target. Rank 64 also fails cost; 9.72 H100-hours. | A real accuracy-cost frontier defeated the declared amortization claim. |
| v19, numerical premise | One ill-conditioned matrix fails the 64-product float32 certificate; a different precision/iteration budget works. | Adverse numerical evidence, not a language-model behavior experiment. |
| v20, certified solver | Only two of eight shards finish. One candidate fails certification; five other shards hit a control-serialization defect. | Mixed numerical and implementation failures. Permanently closed by the user; no cap, certificate, regularizer or shard rescue. |
| InvariantBack | 27/27 robust QPs are feasible, but mapping eligibility fails one of 90 cells. | The per-query eligibility rule closes the route before a semantic-control test. It does not refute robust optimization in general. |
| CacheBack | 72 development cells and 3,888 evaluations complete; strong reachable-subset KL effects, but common-dose coverage reaches at most 62.5% versus required 80%. | Local cache-mediated mechanism survives; the powered full-population claim fails coverage. Eight workers with excessive CPU threads were separately corrected. |
| SemanticOutcomeBack | Candidate beats pooled control in only 3, 5, 5 and 7 of 12 concepts across four factors; loses the pooled mean throughout. | The metric helps relative to Euclidean control; the additional robust-witness contribution is not supported. |
| ResidualFlowBack | Pooled correction wins 10/12 concepts at every target quantile; candidate's low-quantile contrast -0.00855, CI [-0.01673, 0.00122]. | The construction did not target actual residual failure: all 96 original FLAS witness changes were already positive. |
| Flow-step support | Equal-time late support beats full FLAS efficacy by +0.07779 but raises neutral KL by 35% on average and loses to early support in 11/12 concepts. | An efficacy-damage trade-off, not joint improvement. Shapley allocation is not a standalone causal identification argument. |
| Proxy-validity screen | All 14 generations exist; only 13 judge rows and 40/42 rating dimensions parse. | Evaluator failure before proxy correlation was estimated. It does not establish that likelihood and behavior are uncorrelated. |
| AxBench rule-evaluator alternative | Source rules do not implement the general semantic C/I/F construct; additional source problems exist. | A convenient executable score cannot substitute for the intended behavioral endpoint. |
| LRS reproduction audit | Syntax failure, absent weights and training/evaluation/selection issues in the inspected release. | Exact headline reproduction is unavailable here; a labeled controlled rebuild remains a distinct experiment. |
| Outcome interface v1-v2 | v1: 64/64 cap hits, zero accepted labels. v2: six of eight cap hits; two correct boxed outputs rejected by the custom grammar. | Both model budget and scorer design matter. Neither run measures a steering effect. |
| Outcome interface v3-v4 | v3: three accepted parses, three cap hits. v4: all eight terminate, only four parse because conventional punctuation is rejected. | Repeated bespoke parser versions consumed iterations without testing the scientific hypothesis. |
| Outcome interface v5 | Standard Math-Verify yields 8/8 Stage-T and 64/64 Stage-C parses; Stage C has 16 correct and 48 incorrect. | A narrow interface success, not proof that every later source packet is eligible. |
| Observer basis and fit | First basis attempt has a replay-index bug; repaired attempt passes. Fit: 201/1,024 correct, 25 parse failures, 23 cap hits and no successful trajectory beyond token 1,536. | Source contract closes observer-v1 before fitting. Future designs must model termination and include unsuccessful outputs. |

### The strongest adverse result in detail

RCAPS confirmation uses the Brier-gain sign convention `loss(strong comparator) - loss(candidate)`, so positive favors the candidate. The six values are:

| Cell | Paired Brier gain | Adjusted interval |
|---|---:|---|
| BBQ / Mistral | -0.16698 | [-0.21162, -0.12831] |
| BBQ / OLMo-2 | -0.25691 | [-0.31727, -0.20163] |
| BBQ / Granite | -0.54224 | [-0.60348, -0.48518] |
| FaithEval / Mistral | -0.25514 | [-0.34196, -0.17977] |
| FaithEval / OLMo-2 | 0.00000 | [0.00000, 0.00000] |
| FaithEval / Granite | -0.53919 | [-0.66127, -0.42191] |

These are aggregate-verified results with paired cluster BCa intervals and two-sided alpha 0.05/6, controlling the six-comparison family at 5%. The candidate's positive accuracy changes in the same aggregate are versus the unsteered model, not proof of superiority to these Brier comparators. Changing the primary metric after seeing the table would change the question, not rescue the confirmation.

### What the latest 1,024 trajectories add

The unchanged analyzer verified eight complete fit shards and finite remote traces. Newly computed question counts with 0/1/2/3/4 successes across four rollouts are 170/31/18/14/23. The all-output descriptive success rate is 19.629%; an exploratory question-cluster bootstrap gives [15.625%, 23.730%]. This concerns exposed source data, not a new method.

The late-prefix population is selected by survival: 840 trajectories remain after 512 tokens, 249 after 1,024 and only 46 after 1,536. Those 46 all fail eventually; every successful trace terminates by 1,464 tokens. Requiring positive labels at every fixed late position made the old observer route ineligible. It is not a reason to discard successful early termination or to relax the already frozen rule retrospectively.

The collection used 3.5833 summed H100-hours and 29.419 minutes for its longest worker. It generated 845,606 main tokens. The 91.35% shard-balance ratio is not a utilization measurement; the 14.38 GiB peak allocation is not proof that tensor cores were idle. A faster new implementation needs measured batching and queue efficiency, not simply eight occupied devices.

## 3. The source theory needs a narrower contract

The original paper optimizes transport of a selected activation projection. Its strongest behavioral conclusions need additional assumptions. Three exact reasoning checks make the missing links concrete.

**Equal projection distributions do not imply equal intervention effects.** Consider two groups with the same projection `p=0` and an unedited state coordinate `z` distinguishing them. Let terminal correctness be `Y(p,z)=1` when either `z=-1` or `p>0`. At baseline, the `z=+1` group is wrong and the `z=-1` group is correct. The same additive change `p <- p+1` corrects every wrong case and retains every correct case. If both groups are initially confident, behavioral selectivity is maximal despite identical projection laws. Equal laws constrain decisions based only on that projection; they do not constrain a downstream response that also depends on other state. The original additive-selectivity corollary therefore needs an explicit common-response/sufficiency assumption.

**Raw AUROC is not the ROC-hull area in the theorem.** Let positive scores be +1 or -1 with equal probability and all negative scores be 0. Raw AUROC is 0.5, but a threshold between 0 and 1 gives Youden's J=0.5. Thus `J <= 2*raw_AUC-1` fails. The concave hull has area 0.75, for which the stated hull inequality holds. An empirical raw AUROC near 0.77 cannot automatically become a behavioral ceiling of 0.538. Even the valid hull inequality bounds a gate statistic, not the utility of the downstream action.

**Small Cohen's d is not a separability bound.** Two symmetric distributions can have the same mean but disjoint absolute-value supports, for example equal mixtures on {-1,+1} and {-2,+2}. Their mean difference is zero while a nonlinear statistic separates them perfectly. A measured d=0.049 alone cannot establish one-dimensional impossibility without a distributional family assumption.

The scalar monotone-transport result remains useful under appropriate atomlessness and finite-moment assumptions. Multivariate Brenier uniqueness requires stronger regularity than atomlessness alone. Real-valued hidden states do not by themselves imply an atomless sampling law: deterministic activations of discrete token sequences can be atomic. Conditional transport also optimizes a fixed gated population and specified projection target; it does not jointly optimize behavioral utility or select a correct gate population.

Other source-audited gaps remain: fitting `not-OCW` is not the same as fitting the theorem's confident-right population; a finite knot map does not inherit an unqualified empirical-support guarantee if its implementation extrapolates; local Fisher curvature does not globally bound finite generation changes; and a trace-level gate requiring an initial generation cannot inherit the local operator's no-extra-forward serving claim. These are mathematical and implementation boundaries, not requests for cosmetic theorem expansion.

Keep corrected results in an appendix. The empirical paper must be carried by useful controlled behavior, not by an optimality theorem for a surrogate that strong baselines outperform.

## 4. What the deeper literature reading changes

The primary-source companion records mechanisms, information access, datasets, budgets, code availability and limitations for LRH, causal abstraction, cylindrical/manifold geometry, FishBack, COBRAS, FLAS, UniSteer, CLAS, DSAS, RISER, ReFT, ObserverBench, FPCG, PUM, ACTS and LRS. The existing historical corpus supplies the 2022-2025 foundations; the fresh search is bounded, not claimed exhaustive.

The most consequential conclusion is a shrinking novelty boundary. LRH already distinguishes measurement and intervention representations. ObserverBench already distinguishes prediction from useful decisions. RISER already learns latent routing from successful interventions and terminal reward. DSAS and CLAS already learn dynamic coefficients. FPCG already forecasts future behavior and controls sampled continuations. ACTS already learns repeated text-level control under a budget. A new title for one of these ideas will not supply a contribution. [LRH](https://arxiv.org/abs/2311.03658), [ObserverBench](https://arxiv.org/abs/2609.03026), [RISER](https://arxiv.org/abs/2601.09269), [FPCG](https://arxiv.org/abs/2606.11172)

The remaining question is narrower: can supervision from the actual response to finite activation actions make repeated low-cost control more useful than a strong adapter, a prompt-only reward-trained router, a natural-outcome gradient and matched-budget search? This is a plausible experimental direction, not verified priority for a new learning paradigm. If a simple matched baseline performs as well, the added mechanism is unnecessary.

Public CAST and AcT reviews reinforce concrete methodological lessons. CAST reviewers requested a detector/action error decomposition; marginal trigger and refusal rates are insufficient for that decomposition. AcT reviewers asked about nonlinear maps, distribution support and total compute. Author replies reported overfitting/optimization difficulties with richer maps and separated extraction from serving costs. These are specific open questions to operationalize, not reviewer endorsements to borrow. [CAST discussion](https://openreview.net/forum?id=Oi47wc10sm), [AcT discussion](https://openreview.net/forum?id=l2zFn6TIQi)

Several external headlines are incomparable by construction. FLAS's HMean uses three semantic-quality ratings, UniSteer's rule success measures a different construct, and ACTS changes both token consumption and accuracy. FPCG generates multiple candidate sentences, so a one-pass method cannot claim superior efficiency merely by counting its own dot products. These methods belong on common resource-quality frontiers with explicit data and evaluator provenance, not in a single cross-paper score table.

## 5. Failure mechanisms, not a single failure story

| Mechanism | Evidence | Design consequence |
|---|---|---|
| The action is weaker than a feasible baseline | RCAPS versus LoReFT; robust witnesses versus pooled control | Admit strong trainable/simple baselines before spending on an elaborate operator. |
| The feature predicts the wrong object | Repeated static-routing failures; future-conditioned oracle gaps | Measure consequences of candidate actions, not only natural-state labels. |
| The optimized proxy is not the behavioral endpoint | Fisher KL versus HMean; FLAS witnesses already improving | Put executed task success before geometric refinement. Audit proxy ranking rather than assuming it. |
| Coverage or support is missing | CacheBack common reachability; late observer classes | Report the full population and conditional subset separately. Never turn a selected subset into a global claim. |
| A group average hides harm | ETHICS polarity reversal; RCAPS protected strata | Preserve subgroup tables and prospective non-inferiority margins. |
| Infrastructure prevents the scientific test | v20 control serialization; replay suffix; judge/parser failures | Recover engineering errors under preserved lineage. Do not describe unrun scientific tests as negative evidence. |
| The claimed idea already exists | FishBack, RISER, LRS, dynamic gates, future probes | Define a result-relative contribution and compare against the closest mechanism. |

The recurring process error was spending the next expensive run to make the current mathematical object work. The next run should instead distinguish why useful behavior does or does not improve. A valid failure is valuable only if it changes the next decision; dozens of finely versioned gates are not automatically fast research.

## 6. Chosen next increment

Use a learned low-rank residual adapter as the reference action. At a live prefix, allow only three finite-duration strengths: off, half-reference and reference. Collect paired continuations from the identical prefix/cache with isolated randomness, then learn which action improves final exact correctness. The deployed policy updates every 64 tokens using only current activation coordinates, recent state summary and remaining budget.

This changes both the measured quantity and deployment information relative to the closed static-prefix and natural-outcome-observer routes. It does not assert that a finite-dimensional state summary is Markov or that the action library spans every useful intervention. The full prefix/cache is the causal state; the controller's compact observation may still be inadequate.

### Decision sequence

| Stage | Question answered | Stop or promote |
|---|---|---|
| Standard scorer and batched branch smoke | Are outputs, cache branches, randomness and costs measured correctly? | Technical failures block results; no behavioral claim. |
| Strong fixed-adapter comparison | Is there a useful action worth controlling? | Stop if no feasible nonzero action improves the designated selection objective. |
| Same-prefix response pilot | Do the allowed actions offer learnable utility differences? | Stop a demonstrably unpromising family; use only the predeclared larger packet for uncertainty. |
| Held-out full-rollout development | Does a deployable repeatedly updated controller realize improvement? | Require at least +3 accuracy points, a positive lower confidence endpoint and attribution/protection gates. |
| Sealed common-protocol comparison | Does the advantage survive strong competitors and two model families? | Only this can earn registered reasoning-steering SOTA. |
| Orthogonal behavior replication | Does the procedure control semantic behavior while retaining quality? | Required but not sufficient: the full-paper three-family/three-construct breadth gates remain. |

The primary backbone is Qwen3-8B with thinking enabled; DeepSeek-R1-Distill-Llama-8B is the planned independent-family replication. MATH supplies the powered exact-answer endpoint after authoritative split recovery and overlap removal. GSM8K is separate transfer evidence; the small AIME yearly sets are stress tests, not a substitute for thousands of independent questions. The plan explicitly includes an 8,192-token budget and a later native-long-budget frontier; it does not select a cap after seeing which method wins.

The key ablations are matched-capacity natural-outcome versus intervention-response supervision, prompt-only versus repeated control, current-state versus recent-history observation, and random/shuffled policies matched for action occupancy. Additional geometry is postponed unless it explains a measured failure. No human labeling, new SAE training or bespoke judge is needed for the first exact-answer experiment.

## 7. Strict SOTA and statistical contract

SOTA is scoped to a registered base-model, dataset, information and resource envelope. Mandatory comparators include prompting, CAA, adaptive coefficients, LoReFT, LoRA, a task-reward-trained prompt router, natural-outcome gradient steering, and the strongest feasible search/control alternatives. An unavailable public implementation is a documented gap, not a defeated method. Rebuilds retain `-like` labels.

Keep two tracks: common-data/matched-budget methods, and best available externally pretrained methods with their additional data exposed. Trainable parameter count, action locations, tuning runs, source collection, rejected candidates, inference search and controller computation all count. The baseline receives the same total envelope as the candidate, not only the cost of its final fit.

The primary endpoint counts every assigned model output. Missing/invalid model answers and cap exhaustion receive zero utility. Artifact corruption and evaluator exceptions are technical errors with separate recovery records. A terminated trajectory is an absorbing state, not a reason to fabricate late observations. The old experiments remain evaluated by their original rules.

Uncertainty is clustered by independent problem, with repeats retained inside the cluster. Simultaneous intervals cover the predeclared primary competitor contrasts; population claims across all architectures are not inferred from two fixed models. Non-inferiority uses an explicit 2-point margin, not absence of a significant decline. The full protocol lists protection and diagnostic strata.

As a planning calculation, a 3-point paired accuracy effect with discordance 0.25 needs roughly 2,178 independent questions at 80% power for one two-sided comparison, and about 2,900 for three Bonferroni-adjusted contrasts. This is a design approximation, not a completed power analysis for a measured new method. Actual development variance and admitted comparison count determine the final sealed allocation. Repeated sampling of 30 AIME questions cannot create this number of independent questions.

## 8. Faster iteration and GPU use

The most useful speedup is eliminating runs that cannot decide the hypothesis. The previous source pipeline already provides stable IDs, deterministic seeding, model-resident workers, action hooks and receipt validation. Reuse these and upstream ReFT; do not write another transport solver, parser family or generic experiment system.

The new runtime must benchmark microbatches 1/4/8, shared prefills and isolated branches before scientific selection. Store activations at decision points instead of every token. Keep eight resident workers, bound CPU threads, validate completed immutable chunks concurrently, and measure useful completed rows per GPU-hour. Smaller memory allocation alone is not a utilization result. Padding, dtype, attention kernels and per-row sampling are part of the frozen protocol; throughput changes cannot silently change an experiment midway.

Initial maximum allocations are 1 H100-hour for runtime conformance, 8 for fixed-baseline fitting/tuning, 12 for the response pilot, 36 for the remaining response packet/head fits and 16 for full-rollout development. These are ceilings, not predicted runtimes or permission to report an incomplete packet. The next stage is launched only with a measured cost projection; a stage that cannot fit its declared packet is budget-incomplete, not a scientific rejection.

Run cheap CPU aggregation while GPUs are busy. Stop queued jobs for a rejected candidate promptly. Do not keep GPUs occupied with old routes merely to report utilization. Do not download multi-gigabyte traces when small verified result records suffice.

## 9. Deliverables, boundaries and next action

Completed by this reset: immutable attempt inventory; restored/hash-verified historical evidence; deeper primary-source and competitor-review synthesis; source-theory counterexamples; latest source-packet analysis; one selected research increment; SOTA/statistical/resource definitions. The previous observer test is preserved outside the active test suite. The original paper and historical outcomes are unchanged.

Not completed: a new controller implementation, a new behavioral improvement, a sealed SOTA benchmark run, independent final-paper review or ICLR submission compliance. The reference adapter, source split/overlap manifest, generalized math-scoring conformance, branch runtime and replication-model receipt are the next unpassed dependencies. None is disguised as an external VPN blocker: SSH/GPU access worked during this reset.

Immediate execution order is artifact pinning and baseline/runtime conformance, then the finite response pilot. No further broad literature sweep is needed before that packet unless a newly found nearest method changes the registered comparison. If the candidate fails, its exact action/information/learning defect determines the next research increment; the old cap/certificate/proxy routes stay closed.

### Evidence map

| Artifact | Role |
|---|---|
| `research/attempt-inventory-reset-v1.json` | All 176 pre-reset manifest records with ordinals and hashes |
| `research/legacy-lineage-index.md` | Legacy archive location and historical document map |
| `research/source-audit.md` and `research/claim-ledger.md` | Existing source defects and evidence-limited claims |
| `research/literature-reset-v1.md` | Paper/code/review locators and exact novelty boundaries |
| `research/outcome-score-observer-fit-audit-v1.md` | Latest basis/fit packet, counts, costs and closure |
| `research/benchmark-reset-v1.md` | Proposed action-response experiment and promotion contract |
| `research/research-reset-plan-v1.md` | Persistent execution status; unresolved prerequisites |
| `research/research-reset-receipt-v1.json` | Input/output hashes, skill gates and verification boundary |

Source manuscript PDF SHA-256: `ef8d151481a1f4bf5bda58b1244c56ca492dcf2ae260f2b61fbefc76b28a54f2`. Legacy archive SHA-256: `30b445bcb3f27dd9bcaaf2a3ad62f1ac0de14fb0c423d0ca64836c92337831e8`. Pre-reset run-manifest SHA-256: `5f8dd9285c3626e398169a3aceb3dc5be9f9ba69f9fe36f075e841b984973e92`.

AI assistance: Codex performed source retrieval, structured analysis, synthesis and artifact preparation. The responsible human authors' approval and venue-specific disclosure remain unverified. This document records recommendations and evidence boundaries, not an authorship or acceptance claim.
