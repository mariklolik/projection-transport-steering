# Frozen experiment protocol, version 0.1

Status: design freeze before code or compute. Numerical margins and exact dataset sizes are provisional until a development-only power analysis is recorded. Confirmatory examples may not be inspected for controller selection.

## Research question

Can a same-forward observer predict which available intervention, including no action, minimizes held-out downstream loss, and can a low-rank metric-constrained transport improve the target behavior more reliably than current steering methods under matched data, tuning, and inference budgets without exceeding a declared capability-loss margin?

## Split contract

Each model-behavior dataset is partitioned by immutable example ID into:

1. representation fit;
2. observer fit;
3. controller and hyperparameter development;
4. confirmatory in-distribution test;
5. confirmatory shifted test.

All five ID sets must be pairwise disjoint. Family-level templates, paraphrases, source questions, or generated variants must be grouped before splitting. The manifest records source revision, canonicalization, group key, hash, and reason for exclusion. Confirmatory labels and outputs remain sealed until the configuration registry is frozen.

## Model panel

The confirmatory panel must include three independently implemented current decoder families at comparable 4B-9B scale, plus Gemma-2 for backward comparison. Candidate checkpoints are Qwen3.5-9B, Gemma-3-4B-IT, and Llama-3.1-8B-Instruct. Exact immutable Hugging Face revisions will be frozen only after license, chat-template, hook-site, runtime, and local or remote availability verification. Qwen3.5 is an architecture stress case because its official model card describes a 32-layer hybrid of Gated DeltaNet and grouped-query attention with multi-token prediction, rather than a uniform attention-only stack.

No architecture-specific `model.model.layers` assumption is allowed in the controller. Architecture adapters must expose residual-stream sites, attention-head sites where applicable, tokenizer/chat-template behavior, and pass instrumentation through one tested contract.

The adapter contract must distinguish prefill, ordinary decode, cached decode, and speculative draft or verify execution. It must record whether an intervention mutates states that feed a KV cache, recurrent or linear-attention state, MoE routing, or multi-token prediction. Residual actions are the portable core; head-specific baselines are implemented only where their MHA, GQA, or other head semantics are well-defined. Pre-norm, post-norm, sandwich-norm, local/global attention, and hybrid recurrent blocks are separate architecture strata, not assumed-equivalent hook sites.

## Behavior panel

| Behavior | Target construct | Required off-target controls | Closest task-specific baseline |
|---|---|---|---|
| Overconfidence | Reduce confidence assigned to an incorrect answer while preserving confidence on correct answers | Accuracy, calibration on correct items, abstention quality, answer format | PCHI |
| Hallucination or abstention | Reduce unsupported factual assertions and improve selective answering | Supported-answer recall, verbosity, citation or entailment validity | Prompt and finetuned/ReFT control |
| Sycophancy or conditional refusal | Change behavior only under the intended condition | General helpfulness, benign compliance, category and paraphrase shift | CAST and SADI |

A behavior enters the confirmatory panel only if labels measure the construct independently of the intervention and if a held-out scorer audit reaches a predeclared reliability threshold.

## Methods and budget matching

Required controls:

- no intervention;
- system-prompt control;
- random direction and norm-matched random subspace;
- CAA or DiffMean;
- ITI or a localized linear-probe intervention;
- CAST;
- SADI;
- AcT;
- MiMiC;
- LinEAS;
- COAST;
- COBRAS or the strongest feasible nonlinear transport;
- LoReFT or ReFT-r1;
- PCHI for overconfidence;
- IDEEA as the strongest input-dependent direction-routing baseline;
- GAPS and DSAS as the strongest inspected per-dimension and per-token conditional baselines;
- AUSteer as the strongest inspected fine-grained coordinate controller;
- StTP or StMP as the strongest inspected projection-aware open-generation controller;
- FASB as a backtracking conditional comparator with its extra generated-token cost exposed;
- REINS-Gate for refusal, with harmful, refusal, other-safe, and collapse outcomes reported separately;
- Spherical Steering and Selective Steering as norm-preserving geometric controls;
- OPIUM and ALTSTEER for refusal externality reduction and desired-alternative generation;
- INNSTEER as the strongest feasible learned invertible nonlinear comparator;
- SAS or YaPO as sparse-space controls when the exact SAE and fit budget are reproducible;
- ITI-RRF to test and correct detection-control sign inversion;
- A-LQR if its released implementation and exact budget can be reproduced.

Every method receives the same representation-fit examples and the same development examples. Search spaces and maximum trials are frozen before evaluation. Results are reported both at matched total tuning compute and at each method's recommended configuration. Missing or incompatible baselines are labeled, not silently approximated.

## Variants under test

- fixed additive action;
- unconditional scalar transport;
- two-pass trace observer plus transport;
- same-forward observer available before the controlled layer;
- rank-1 and low-rank action subspaces;
- Euclidean and covariance/Fisher-local metrics;
- observational label observer and treatment-benefit observer;
- per-action loss observer with an explicit no-op action;
- open-loop action and, if justified, lightweight feedback action.
- exact norm-preserving, angular-only, radial-only, and angular-plus-radial actions;
- deterministic full-token and Bernoulli token-sparse actions at matched controlled-token budgets.

The target method is not named or claimed novel until ablations show which coupling is necessary.

## Primary estimands

For item `i`, let `Y_i(1)` and `Y_i(0)` be the paired target-behavior score with and without intervention, and let `C_i(1)` and `C_i(0)` be the paired capability score. The primary estimand is the mean paired target gain under the constraint that the lower confidence bound for capability change remains above a frozen non-inferiority margin.

The provisional capability margin is 1.5 percentage points for binary accuracy. It will be replaced only by a development-only power and practical-significance analysis recorded before unsealing. Continuous task metrics receive scale-specific margins.

Secondary estimands include calibration error with bias-corrected intervals, risk-coverage area, treatment-benefit AUROC, conditional precision and recall, off-target behavior change, and Pareto hypervolume over target gain, capability, and inference cost.

Observer selection uses downstream action loss under the frozen action set. AUROC and calibration remain diagnostics and cannot select the final policy alone. Report always-act, never-act, score-threshold, learned-policy, and oracle-action bounds on the same paired items. The observer must be available at an earlier execution point than every action it selects; otherwise the method is labeled two-pass.

## Representation and specificity controls

- Compare prompt-end, decision-boundary, generated-answer, mean-token, last-token, and tail-subtracted source representations on development data only.
- For encoded-answer tasks, remap semantic labels to at least three disjoint identifier alphabets and permute row order while freezing task semantics.
- Fit the operating intervention strength under a neutral-prompt KL budget before confirmatory evaluation.
- Compare against norm- and KL-matched isotropic, PCA-subspace, and sign-randomized null families.
- Require polarity mirroring, semantic open-generation checks, and a blinded scorer audit where the behavior permits them.
- Report the fraction of anti-steerable items and test sign calibration independently of discrimination; a high AUROC cannot determine action orientation.
- Measure detection-intervention angle and downstream functional overlap, but never use either as the sole policy selector.
- Train a lightweight intervention-awareness detector on development data and report its held-out transfer as an evaluation-channel threat; detection is not treated as resistance.
- Evaluate grouped paraphrases and semantics-preserving TextFooler-, TextBugger-, or BERT-Attack-style perturbations on the shifted test; layer and action selection remain frozen from development.
- A model-layer-behavior cell passes specificity only if target improvement and protected-tail non-inferiority survive a prespecified intersection-union gate and the construction-matched nulls do not explain the effect.

## Collateral and safety controls

Each primary intervention receives a frozen cross-behavior panel spanning helpfulness, refusal, toxicity, sycophancy, calibration, verbosity, instruction following, and task capability. Report the full effect matrix, multiplicity-adjusted intervals, worst protected-tail change, severe-event counts, and sign asymmetry. Safety evaluation includes ordinary harmful prompts, at least one static jailbreak family, at least one held-out adaptive attack family, and false-refusal evaluation on benign and borderline-safe prompts. Harmful generations are partitioned into harmful fulfillment, coherent safe refusal, other safe output, and collapse so that refusal quality cannot be replaced by generic degradation. A learned side-effect forecaster may prioritize development diagnostics but cannot replace direct confirmatory measurement. Constrained safety-component ablation is included as a post-hoc safety control when its code and budget are reproducible.

## Uncertainty and aggregation

- Use paired bootstrap resampling at the grouped example level.
- Stratify by dataset domain, difficulty, baseline correctness, baseline confidence, response length, and observer score bin.
- Report percentile and bias-corrected intervals when stable; otherwise report the failure and a justified alternative.
- Aggregate across model-behavior cells with a hierarchical model or a prespecified random-effects analysis; never treat overlapping subsets as independent seeds.
- Correct confirmatory family-wise testing with a prespecified procedure. Exploratory cells are visibly labeled.
- Report effect distributions and cell-level exceptions, not only an overall mean.

## Failure decomposition

For each method, record:

1. observer false positive and false negative;
2. oracle-observer performance with the learned action;
3. learned observer performance with an oracle-selected action or best valid action family;
4. action failure when intervention was indicated;
5. target improvement accompanied by capability damage;
6. out-of-support observer or transport state;
7. format, decoding, refusal, and scorer failures;
8. behavior and model-family strata where the effect reverses.
9. identifier-following or generic-opposition effects under cross-encoding and null controls;
10. collateral effects that exceed the mean or protected-tail budget.
11. detection-control dissociation, direction-equivalent but geometrically distinct actions, and intervention-awareness failures.

## Pass and resource accounting

Instrument and publish model forwards per answer, hooked layers, controlled tokens, generated tokens, observer parameters, action parameters, offline activation volume, optimizer steps, wall time, accelerator hours, peak device memory, throughput, and latency. A baseline trace followed by regeneration counts as two generation trajectories even when the vector action itself is `O(d)`.

## Sequential decision rule

Development proceeds in three gates:

1. synthetic and replay tests establish estimator correctness, ties, tails, and pass accounting;
2. one model-behavior development cell must beat strong controls and survive oracle decomposition;
3. the registered configuration runs once on the full sealed panel.

Failure at a gate triggers diagnosis and a new versioned protocol. Confirmatory results are never recycled as development data. State-of-the-art wording is permitted only for the exact panel, metric, and budget where the registered comparison supports it.
