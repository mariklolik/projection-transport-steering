# Steering literature corpus, 2022-present

Coverage date: 2026-09-06. Status: systematic identification pass reopened after a missed August revision of FishBack and closed at this date; this is a bounded corpus, not a claim that every database record or adjacent application has been found.

## Inclusion boundary

Include work that identifies or learns an internal representation in a language model and intervenes on intermediate activations or a compiled representation operator to control behavior. Also include benchmarks, negative results, robustness studies, causal or geometric foundations, and sparse-feature studies that can invalidate a steering claim.

Exclude input-only prompting, logit-only decoding, ordinary full-model fine-tuning, activation patching used only for interpretation, image-only steering, and model editing without a representation-steering question. Prompting, ReFT or LoRA, and decoding controls remain benchmark baselines even when excluded from the direct-method corpus.

Search combines forward and backward citation chasing from Latent Steering Vectors, ITI, ActAdd, RepE, and CAA; the 2025 RepE survey's query families; alphaXiv discovery; arXiv title search; and latest-paper sweeps through the coverage date. The 2025 survey states that its exhaustive portion ends on 2024-08-31, so 2024-09 onward requires an independent update.

Evidence levels are `deep` for inspected method, experiments, results, and limitations; `primary-abstract` for a verified primary abstract awaiting full extraction; and `candidate` for a discovery lead not yet admitted.

## Direct methods and core evaluations

### 2022

| ID | Work | Role | Evidence |
|---|---|---|---|
| 2205.05124 | Extracting Latent Steering Vectors from Pretrained Language Models | Optimized latent vectors and sentiment transfer; earliest included direct method | deep |

### 2023

| ID | Work | Role | Evidence |
|---|---|---|---|
| 2308.10248 | Steering Language Models With Activation Engineering | ActAdd from contrastive prompt pairs | deep |
| 2310.01405 | Representation Engineering: A Top-Down Approach to AI Transparency | General RepE pipeline and reading/control operators | deep |
| 2312.06681 | Steering Llama 2 via Contrastive Activation Addition | CAA across behavioral contrasts | deep |
| 2311.06668 | In-context Vectors | Task-vector steering for in-context learning | deep |
| 2310.15213 | Function Vectors in Large Language Models | Causal head selection and task-function vectors | deep |
| 2309.08600 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | SAE foundation and intervention evidence | deep |
| 2301.04709 | Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability | Exact and approximate abstraction through interchange interventions | deep |
| 2305.08809 | Boundless Distributed Alignment Search | Learned distributed causal subspaces | deep |
| 2311.03658 | The Linear Representation Hypothesis and the Geometry of Large Language Models | Counterfactual linear concept geometry | deep |

### 2024

| ID | Work | Role | Evidence |
|---|---|---|---|
| 2402.09631 | Representation Surgery: Theory and Practice of Affine Steering | MiMiC full-covariance affine transport | deep |
| 2403.05636 | Tuning-Free Accountable Intervention for LLM Deployment | CLEAR sparse metacognitive intervention | deep |
| 2403.05767 | Extending Activation Steering to Broad Skills and Multiple Behaviours | Broad and composed behavior vectors | deep |
| 2404.03592 | ReFT: Representation Finetuning for Language Models | Trained low-rank representation control | deep |
| 2405.15454 | Linearly Controlled Language Generation with Performative Guarantees | Learned linear controller with formal guarantees | deep |
| 2406.00034 | Adaptive Activation Steering | Adaptive truthfulness directions and intensities | deep |
| 2406.00045 | Personalized Steering via Bi-directional Preference Optimization | Output-optimized preference vector | deep |
| 2406.03631 | SteerFair / Discovering Bias in Latent Space | Unsupervised counterfactual option-bias removal across multimodal heads | deep |
| 2406.11717 | Refusal in Language Models Is Mediated by a Single Direction | Refusal direction and ablation/addition | deep |
| 2406.15518 | Steering Without Side Effects | KL-then-steer model adaptation | deep |
| 2407.12404 | Analyzing the Generalization and Reliability of Steering Vectors | In- and out-of-distribution reliability audit | deep |
| 2408.11491 | Nothing in Excess | Safety-conscious steering against over-refusal | deep |
| 2409.05907 | Programming Refusal with Conditional Activation Steering | CAST condition gate plus behavior action | deep |
| 2409.10053 | Householder Pseudo-Rotation | Norm-preserving rotational action | deep |
| 2409.14026 | Uncovering Latent Chain of Thought Vectors | Reasoning-style steering | deep |
| 2410.00153 | Beyond Single Concept Vector | Gaussian Concept Subspace | deep |
| 2410.04962 | Activation Scaling for Steering and Interpreting Language Models | Sparse multiplicative action | deep |
| 2410.12299 | Semantics-Adaptive Dynamic Intervention | SADI input-scaled masked action | deep |
| 2410.16314 | Steering LLMs using Conceptors | Ellipsoidal soft-projection operator and Boolean composition | deep |
| 2410.17245 | Towards Reliable Evaluation of Behavior Steering Interventions | Likelihood-aware standardized evaluation | deep |
| 2410.23054 | Controlling Language and Diffusion Models by Transporting Activations | AcT coordinate-wise affine OT | deep |
| 2411.02193 | Improving Steering Vectors by Targeting Sparse Autoencoder Features | SAE-targeted vector refinement | deep |
| 2411.02461 | Sparse Activation Control | Multi-property sparse head control | deep |
| 2411.09003 | Refusal in LLMs Is an Affine Function | Affine refusal geometry | deep |

### 2025

| ID | Work | Role | Evidence |
|---|---|---|---|
| 2501.09929 | Feature Guided Activation Additions | Optimized SAE-guided additions | deep |
| 2501.11036 | LF-Steering | SAE-feature control of paraphrase-level semantic consistency | deep |
| 2501.17148 | AxBench | 500-concept steering and detection benchmark | deep |
| 2502.02716 | A Unified Understanding and Evaluation of Steering Methods | Mean-difference, PCA, classifier, and affine comparison | deep |
| 2502.11356 | SAIF | SAE-based instruction-following control | deep |
| 2502.19649 | Taxonomy, Opportunities, and Challenges of Representation Engineering | Systematic survey and method taxonomy | deep |
| 2502.04878 | Sparse Autoencoders Do Not Find Canonical Units | SAE width and decomposition audit | deep |
| 2502.01179 | Joint Localization and Activation Editing | JoLA learned sparse additive and multiplicative head edits | deep |
| 2503.00177 | Steering Large Language Model Activations in Sparse Spaces | SAS contrastive steering in a pretrained SAE basis | deep |
| 2503.10679 | LinEAS | Sparse coordinated multi-layer affine action | deep |
| 2504.04635 | Steering off Course | Thirty-six-model reliability audit of DoLa, function vectors, and task vectors | deep |
| 2505.04260 | Steerable Chatbots | Preference-based personalization | deep |
| 2505.20809 | Reference-Free Preference Steering | RePS bidirectional preference optimization for rank-one and low-rank interventions | deep |
| 2505.20063 | SAEs Are Good for Steering if You Select the Right Features | SAE feature-selection evidence | deep |
| 2505.22637 | Understanding (Un)Reliability of Steering Vectors | Prompt, directional-agreement, and anti-steerability study | deep |
| 2506.06686 | Distribution-wise Representation Finetuning | Stochastic mean-and-scale representation interventions | deep |
| 2506.03292 | HyperSteer | Hypernetwork-generated scalable interventions | deep |
| 2506.07022 | AlphaSteer | Learned refusal steering under null-space utility constraints | deep |
| 2507.21509 | Persona Vectors | Monitoring and controlling personality traits | deep |
| 2507.08799 | KV Cache Steering | One-shot contrastive key/value-cache modification for reasoning behavior | deep |
| 2508.12535 | CorrSteer | Steering via correlated SAE features | deep |
| 2508.17621 | Steering When Necessary | Backtracking conditional steering | deep |
| 2509.22067 | The Rogue Scalpel | Random- and SAE-direction safety failures plus a universal steering attack | deep |
| 2510.26243 | Angular Steering | Two-dimensional rotational control and adaptive angle selection | deep |
| 2511.21399 | Steering Awareness | Internal detection of steering and intervention-induced safety regressions | deep |
| 2511.18284 | What Can We Actually Steer? | Fifty-behavior effectiveness and data-size study | deep |
| 2512.03661 | Dynamically Scaled Activation Steering | DSAS learned context-dependent strength | deep |
| 2512.07667 | Depth-Wise Activation Steering | Gaussian depth schedules for honesty | deep |

### 2026

| ID | Work | Role | Evidence |
|---|---|---|---|
| 2601.05679 | Do Sparse Autoencoders Identify and Control Reasoning Features? | Surface-cue falsification and negative steering evidence | deep |
| 2601.08441 | YaPO | Preference-optimized sparse steering in a pretrained SAE basis | deep |
| 2601.09269 | RISER | Learned routing over a library of reasoning vectors | deep |
| 2601.19375 | Selective Steering | Discriminative-layer selection and norm-preserving planar rotation | deep |
| 2602.04428 | Fine-Grained Activation Steering | AUSteer dimension-level adaptive scaling | deep |
| 2602.06801 | Non-Identifiability of Steering Directions | Local Jacobian equivalence classes and empirical direction non-uniqueness | deep |
| 2602.08169 | Spherical Steering | Norm-preserving geodesic action with a vMF confidence gate | deep |
| 2602.09870 | Steer2Edit | Steering-vector-guided sparse rank-one component editing | deep |
| 2602.16080 | GCM | Gradient causal mediation for site selection | deep |
| 2603.24543 | Analysing Safety Pitfalls in Activation Steering | Multi-model, multi-behavior jailbreak sensitivity audit | deep |
| 2603.06745 | DIRECTER | Plausibility-guided dynamic instruction-key scaling with sensitivity-ranked layers | deep |
| 2604.08169 | Activation Steering for Aligned Open-Ended Generation | Projection-aware coherent open generation | deep |
| 2604.08524 | What Drives Representation Steering? | Multi-token circuit analysis and refusal-vector sparsification | deep |
| 2604.19018 | A-LQR | Local linear dynamics and Riccati feedback | deep |
| 2604.24693 | CLAS | Learned block- and context-dependent coefficient on a fixed probe direction | deep |
| 2605.01167 | COAST | Covariance-metric minimum action on a sphere | deep |
| 2605.06225 | Memory Inception | Query-routed latent KV memory banks for persistent training-free guidance | deep |
| 2605.03907 | Steer Like the LLM | Prompt-steering replacement with token-specific coefficients | deep |
| 2605.05115 | Manifold Steering | Nonlinear causal-geometric paths | deep |
| 2605.05892 | FLAS | Concept-conditioned token-varying flow steering for unseen concepts | deep |
| 2605.30076 | UniSteer | Text-conditioned flow inversion across behaviors, fine-grained concepts, and compositional constraints | deep |
| 2605.10664 | GCAD | System-prompt attention-delta steering with per-token gates | deep |
| 2605.16362 | GRACE | Geometry-guided layer search and context-granularity diagnosis | deep |
| 2605.17231 | FishBack | Pullback-Fisher minimum-distortion steering with a matrix-free 8B approximation | deep |
| 2605.01844 | Cylindrical Representation Hypothesis | Sample-dependent axes, predictable normal planes, and unknown sensitive sectors | deep |
| 2605.24942 | Riemannian-Manifold Steering | Learned output-distance metric and geodesic activation replacement | deep |
| 2605.31183 | Steering LLMs? Actually, SAEs Can Outperform Simple Baselines | AxBench-focused SAE counterevidence | deep |
| 2606.07696 | Adversarial Robustness of Activation Steering | Input-perturbation and layer-selection failure | deep |
| 2606.11172 | Future Probe Controlled Generation | Future-behavior probes with sampled sentence-level control | deep |
| 2606.06735 | Angle-Norm Analysis of Activation Steering | Separates angular alignment from radial magnitude effects | deep |
| 2606.08365 | Forecasting Side Effects Before Intervention | SAE-geometry predictors of stability and collateral effects | deep |
| 2606.08454 | INNSTEER | Invertible nonlinear latent translation with exact reconstruction | deep |
| 2606.08682 | Activation Steering Induces Emergent Misalignment | Broad safety regression and phase transition | deep |
| 2606.09876 | PCHI | Same-forward wrong-confidence intervention | deep |
| 2606.30449 | Internal-State Probes Read Situation Not Action | Pre-action validity failures | deep |
| 2606.24952 | Perfect Detection, Failed Control | Cross-model detection-intervention dissociation | deep |
| 2607.05615 | Bernoulli Sparse Steering | Stochastic token-sparse SAE-feature intervention | deep |
| 2607.10517 | Conditional Optimal Bridge for Riemannian Activation Steering | COBRAS spherical Schrödinger bridge | deep |
| 2507.04742 | Activation Steering for Chain-of-Thought Compression | KL-bounded scale for reasoning-efficiency steering | deep |
| 2608.08829 | Deployable Per-Instance Multi-Layer Activation Steering | Prompt-conditioned subset ranking, Shapley structure, and adaptive stopping | deep |
| 2607.19806 | OPIUM | Downstream representation matching to reduce steering externalities | deep |
| 2607.25270 | Where Steering Signals Come From | Source-context and execution-boundary analysis | deep |
| 2608.02957 | Inverted Detection and Control in Steering Vectors | ITI-RRF downstream response sign correction | deep |
| 2608.05732 | CircuitSteer | Cross-layer SAE circuit action | deep |
| 2608.08383 | Safety Cost of Steering Vectors Is Separable and Reducible | Constrained safety-component ablation | deep |
| 2608.11227 | Forecasting Side Effects of Activation Steering | Sixty-seven-behavior collateral-effect prediction | deep |
| 2608.22985 | What Does Activation Steering Control? | Cross-encoding specificity audit | deep |
| 2608.24335 | SteerCheck | Matched-KL construction-null audit | deep |
| 2608.28233 | REINS | Gated SAE refusal control | deep |
| 2608.30197 | ALTSTEER | Staged refusal and orthogonal alternative-action control | deep |
| 2609.00597 | Topological Steering | Persistence-guided local direction aggregation | deep |
| 2609.01878 | GAPS | Static separability plus dynamic per-dimension posterior gates | deep |
| 2609.02089 | IDEEA | Input-dependent cluster-matched head directions | deep |
| 2609.03026 | ObserverBench | Downstream action-loss observer evaluation | deep |

## Required extraction queue

No `extraction-pending` or `primary-abstract` row remains. The frontier, earlier-corpus, late-frontier, FishBack correction, and claim-relative geometry passes are recorded in `frontier-extractions.md`, `corpus-extractions.md`, `late-frontier-extractions.md`, `fishback-novelty-audit-v18.md`, and `nearest-work-claim-relative-geometry-v1.md`. The search boundary is language-model internal-representation control and its direct validity audits; agent-memory applications, audio or vision steering, input-only controls, and generic model editing are excluded unless they test a steering assumption. Remaining work includes reproducibility: pin released code and model revisions, verify raw artifacts, and reproduce the strongest feasible comparators under each frozen pilot protocol.

## Rebuttal coverage

Primary OpenReview discussion is independently tracked because manuscript PDFs cannot substitute for reviewer objections or author responses. CAST, AcT, and SADI currently have secondary ReviewBench and rebuttal-analysis transport only; the primary forum is challenge-gated. No response is attributed to an author without a primary thread or a clearly labeled secondary summary. Accepted-paper revisions are compared to initial versions where review text is unavailable.
