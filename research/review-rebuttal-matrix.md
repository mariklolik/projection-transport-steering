# Competitor review and rebuttal matrix

Primary OpenReview retrieval was blocked by an interactive verification challenge on 2026-09-05. The review text below is provisionally reconstructed from `Samarth0710/reviewbench`, a secondary transport of OpenReview content. Score transitions and response summaries come from `MlouisBE/iclr-rebuttal-analysis`; its response-cause field is machine-derived and is not treated as an author quote or primary rebuttal record.

## Source-paper review

The user-provided scope contains one review of Projection-Transport Steering with rating 8 and confidence 3. The reviewer considered the theory, five-subset MMLU experiments, controlled comparisons, and artifacts strong, but identified one deciding gap: validate the detection axis across more models and behaviors. The requested evidence is a targeted confirmatory experiment with uncertainty, stratified error analysis, explicit essential assumptions, a controlled boundary stress test, and claim-specific limitations. No forum ID or author-response body is available, so this records reviewer pressure rather than a rebuttal outcome.

The mandatory response is empirical, not rhetorical: at least three architecture families, at least three behavior constructs, action-loss rather than detection-only evaluation, sealed paired inference, stratification, and an assumption-breaking stress suite. The source result is a starting prior and cannot be reused as confirmatory evidence because its fitting and selection audit found split leakage.

| Work | OpenReview forum | Observed scores before to after | Repeated reviewer pressure | Provisional response evidence | Design implication |
|---|---|---|---|---|---|
| CAST | `Oi47wc10sm` | 8 to 8; 6 to 6; 6 to 8 | Oracle decomposition between condition failure and action failure; prompting controls; hard prompts; larger models; jailbreak and multi-turn tests | Secondary summary reports added F1 and quantitative tables, clarification that PCA is replaceable, and sharper condition/action duality | Predeclare oracle-observer, oracle-action, prompt, and hard-distribution controls; do not hide failures in an aggregate refusal score. |
| AcT | `l2zFn6TIQi` | 6 to 6; 8 to 8; 6 to 8; 6 to 8 | Multimodal or nonlinear distributions; representativeness of small samples; breadth and safety; out-of-distribution behavior; pooling/layer ablations; compute | Secondary summary reports added complexity and fusion discussion, Self-BLEU diversity, nonlinear-alternative results, and support ablation | Include nonlinear and multimodal stress tests, learning curves, support/OOD gates, diversity, and full offline plus online cost. |
| SADI | `8WQ7VTfPTl` | 6 to 6; 6 to 6; 3 to 6; 6 to 6; 8 to 8 | Why the method works; sensitivity to mask size and strength; inference overhead | Secondary summary reports ranges for mask size and strength, a small held-out grid, and measured inference overhead | Freeze hyperparameter budgets, publish sensitivity surfaces, and report intervention overhead under identical generation settings. |

## Evidence rule

No manuscript sentence may say “the authors rebutted” on the basis of the secondary machine summary. Until primary replies are available, wording is restricted to “a secondary analysis reports.” Review concerns may guide experiment design, but cannot be used as primary evidence for a competitor weakness.

## Cross-paper rebuttal lessons

1. Reviewers reward new evidence more than verbal clarification: missing baselines, stress tests, and cost accounting must be run before submission.
2. Conditional methods are expected to separate condition recognition from action effectiveness.
3. Transport methods are expected to test nonlinear, multimodal, small-sample, and out-of-support regimes.
4. Hyperparameter search must be visible and budget-matched; a best-layer or best-strength result is not tuning-free.
5. Practical efficiency includes all activation collection, selection, and additional model passes, not only the final vector update.
