# Nearest-work audit for semantic-outcome steering, version 1

Coverage date: 2026-09-07. Status: bounded primary-paper and released-code audit before implementation or model output. This document constrains the candidate claim; it is not a universal priority proof.

## Search boundary

The search targeted activation steering constructed from multi-token continuation likelihood, contrastive sequence energy, multiple prompts or semantic views, robust or worst-case constraints, output-Fisher and KL trust regions, open-ended evaluation, future-outcome prediction, and concept-conditioned learned actions. Exact-title, equation, and paired-keyword searches covered arXiv, OpenReview discovery, official repositories, AxBench, FLAS, and Steer Like the LLM.

## Direct neighbors

| Work | Verified contribution | Boundary inherited by this project |
|---|---|---|
| [Towards Reliable Evaluation of Behavior Steering, arXiv:2410.17245](https://arxiv.org/abs/2410.17245) | Evaluates interventions with behavior-matching and behavior-mismatching multi-token continuations using model log likelihood in task-like contexts | Continuation likelihood is an evaluation object, not a new contribution here |
| [Activation Steering in Generative Settings via Contrastive Causal Mediation Analysis, OpenReview:bUXa74EiOL](https://openreview.net/forum?id=bUXa74EiOL) | Uses the contrastive likelihood of free-form response pairs to localize concept-sensitive attention heads, then applies mean-activation steering at selected heads | Contrastive sequence likelihood for generative localization is prior art; a residual-action optimizer must be distinguished from localization followed by CAA |
| [Activation Steering for Chain-of-Thought Compression, arXiv:2507.04742](https://arxiv.org/abs/2507.04742) and [OpenReview:d3sGvc0TLt](https://openreview.net/forum?id=d3sGvc0TLt) | The arXiv version extracts a mean-difference vector and calibrates its scale under an output-KL bound. The indexed OpenReview PDF additionally defines length-normalized contrastive sequence energy and a tokenwise KL trust region for optimizing a shared vector | Neither sequence-NLL optimization nor a KL-constrained activation vector can be claimed as new. The stronger indexed OpenReview formulation is treated as the comparator boundary even though its current source-code receipt is unavailable |
| [FishBack, arXiv:2605.17231](https://arxiv.org/abs/2605.17231) | Derives the minimum pullback-Fisher action for a declared output covector and evaluates matched-effect off-target KL | Minimum-metric steering for one output functional is prior art. Any distinction must come from the semantic uncertainty set and simultaneous constraints, not the quadratic program alone |
| [LF-Steering, arXiv:2501.11036](https://arxiv.org/abs/2501.11036) | Selects SAE features from paraphrase-consistency supervision and steers semantic consistency | Paraphrase-aware semantic control is prior art; the candidate cannot claim first semantic or consistency steering |
| [SADI, arXiv:2410.12299](https://arxiv.org/abs/2410.12299) | Uses input-semantic signals to construct a dynamic elementwise steering mask | Input-adaptive semantic steering is prior art; the candidate is a fixed concept action under outcome-view uncertainty |
| [Multi-property Steering with Dynamic Activation Composition, arXiv:2406.17563](https://arxiv.org/abs/2406.17563) | Composes several property vectors and modulates their intensity through generation | Multiple-property composition is prior art and is distinct from satisfying several witnesses of one semantic outcome |
| [Future Probe Controlled Generation, arXiv:2606.11172](https://arxiv.org/abs/2606.11172) | Predicts future behavior and selects among candidate sentence continuations | Future-outcome steering is prior art; a one-pass residual action cannot claim first future-aware control |
| [What Does Activation Steering Control?, arXiv:2608.22985](https://arxiv.org/abs/2608.22985) | Shows that answer-encoding and open-generation conclusions can diverge | Open generation is a necessary validity endpoint; teacher-forced likelihood alone cannot support semantic control |

## Performance frontier and released implementations

AxBench commit `41c8332543e5a631f9a8c0a9df38799893ace758` defines the common open-generation task, prompt split, and three-part judge. FLAS commit `720ef8a67697d9b94130b374b5b3a1522a782566` reports held-out HMean `1.015` on Gemma-2-2B and `1.113` on Gemma-2-9B. Steer Like the LLM commit `3d916c618d146c5d657f055e432a432b0fa493c6` reports `0.871` on Gemma-2-2B and `1.120` on Gemma-2-9B. These values are published frontier markers. A SOTA comparison still requires common data, prompts, judge version, tuning split, failures, and compute receipts.

AxBench silently maps malformed judge responses to zero. The common evaluation preserves that numerical convention for comparability but also records raw responses and parse failures; any missing row, differential parse failure, or incomplete method packet blocks promotion.

## Surviving candidate distinction

No inspected work was verified to solve one positive-definite minimum-cost residual action subject to a separate first-order progress constraint for every frozen, matched, multi-token semantic witness, then evaluate the same action on disjoint witnesses and open generations. The only admissible novelty hypothesis is therefore:

> A witness-robust minimum sequence-score action can improve worst-witness and open-generation concept control at matched distributional cost relative to pooled sequence-energy, Euclidean robust, DiffMean, prompt, and learned-flow baselines.

The mathematical combination is not itself sufficient for a contribution. Novelty remains `unverified` until a released-code recheck, direct CES comparison, ablations isolating the simultaneous constraints and metric, and a fair-strong empirical result pass.

## Prohibited claims

The candidate is not first sequence-likelihood steering, first KL-constrained steering, first Fisher steering, first semantic steering, first paraphrase-robust steering, first multi-property control, or first open-ended steering. A single-model premise result cannot support SOTA, modern-architecture breadth, global robustness, or finite-action guarantees.
