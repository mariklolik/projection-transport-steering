# Residual-flow nearest-work audit, version 1

Audit date: 2026-09-07. Status: bounded primary-source search complete; candidate novelty unverified.

## Candidate boundary

The internal working candidate is `ResidualFlowBack`. It does not replace or retrain the released FLAS flow. For each concept, it measures the released flow's finite construction score change on frozen matched multi-token outcomes, differentiates the same score with respect to an additive correction after the flow, estimates a trajectory-local neutral sequence-score metric, and fits the minimum-metric correction needed to lift weak construction outcomes to a declared within-concept target. The deployed action is one unchanged FLAS trajectory followed by one constant residual addition at the same layer.

## Direct prior work

| Work | Owned contribution | Boundary for this route |
|---|---|---|
| FLAS, arXiv:2605.05892 and released commit `720ef8a67697d9b94130b374b5b3a1522a782566` | Concept-conditioned, token-varying learned velocity field with three-step Euler integration; published Gemma-2 AxBench frontier | FLAS is the unchanged base and primary baseline. Learned flow, natural-language concept conditioning, curved trajectories, and the reported 2B/9B scores are not new |
| UniSteer, arXiv:2605.30076 | Text-conditioned activation distribution, partial source-flow inversion, target-flow regeneration, and compositional conditions across three target models | Flow inversion, universal text conditioning, multi-constraint flow steering, and activation-energy classification are prior art. Its AxBench evidence is Concept10 on Llama/Qwen, not the Gemma frontier |
| Contrastive Energy Steering, OpenReview `d3sGvc0TLt` | Length-normalized sequence energy optimized under a tokenwise KL trust region | Sequence-level semantic energy and KL-limited vector optimization cannot be claimed |
| FishBack, arXiv:2605.17231 | Minimum pullback-Fisher action for one target covector | Minimum metric action and downstream output-Fisher geometry cannot be claimed |
| COAST, arXiv:2605.01167 | Covariance-metric minimum action on a sphere | Metric-aware constrained activation movement is prior art |
| OPIUM, arXiv:2607.19806 | Downstream representation matching to reduce steering externalities | Output-preserving correction and externality reduction are not independently new |
| AlphaSteer, arXiv:2506.07022; PCHI, arXiv:2606.09876 | Null-space or functional preservation around a behavior intervention | Projection away from protected directions and same-forward functional control are prior art |
| Multi-property Dynamic Activation Composition, arXiv:2406.17563; instruction-vector composition, arXiv:2410.12877 | Composition of separately constructed steering actions | Adding a vector to another intervention is not a contribution |

## Literature correction

The previous corpus omitted UniSteer. The corpus, related-work matrix, and frontier extraction now include it. This omission narrows the candidate: universal or compositional flow steering is not available as novelty. The OpenReview FLAS forum remains challenge-gated; the accessible primary paper and released code are used, and no author rebuttal is inferred from unavailable discussion.

## Surviving hypothesis

The only candidate distinction is the conjunction of:

1. finite, per-witness residual deficits measured under an unchanged released flow;
2. additive score covectors differentiated at that flow trajectory rather than at the base model;
3. a trajectory-local neutral sequence-score metric for the correction;
4. a minimum correction with nonuniform witness deficits as its right-hand side;
5. direct unchanged-flow, pooled-residual, Euclidean-residual, shuffled-deficit, and no-correction controls.

This distinction is testable but not yet a contribution. If the residual correction does not beat both unchanged FLAS and the matched-cost pooled residual on fresh outcomes, the route closes. Even a local pass cannot support novelty or SOTA without direct CES, UniSteer, FLAS, and strongest AxBench frontier comparisons.
