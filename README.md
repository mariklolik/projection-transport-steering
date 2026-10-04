# Projection-Transport Steering

Reproducibility package for **Projection-Transport Steering: Conditional Control of Reasoning Behavior in LLMs**.

- [Paper PDF](paper/iclr2027_upd_proposal.pdf)
- [LaTeX source](paper/iclr2027_upd_proposal.tex)
- [Reproducibility guide](REPRODUCIBILITY.md)

## Method

PTS splits an activation edit into an action and a decision. The action is the optimal-transport map of a projection onto a behavioral subspace. Shifts, ablation, clamps and quantile maps are all members of this family. The decision is a likelihood-ratio test on a learned detection statistic, read after the answer (post-hoc) or from the prompt (single pass).

The selectivity of the composed operator is `TPR * rho_O - FPR * rho_C`. This caps what any decision can buy from unsteered answers alone, and it predicts every gated configuration from one ungated run.

## Results

The frozen protocols are in `rebuttal/protocols/`.

- **Overconfidence, Gemma-2-2B, 3,000 held-out MMLU questions.**
  - Selectivity is 0.294 [0.256, 0.334], the highest of eight methods.
  - The share of overconfident-wrong answers falls by 15.0 points, and ECE falls from 0.354 to 0.255.
  - PTS scores above all 100 random directions sent through the same decision.
- **Replications.** Qwen2.5-7B reaches 0.172 and ARC-Challenge reaches 0.235.
- **Free-text toxicity, 3,000 held-out prompts per model.**
  - Single-pass PTS reaches 0.526 on Gemma and 0.534 on Qwen.
  - It scores above the global shift, prompting, and refusing on the same decision, and above CAST on Qwen.
- **Latency.** Single-pass PTS runs at 1.01x the unsteered wall clock.

`paper/recompute_from_records.py` recomputes the multiple-choice table cells from `results/release/records.csv.gz`.

## Repository map

| Path | Contents |
|---|---|
| general/steering.py | Projection, ablation, clamping, quantile transport, Gaussian transport, and Bures-Wasserstein operators |
| behaviour_specific/overconfidence/ | Direction fitting, confidence readouts, gating, baselines, evaluation, and analysis |
| models_specific/ | Gemma-2B and Gemma-2-9B adapters |
| results/ | Paper-level summaries and compressed per-question rollouts |
| behaviour_specific/overconfidence/directions/ | Released fitted directions and projection statistics |
| paper/ | ICLR submission source, generated tables, figure data, and compiled PDF |
| rebuttal/ | Frozen protocols, audit reports, and per-question post-review evidence |

## Quick verification

see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Main paper pipeline

    make extract
    make compare SEED=7
    make steer SEED=7

The five evaluation subsets use seeds 7, 11, 23, 31, and 47. Greedy decoding is deterministic; these seeds resample questions rather than decoding noise.

## Artifact integrity

Key hashes and the exact reviewer-to-change mapping are recorded in rebuttal/manifest.json and rebuttal/review.md. The public repository contains no model weights, credentials, caches, or private data.
