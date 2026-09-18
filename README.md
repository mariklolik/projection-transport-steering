# Projection-Transport Steering

Minimal reproducibility package for **Projection-Transport Steering: Conditional Control of Reasoning Behavior in LLMs**.

- [Paper PDF](paper/paper.pdf)
- [LaTeX source](paper/paper.tex)
- [Reproducibility guide](REPRODUCIBILITY.md)
- [Post-review evidence](rebuttal/)

## Result

Projection-Transport Steering rewrites only a low-dimensional behavioral projection and composes the action with a trace-level detector. In the main Gemma/MMLU evaluation, the gated action improves the joint selectivity/accuracy trade-off over additive steering, prompting, CAST-style steering, Linear-AcT, and MiMiC under the same records and readouts.

The post-review held-out experiment extends the detector-axis result to OLMo-2-7B on 237 TruthfulQA questions. Against a norm-matched spherical action, calibrated MC2 improves by +0.137 with adjusted 95% CI [+0.105, +0.168], and MC1 improves by +0.105 with adjusted 95% CI [+0.025, +0.177]. All seven pre-frozen construction and specificity controls pass.

The paper limits the claim to the evaluated settings. Full stress-test outcomes remain available in the appendix and evidence packet.

## Repository map

| Path | Contents |
|---|---|
| general/steering.py | Projection, ablation, clamping, quantile transport, Gaussian transport, and Bures-Wasserstein operators |
| behaviour_specific/overconfidence/ | Direction fitting, confidence readouts, gating, baselines, evaluation, and analysis |
| models_specific/ | Gemma-2B and Gemma-2-9B adapters |
| results/ | Paper-level summaries and compressed per-question rollouts |
| behaviour_specific/overconfidence/directions/ | Released fitted directions and projection statistics |
| paper/ | Complete paper source, generated tables, figures, and compiled PDF |
| rebuttal/ | Frozen protocols, audit reports, and per-question post-review evidence |

## Quick verification

    uv sync
    make smoke
    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex

The smoke target imports each core module and runs embedded numerical checks without loading model weights. Full experiments require gated Hugging Face model access and a CUDA device; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Main paper pipeline

    make extract
    make compare SEED=7
    make steer SEED=7

The five evaluation subsets use seeds 7, 11, 23, 31, and 47. Greedy decoding is deterministic; these seeds resample questions rather than decoding noise.

## Artifact integrity

Key hashes and the exact reviewer-to-change mapping are recorded in rebuttal/manifest.json and rebuttal/review.md. The public repository contains no model weights, credentials, caches, or private data.
