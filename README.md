# Projection-Transport Steering

Minimal reproducibility package for **Projection-Transport Steering: Conditional Control of Reasoning Behavior in LLMs**.

- [Paper PDF](paper/paper.pdf)
- [LaTeX source](paper/paper.tex)
- [Reproducibility guide](REPRODUCIBILITY.md)
- [Post-review evidence](rebuttal/)

## Results

Projection-Transport Steering separates two decisions that fixed-vector methods conflate: when the model should be changed and how its activations should move. Across five resampled MMLU subsets, gated ablation reaches **+0.266 selectivity with no detectable accuracy change**. The fixed additive edit is essentially nonselective and lowers accuracy, showing that stronger global steering is not a substitute for targeted control.

The held-out confirmation reaches the same conclusion beyond Gemma. On OLMo-2/TruthfulQA, the detector-axis action improves calibrated truthfulness by **+0.137** over a norm-matched action, also improves correct-answer selection, and passes all seven predeclared construction controls.

The practical result is simple: detect whether intervention is warranted, then apply an input-dependent edit confined to the behavioral projection. Claims remain limited to the evaluated settings; the appendix and rebuttal packet preserve the full stress-test boundary.

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
