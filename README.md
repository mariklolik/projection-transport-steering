# What the Decision Buys

Reproducibility package for **What the Decision Buys: An Exact Account of Conditional Activation Steering**.

- [Paper PDF](paper/iclr2027.pdf)
- [LaTeX source](paper/iclr2027.tex)
- [Reproducibility guide](REPRODUCIBILITY.md)

## Summary

A conditional activation edit makes two choices: which inputs to edit and how to edit them. Its selectivity equals the decision's error rates weighted by the rates at which the edit converts the flagged inputs. The paper uses this identity to predict gated edits from one ungated run, and to compare each conditional edit with three references built on the same decision: no edit, a random edit, and an override of the behavior's readout.

Probe-Gated Steering (PGS) puts a correctness probe in front of a projection edit. Under the frozen protocols in `rebuttal/protocols/` (confirmatory splits of 3,000 MMLU questions per model and 1,165 ARC-Challenge questions):

- Gemma-2-2B, MMLU: selectivity 0.294 [0.256, 0.334]. This is above all seven baselines after Holm adjustment, 0.245 above the no-edit null and 0.079 above the random-edit null, with ECE lower by 0.099.
- Qwen2.5-7B, MMLU: 0.172, which is 0.081 above the no-edit null. A calibration prompt reaches 0.212.
- Gemma-2-2B, ARC: 0.235, which is 0.219 above the no-edit null, with ECE lower by 0.060.

On Qwen, overriding the readout with the same probe reaches 0.433. `paper/recompute_from_records.py` recomputes every table cell from `results/release/records.csv.gz`.

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
