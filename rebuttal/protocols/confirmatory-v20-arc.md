# Confirmatory protocol v20: ARC-Challenge replication (gemma-2-2b-it)

Frozen before any steered rollout on the ARC splits was generated. Everything
not stated here is as in confirmatory-v19.md (readout M5, selection rule,
grids, tests, margin).

## Splits (4-option ARC-Challenge records, `label_pool.split_records`)

| split | n | source |
|---|---|---|
| arcx_detector | 1,012 | ARC train + validation after a seeded shuffle (seed 11), all but the first 400 |
| arcx_tuning | 400 | the first 400 of that shuffle |
| arcx_confirm | 1,165 | the full ARC-Challenge test split |

## What is reused and what is refitted

- Reused unchanged from the MMLU protocol: the action layer (14), the
  confidence direction, projection statistics, MiMiC and Linear-AcT moments,
  the CAST-style trace direction and its thresholds.
- Refitted on ARC: every decision. Trace, prefix and prompt probes and the CAST
  prompt condition are fitted on arcx_detector together with the MMLU detector
  pool and extraction split, with layer and regularisation chosen by AUROC on
  arcx_tuning; they are stored in `directions/gemma_arc/`.
- Every method's configuration is selected on arcx_tuning with the v19 rule
  and grids, and run once on arcx_confirm.

## Endpoints

Primary: pooled M5 selectivity of PTS post-hoc against the seven baselines,
paired bootstrap, Holm over seven. Secondary: single-pass variants, the
detect-and-override reference of each decision, calibration (ECE, Brier,
AUROC of the stated confidence).
