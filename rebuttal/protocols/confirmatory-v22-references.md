# Confirmatory protocol v22: reference arms for every conditional method

Frozen and pushed before any rollout listed in `confirmatory-v22-references.jobs`
exists. Splits, frozen decisions, frozen actions, readout M5, bootstrap
(10^4 paired resamples) and Holm as in v19, v20 and v21.

## Conditional arms covered

For each confirmatory split (Gemma MMLU, Qwen MMLU, Gemma ARC): PGS post-hoc
(`det_`), PGS at the prompt (`detprompt_`), CAST prompt condition (`castdim_`),
CAST-style trace condition (`tuned_cast`), at their frozen configurations.
Sycophancy (Gemma confirm): the frozen gated deference edit (probe q30, dose -0.25).

## Reference arms, same decision and same flags

- no-edit null: `null-<tag>`, identity action on the flagged questions.
- random-edit nulls: `rand<s>-<tag>`, s = 0..9, the frozen action on a unit
  random direction (seed s) at the frozen dose.
- matched override: TPR - FPR at the same flags (computed, no rollout).
- sycophancy: ungated random directions `syco_rand<s>-0.25`, s = 0..9, composed
  with the frozen flags.

## Hypotheses

- H1 (12 tests): Sel(arm) - Sel(no-edit null of arm) > 0.
- H2 (12 tests): Sel(arm) - mean Sel of its 10 random-edit nulls > 0.
- Holm over the 24 one-sided tests.
- H3 (prediction, 13 arms incl. sycophancy): the sign of Sel(arm) - (TPR - FPR)
  predicted by Eq. 2 with ungated conversion rates agrees with the observed sign.
  Reported as a count of agreements.

## Secondary (reported for PGS post-hoc against CAST-style on Gemma)

Paired intervals for selectivity at confidence cutoffs 0.4 and 0.6, selectivity
over boxed traces only, and the change in AUROC of the stated confidence.
