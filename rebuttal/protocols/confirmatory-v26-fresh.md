# Confirmatory protocol v26: fresh free-text split, strongest conditional baselines

Frozen and pushed before any steered rollout on the split `confirm2`
(RealToxicityPrompts prompts 8000-11000 of the v23 shuffle, never used before;
only unsteered base rollouts exist). Judge, retention criterion and statistics
as in v23.

## Action family of PTS (tuning split, v23 decision: prompt probe at q50)
Shift (v23 grid), displacement of the quantile map onto the clean law shifted
by delta (monotone, an optimal-transport map; delta in 0..-0.5 of the norm),
and the shift scaled per token by a logistic token gate fitted on toxic against
clean continuation tokens of the extraction split. The rule (highest tuning
selectivity) picks per model:
- Gemma: prompt probe q50 + token-gated shift -1.0 (tuning 0.643)
- Qwen: prompt probe q50 + token-gated shift -2.0 (tuning 0.635)
- Mistral: prompt probe q50 + shift -0.75 (tuning 0.605)
Baselines tuned under the same rule: DSAS-style token-gated shift without the
prompt decision (6 doses; selected -2.0 in all three), Linear-AcT per-neuron
affine transport from toxic to clean token statistics (strength 0.25-1.5;
selected 1.5, 1.5, 1.0), global shift, prompting, CAST (v23 selections), and
the v23 PTS shift arm for continuity. Selections in
`confirmatory-v26-fresh.selection.json`.

## Hypotheses (one-sided paired bootstrap 10^4 over prompts; Holm over 16)
Per model m in {Gemma, Qwen, Mistral}:
H1m PTS > DSAS; H2m PTS > global shift; H3m PTS > prompting;
H4m PTS > refusal on its decision; H5m PTS > mean of 100 random directions
through its decision.
H6 (Gemma) monotone transport map (delta -0.375) > gated shift of equal mean
displacement on toxic tokens (-0.52).
Reported: Linear-AcT and CAST comparisons, no-edit null, per-arm removal and
retention.
