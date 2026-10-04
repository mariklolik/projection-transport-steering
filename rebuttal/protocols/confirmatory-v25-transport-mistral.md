# Confirmatory protocol v25: transport map on free text, third model family

Frozen and pushed before any rollout in `confirmatory-v25-transport-mistral.jobs`
exists. Setting, judge, retention criterion and statistics as in v23.

## Part 1: quantile transport map (Gemma-2-2B)
The action family of PTS now also contains the quantile map of the per-token
projection on the toxicity direction from the law over toxic continuations to
the law over clean continuations (extraction split, 41 quantiles,
`decisions/transport.pt`), applied as h + lambda (S(p) - p) under the frozen
prompt decision (probe, q50). Tuning selectivity: lambda = 1, 2, 3, 4, 6 gives
0.373, 0.577, 0.605, 0.586, 0.552; the tuned PTS shift gives 0.594; the rule
selects the transport map with lambda = 3. Its mean displacement on toxic
tokens equals a shift of -0.432 times the mean norm.
H1: Sel(transport map, lambda 3) > Sel(gated shift at -0.432), same decision.
On Qwen the rule keeps the shift (transport map at most 0.289 on tuning).

## Part 2: Mistral-7B-Instruct-v0.3
Directions, probes and tuning selections in `results/toxicity/mistral_7b_it/`.
Arms: global shift (-0.5), prompting, PTS prompt decision (probe q50, -0.75),
PTS post-hoc (q70, -1.0), CAST (q30, -0.5), no-edit null, 100 random
directions through the PTS decision; refusal on the PTS decision computed.
H2: PTS > global shift. H3: PTS > prompting. H4: PTS > refusal on the same
decision. H5: PTS > mean of its 100 random directions.
Holm over H1-H5, one-sided paired bootstrap 10^4.
