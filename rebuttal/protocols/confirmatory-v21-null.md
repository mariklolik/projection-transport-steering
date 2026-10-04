# Confirmatory protocol v21: noise floor of selectivity

Frozen and pushed before any null rollout exists. Everything not stated here is
as in confirmatory-v19.md and confirmatory-v20-arc.md (readout M5, splits,
frozen decisions, bootstrap, Holm).

## Arms (`rebuttal/protocols/confirmatory-v21-null.jobs`), on each confirmatory split

| arm | tag | what it is |
|---|---|---|
| ungated null | `plain_null` | every question re-generated with an identity hook |
| gated null | `detnull_<cfg>` | the frozen PTS decision, identity action on flagged items |
| gated random | `detrand_<cfg>` | the frozen PTS decision, the selected action applied to a unit random direction (seed 0) at the selected dose |

`<cfg>` is the selected PTS configuration: `q60_alpha-0.375` (Gemma MMLU),
`q40_ablate` (Qwen MMLU), `q30_alpha-0.25` (Gemma ARC).

## Hypotheses

- H1: Sel(PTS) − Sel(gated null) > 0, paired bootstrap, one test per setting.
- H2: Sel(PTS) − Sel(gated random) > 0, one test per setting.
- Holm over the six tests.

## Reported for every Table 2 arm and the three null arms

Net change in the overconfident-wrong count, P(confident | wrong),
P(confident | right), ΔECE and ΔBrier of the stated confidence, each with a
paired 95% bootstrap interval. Selectivity net of the gated null is reported
next to raw selectivity.

## Pre-declared reading

A setting where H1 fails is reported as no effect beyond re-generation noise.
