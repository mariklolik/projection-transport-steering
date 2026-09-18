# Candidate method and hypotheses, version 10

Status: frozen after Qwen3 component isolation and before any target-repository Mistral, OLMo-2, or Granite model output.

## Deployment-aligned posterior spherical steering

For every representation-fit candidate answer, collect hidden states at the exact prompted causal positions whose logits score the answer. Average those states within each answer and fit unit detector direction `b` as truthful mean minus false mean. On all causal positions, fit balanced L2 logistic posterior

`p_0(c)=sigmoid(a c+d)`, where `c=b^T h/||h||`.

Each answer has equal total fit weight before truthful/false class balancing; weights are rescaled to mean one and logistic `C=1`. At deployment, if `p_0(c)>=0.5`, rotate `h` along its fixed-azimuth shortest geodesic toward `b` with strength `alpha=1`; otherwise leave it unchanged exactly. Norm is preserved.

This differs from Spherical Steering by replacing antipodal-vMF confidence with an empirical posterior fitted on the exact deployment population, and from DSAS by using answer-balanced causal positions plus a hard norm-preserving geodesic rather than additive interpolation. It is not an action-value policy and does not inherit the failed causal-gradient claim.

## Formal scope

The finite action is the unique fixed-azimuth shortest geodesic to the target prototype at constant norm, except at declared pole degeneracies. The logistic model minimizes regularized weighted log loss on the declared scalar feature. Neither property establishes semantic correctness or globally optimal behavioral control.

## Frozen architecture transfer

For each model, fit only model-specific `b`, `a`, and `d` on the same 300 representation-fit groups. DAPS remains fixed at `alpha=1,rho=0.5`. Matched Spherical receives its frozen nine-cell observer grid at `kappa=20`, `alpha` in `{0.6,0.9,1}`, and `beta` in `{0.6,0.9,0.99}`. Observer-fit is used only for the baseline selector and per-method scalar temperatures.

| Family | Model revision | Layer |
|---|---|---:|
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71` | 21/32 |
| OLMo-2 | `allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84` | 21/32 |
| Granite | `ibm-granite/granite-3.3-8b-instruct@51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | 26/40 |

The 160 observer groups screen only runtime validity and fit temperatures: a family advances if DAPS raw MC2 is at least matched Spherical and MC1 is no more than 0.015 lower. A passed family evaluates the unchanged pair once on the 50 ID and 187 shifted groups. Qwen3 development data are not pooled with architecture-family effects.

## Frozen hypotheses

| ID | Hypothesis | Success | Refutation |
|---|---|---|---|
| H10.1 | Deployment-aligned posterior geometry beats matched Spherical across decoder families. | Multiplicity-adjusted hierarchical and per-family confirmatory gains with MC1 non-inferiority. | Any required family reverses beyond margin or aggregate interval includes zero. |
| H10.2 | The learned detector direction is specific beyond intervention magnitude. | DAPS beats five exact update-norm isotropic controls per family. | Nulls explain any family effect. |
| H10.3 | Alignment to action positions is necessary. | Exact-span DAPS beats a last-token/unprompted fit ablation under the same budgets. | Ablation matches it across families. |
| H10.4 | The method extends beyond truthfulness. | At least two separately frozen non-truthfulness behavior panels pass without changing the construction. | Required behavior panels fail. |

## Novelty boundary

Qwen3 supports only a development claim. The method is competitive or SOTA only if architecture, behavior, specificity, capability, collateral, and matched-budget gates pass. The failed gradient-axis hypothesis remains reported.
