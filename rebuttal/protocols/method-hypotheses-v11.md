# Cross-behavior hypotheses, method version 11

Status: frozen before any BBQ or ETHICS model output in the target repository.

## Unchanged method construction

Deployment-aligned posterior spherical steering remains unchanged from method version 10. For each behavior panel and model, desired and undesired candidate answers replace truthful and false answers. The unit mean contrast, answer-balanced causal-position posterior, hard `rho=0.5` decision, norm-preserving shortest-geodesic action, and fixed `alpha=1` are identical. The panel supplies only its instruction, candidate answers, desired labels, immutable groups, and strata.

BBQ uses the original natural-language answer alternatives. The instruction requires answering only from the supplied context and selecting the indeterminate alternative when the context is insufficient. Each four-row template is an indivisible statistical cluster. ETHICS commonsense presents each scenario with the natural-language alternatives `No, the action is not morally wrong` and `Yes, the action is morally wrong`; label 1 selects the second alternative.

## Frozen hypotheses

| ID | Hypothesis | Success | Refutation |
|---|---|---|---|
| H11.1 | DAPS improves desired-answer probability over the strongest observer-selected classical steering baseline across social-bias and moral-judgment panels and three decoder families. | In every cell, DAPS passes its observer screen against the strongest of CAA, empirical projection transport, AcT-style coordinate transport, MiMiC full-covariance transport, and Spherical Steering; all six multiplicity-adjusted confirmatory intervals have calibrated desired-probability lower bound above zero and accuracy lower bound above -0.015. | Any cell stops, reverses, or misses either bound. |
| H11.2 | BBQ gains reduce ambiguous-context stereotype reliance rather than only moving disambiguated accuracy. | Each model has positive ambiguous desired-probability change, nonnegative bias-free-score change, and no category with an adjusted accuracy reversal beyond -0.025. | The effect is confined to disambiguated contexts or creates a protected-category reversal. |
| H11.3 | ETHICS gains are not a fixed yes/no label shift. | Desired-probability gains are positive in both label strata and their difference is no larger than 0.05 in magnitude. | One label stratum reverses or the polarity gap exceeds 0.05. |
| H11.4 | The cross-behavior result requires both the posterior gate and aligned direction rather than generic intervention magnitude. | Every advancing cell beats no-op, an always-act same-direction action, and five exact update-norm isotropic directions under a separately frozen family-wise analysis. | Removing the gate or replacing the aligned update with a norm-matched null explains an advancing cell. |

## Claim boundary

These panels test transfer of a fixed geometric construction, not universal moral correctness or absence of social bias. BBQ and ETHICS labels are dataset-defined constructs. The expanded classical comparator family makes a passed panel substantially stronger than a Spherical-only comparison, but it still does not cover prompting, finetuned controllers, nonlinear bridges, capability, collateral behavior, or cost. SOTA wording remains blocked until those separate gates pass.
