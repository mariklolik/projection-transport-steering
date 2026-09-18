# Architecture confirmatory audit, method version 10

Audit date: 2026-09-05. All estimates use the frozen allocation and rules in benchmark versions 16 and 17.

## Architecture gate

Granite failed the observer screen and its 237 held-out groups remain unopened. The frozen three-family hypothesis H10.1 therefore fails regardless of the two advancing-family results.

Mistral and OLMo-2 were evaluated once on the same fixed composition of 50 in-domain and 187 shifted questions. Intervals in the primary architecture gate use 10,000 paired BCa resamples and Bonferroni two-sided alpha `0.05/3`.

| Family | DAPS calibrated MC2 | Spherical calibrated MC2 | MC2 difference | adjusted interval | MC1 difference | adjusted interval | joint gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Mistral | 0.72932 | 0.57628 | +0.15303 | [0.10795, 0.19718] | +0.03376 | [-0.05907, 0.12236] | fail |
| OLMo-2 | 0.74700 | 0.60981 | +0.13719 | [0.10487, 0.16815] | +0.10549 | [0.02532, 0.17722] | pass |
| Granite | not opened | not opened | not opened | not opened | not opened | not opened | stopped at observer |

Mistral shows a large calibrated-MC2 gain, including positive unadjusted ID and shift MC2 intervals, but its multiplicity-adjusted MC1 interval crosses the frozen non-inferiority margin. It did not authorize controls.

OLMo-2 passes both primary criteria. Its ID calibrated-MC2 difference is +0.11409 with 95% interval [0.07237, 0.16032], while its ID MC1 interval is inconclusive. Its shifted stratum passes both descriptive criteria: calibrated-MC2 +0.14337 [0.11281, 0.17410] and MC1 +0.12299 [0.04813, 0.19251].

## OLMo-2 specificity and alignment controls

The seven-control family was frozen before any control output. Intervals use 10,000 paired BCa resamples and Bonferroni two-sided alpha `0.05/7`.

| Control | control calibrated MC2 | DAPS minus control MC2 | adjusted interval | DAPS minus control MC1 | adjusted interval |
|---|---:|---:|---:|---:|---:|
| No intervention | 0.44831 | +0.29869 | [0.25076, 0.34505] | +0.28692 | [0.17722, 0.38397] |
| Isotropic 1 | 0.49851 | +0.24849 | [0.20735, 0.28881] | +0.22363 | [0.11814, 0.31646] |
| Isotropic 2 | 0.50942 | +0.23758 | [0.19810, 0.27544] | +0.22785 | [0.13080, 0.32489] |
| Isotropic 3 | 0.52744 | +0.21956 | [0.17968, 0.25751] | +0.19409 | [0.09705, 0.28692] |
| Isotropic 4 | 0.52655 | +0.22045 | [0.18022, 0.25848] | +0.20253 | [0.10549, 0.28892] |
| Isotropic 5 | 0.52607 | +0.22093 | [0.18009, 0.26059] | +0.16878 | [0.07173, 0.26160] |
| Unprompted last-token fit | 0.45157 | +0.29543 | [0.24676, 0.34286] | +0.28692 | [0.17722, 0.38397] |

All seven comparisons pass. On OLMo-2, the effect is not explained by intervention magnitude alone, and aligning the representation fit to prompted causal answer positions is necessary relative to the frozen unprompted last-token ablation.

## Claim decision

H10.2 and H10.3 are supported only for the confirmed OLMo-2 subset. H10.1 is contradicted by the required Granite stop and Mistral joint-gate failure. H10.4 remains untested. These results support a strong OLMo-2 truthfulness result with explicit uncertainty, not a multi-architecture or multi-behavior SOTA claim.
