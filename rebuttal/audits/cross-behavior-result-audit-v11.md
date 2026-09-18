# Cross-behavior result audit, method version 11

Status: frozen six-cell transfer hypothesis contradicted. One observer cell advanced, and its confirmatory behavior gate failed.

## Experiment contract

Benchmark version 18 fixed DAPS at `alpha=1,rho=0.5` across BBQ and ETHICS commonsense on Mistral, OLMo-2, and Granite. Each cell compared DAPS with 23 configurations spanning CAA, empirical projection transport, AcT-style coordinate transport, MiMiC full-covariance transport, and Spherical Steering. Representation fit, observer fit, and confirmatory units were disjoint. BBQ used template clusters; ETHICS used scenario rows. The observer rule required DAPS raw MC2 to be at least the strongest selected classical baseline and raw MC1 to be no more than 0.015 lower.

## Observer results

| Panel | Model | DAPS MC1 | DAPS raw MC2 | Selected baseline | Baseline MC1 | Baseline raw MC2 | Screen | Runtime, s |
|---|---|---:|---:|---|---:|---:|---|---:|
| BBQ | Mistral | 0.60682 | 0.60957 | Spherical `alpha=1,beta=0.6` | 0.77045 | 0.75932 | fail | 230.32 |
| BBQ | OLMo-2 | 0.49318 | 0.48945 | Spherical `alpha=0.9,beta=0.99` | 0.56591 | 0.56386 | fail | 254.99 |
| BBQ | Granite | 0.49773 | 0.49593 | Spherical `alpha=0.6,beta=0.99` | 0.50682 | 0.50665 | fail | 306.61 |
| ETHICS | Mistral | 0.91000 | 0.90880 | CAA `strength=1` | 0.85667 | 0.85043 | pass | 70.56 |
| ETHICS | OLMo-2 | 0.76333 | 0.76320 | Spherical `alpha=0.9,beta=0.6` | 0.90667 | 0.89813 | fail | 75.94 |
| ETHICS | Granite | 0.90667 | 0.90795 | Spherical `alpha=0.9,beta=0.6` | 0.91667 | 0.91200 | fail | 76.86 |

Five cells stopped without confirmatory output. The full six-cell observer denominator remains visible; no failed family or panel is removed.

## Authorized confirmatory result

ETHICS Mistral evaluated DAPS and its observer-selected CAA baseline once on 1,000 confirmatory scenarios. With 10,000 paired BCa resamples and two-sided Bonferroni alpha `0.05/6`, calibrated MC2 improved by `0.10730`, interval `[0.09030,0.12417]`, and MC1 improved by `0.02400`, interval `[-0.00400,0.05000]`. The primary comparison passed.

The frozen polarity gate failed. Label-0 calibrated MC2 improved by `0.18429`, while label-1 improved by `0.01321`; their difference was `0.17108`, above the maximum permitted `0.05`. The label-1 interval `[-0.00608,0.03037]` also included zero, and its MC1 change was `-0.09333`, interval `[-0.12444,-0.07111]`. This pattern is consistent with a fixed response-polarity shift and does not support behavior-balanced moral-judgment steering.

## Decision

- H11.1 is contradicted because five of six observer cells failed.
- H11.2 is contradicted at the observer stage because all three BBQ cells failed.
- H11.3 is contradicted by the ETHICS Mistral label-stratum imbalance.
- H11.4 was not authorized because no cell passed both the primary and behavior gates.
- Strong-baseline, specificity, capability, and collateral stages were not authorized for v11.
- The v11 cross-behavior evidence class is `EVIDENCE_THIN`; it cannot support multi-behavior, multi-architecture, competitive, Pareto-optimal, or SOTA wording.

## Runtime and lineage

All six observer cells and the one confirmatory cell ran on `avi-gn-fsk42`, NVIDIA H100 80GB HBM3, Torch `2.4.1+cu121`, CUDA `12.1`. The uncached ETHICS Mistral confirmatory pair took 17.55 seconds after launch, including 5.78 seconds model load, 5.46 seconds DAPS evaluation, and 5.06 seconds matched-baseline evaluation. These are batch-run receipts, not deployment-throughput claims.

| Receipt | SHA-256 |
|---|---|
| BBQ Mistral observer | `39489bc39fefb049ed62989302cf7107f6e811599d686e3e069bd66be29bbafb` |
| BBQ OLMo-2 observer | `53eca04b1def2d626db4004a23e32441b31d852eecb374ceeec90c9f09e3d82c` |
| BBQ Granite observer | `ee5b8346f675d71207e649a1d95bfefaaee75ffaedb467226a1aa44e29921bf3` |
| ETHICS Mistral observer | `ba97374e95856e5efa99864418912bae8de4f5767ab11d74ffc9df1a3109b272` |
| ETHICS OLMo-2 observer | `fa9429ab554a8827c955d9907d2f414998b6f179f9a0b32b0b29903f744aa913` |
| ETHICS Granite observer | `75bdb58d0f4beacdcd4547480195c22d294ad9efd3f9f46924f522a252c25510` |
| ETHICS Mistral confirmatory | `c611ff4517d850b461844a28fce5f93d90545b52ab7e350c2b300a91cf4f42b0` |

