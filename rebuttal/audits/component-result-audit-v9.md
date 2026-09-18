# Qwen3 TruthfulQA component audit, method version 9

Status: causal-gradient component claim failed. Architecture transfer for CDAS is not authorized.

## Family-wise result

On the 237-group fixed extension, CDAS remained far above no-op, CAA, scalar transport, AcT, MiMiC, matched Spherical, and every exact update-norm random direction under 13-comparison Bonferroni BCa intervals. The closest controls falsified the claimed mechanism.

| Control | Control calibrated MC2 | CDAS minus control | Adjusted interval |
|---|---:|---:|---:|
| Detector-axis hard spherical | 0.73909 | -0.01404 | [-0.02982,+0.00174] |
| Sign-flipped control axis | 0.72781 | -0.00277 | [-0.02431,+0.01876] |
| Matched Spherical | 0.59044 | +0.13460 | [+0.09058,+0.17910] |
| Strongest of five random controls | 0.53859 | +0.18645 | [+0.14497,+0.22974] |
| No-op | 0.45522 | +0.26982 | [+0.21944,+0.32126] |

The unadjusted candidate-minus-detector interval was strictly negative `[-0.02506,-0.00345]`. The signed-likelihood gradient therefore does not explain the Qwen3 improvement and is not promoted.

## Supported component

The deployment-aligned detector-axis action itself beats matched Spherical by `0.14865` calibrated MC2, 95% BCa `[0.11376,0.18247]`, and by `0.10549` MC1 `[0.03566,0.16878]`. This component uses a contrast direction fitted on exact prompted action-span answer means, a balanced token posterior, and a hard norm-preserving geodesic action.

## Decision

H9.2 fails and CDAS stops. A versioned detector-axis method may proceed to still-sealed architecture families because it is fully determined from the component result, but Qwen3 remains development-only. Any architecture-transfer claim must include the failure of the gradient axis and must compare against matched Spherical and construction nulls.

## Receipts

- `artifacts/development/truthfulqa_qwen3_cdas_v1/component-selection.json`
- `artifacts/development/truthfulqa_qwen3_cdas_v1/development-extension-v9/component-analysis.json`
- `artifacts/development/truthfulqa_qwen3_cdas_v1/development-extension-v9/*.json`
- `artifacts/development/truthfulqa_qwen3_cdas_v1/components_extension.log`
