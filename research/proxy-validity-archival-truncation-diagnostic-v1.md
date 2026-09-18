# Local-judge archival truncation diagnostic, version 1

Date: 2026-09-07. Status: post-hoc diagnostic after proxy-validity version 1 closed.

## Question and source boundary

The proxy-validity smoke lost two required ratings at the frozen 160-token limit. To test whether this was an isolated output, this diagnostic reads two already opened Fisher-protected FLAS local-Qwen packets that had been generated with a longer judge budget. It does not generate, judge, relabel, retry, or select anything.

The 2B source is `artifacts/development/fisher_protected_flas_v16_validation/local_qwen3_14b_proxy.json`, SHA-256 `548e084e1e8f49d2f5d1381decf20e2c1120969d2901002de96120ad3c2fc94b`. The 9B source is `artifacts/development/fisher_protected_flas_v16_9b_validation/local_qwen3_14b_proxy.json`, SHA-256 `7b6162caea4d7e1dc030ef7f617ae919faa3d80d1400b22ac5035768c8932dcf`. Each contains 320 completely parsed rows and 960 preserved raw completions from the same Qwen3-14B model and upstream prompt family.

For each raw completion, the diagnostic tokenizes the prefix through the line containing the last `Rating:` marker with the pinned Qwen tokenizer. A rating position above 160 indicates that a 160-token truncation would occur before the complete rating line. This is a token-position stress test of archived text, not an exact replay under the current Transformers runtime.

## Results

| Archive | Rows with any rating after token 160 | Dimensions after token 160 | Maximum rating position |
|---|---:|---:|---:|
| Gemma-2-2B conditions | 23/320 (7.19%) | 23/960 (2.40%) | 238 |
| Gemma-2-9B conditions | 30/320 (9.38%) | 30/960 (3.12%) | 193 |

Across both archives, late ratings occur in 9/640 Concept evaluations, 1/640 Instruction evaluations, and 43/640 Fluency evaluations. The affected-row counts also vary across the five methods: Fisher no-rescale 6/128, Fisher selected 13/128, FLAS 13/128, random basis 7/128, and shuffled basis 14/128.

## Interpretation

The smoke failure is consistent with a recurring explanation-length problem, concentrated in the Fluency prompt. More importantly, affected-row frequency is not constant across methods in these archives. Deleting incomplete rows or assigning them zero can therefore distort method contrasts rather than merely reduce sample size.

This post-hoc evidence strengthens only the evaluator-failure diagnosis. Runtime versions differ, the archived conditions are not the Flow-step conditions, and token position does not prove what a fresh truncated decode would emit. It neither estimates PV.2--PV.5 nor validates an alternative judge. A successor evaluator needs a new prospective calibration packet and an external behavioral anchor; parser completeness alone is insufficient.
