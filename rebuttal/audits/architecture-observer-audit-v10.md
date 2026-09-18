# DAPS architecture observer audit, method version 10

Status: Mistral and OLMo-2 passed; Granite failed and stopped before held-out output.

| Family | DAPS MC1 / raw MC2 | Matched Spherical MC1 / raw MC2 | Screen |
|---|---:|---:|---|
| Mistral-7B-Instruct-v0.3 | 0.63750 / 0.75135 | 0.52500 / 0.65488 | pass |
| OLMo-2-7B-Instruct | 0.63750 / 0.76452 | 0.58750 / 0.71660 | pass |
| Granite-3.3-8B-Instruct | 0.58125 / 0.75603 | 0.60000 / 0.77776 | fail |

Mistral selected Spherical `alpha=1,beta=0.6`; OLMo-2 selected the same setting. Granite selected `alpha=0.9,beta=0.99`, exceeding DAPS by 0.02173 raw MC2 and 0.01875 MC1. Granite's 120 development and 237 held-out groups remain unopened.

The frozen three-family hypothesis cannot pass because failed families may not be dropped. Benchmark version 16 nevertheless authorizes separate confirmatory estimates for Mistral and OLMo-2, with architecture-subset wording only if they pass.

An initial parallel-launch attempt was terminated before fit output because the log parent directory did not exist. Only deterministic split manifests were present. The unchanged commands were rerun after creating their artifact roots.
