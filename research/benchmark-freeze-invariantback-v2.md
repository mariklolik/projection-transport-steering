# InvariantBack pilot freeze, version 2

Freeze date: 2026-09-06. Status: operational amendment frozen before dataset materialization or model output. It supersedes version 1 only where this document is more specific; all hypotheses, methods, endpoints, grids, uncertainty rules, gates, and stop rules remain unchanged.

## Source revisions

- NormBank: `SALT-NLP/NormBank` revision `dacdc9a905d509d9d1aca1c9a031e4927d8ab815`, file `NormBank.csv`; valid labels are exactly `taboo`, `normal`, and `expected`. The independent source key is normalized exact `setting + behavior`, and a group is eligible only when all three labels have a nonempty endpoint.
- MNLI: `nyu-mll/multi_nli` revision `da70db2af9d09693783c3320c4249840212ee221`, file `data/train-00000-of-00001.parquet`; valid labels are exactly entailment `0`, neutral `1`, and contradiction `2`. The independent source key is the whitespace-normalized, lowercased premise, and a group is eligible only when all three labels have a nonempty hypothesis.
- SC101: `wassname/social_chemistry_101` revision `a7869978d5d441d89327067d8ff543d6dcc3fc32`, file `social_chem_101.parquet`, restricted to source split `train`; action moral judgments `{-2,-1}`, `{0}`, and `{1,2}` map prospectively to `bad`, `ok`, and `good`. A group is one deterministic without-replacement triplet of unique normalized actions, one per label. SC101 is action-level and is not called context matched.

Whitespace is normalized before grouping. Duplicate eligible endpoints are resolved by the lexicographically smallest SHA-256 source-row identifier. Empty, invalid-label, and incomplete groups are excluded. All exclusions and counts are recorded before model output.

## Allocation

Allocation seed is `20260906`. For each dataset, eligible group identifiers are ordered by SHA-256 of the seed, dataset, and group identifier. The first 80 groups are construction, the next 8 are sentinel, the next 24 are development, and the next 48 are sealed pilot-test groups. This resolves the construction-pool size left unspecified in version 1 without changing the already frozen exposed and sealed evaluation sizes.

The public packet contains construction, sentinel, and development groups. The pilot packet is serialized separately, hashed before public-packet creation, and is not copied to the GPU host during replay, sentinel, or development. No source endpoint or group identifier may occur in more than one allocation.

The three construction row permutations are zero-based `(0,1,2)`, `(1,2,0)`, and `(2,0,1)`. The three held-out row permutations are `(0,2,1)`, `(2,1,0)`, and `(1,0,2)`. Each family is row-balanced. Construction crosses its three permutations with all six semantic mappings and the identifier vocabularies `A/B/C`, `X/Y/Z`, and `1/2/3`. Held-out encoded evaluation uses only `I/II/III` and `one/two/three` and the held-out permutations.

## Promotion boundary

Only construction and sentinel data may be used for IA.1 replay, feasibility, and premise checks. Development opens only after those checks pass. Pilot-test data remains sealed until a single candidate configuration and strongest feasible baseline are frozen from development. Any insufficient eligible count, allocation overlap, hash mismatch, replay mismatch, mapping-comprehension failure, or feasibility below the version-1 threshold closes the route without changing sources, filters, sizes, or seed.
