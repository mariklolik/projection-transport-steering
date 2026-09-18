# CacheBack premise-sentinel audit, version 1

Date: 2026-09-06. Status: premise supported; development comparison remains blocked.

## Decision

The exposed eight-state sentinel supports the narrow premise that the future-distribution metric is numerically distinct from exact `H=0` FishBack and can substantially reduce teacher-forced `H=8` forward KL at nearly unchanged immediate semantic effect. This is not a powered result, a behavioral result, or a method-comparison result. Pilot states remain unopened.

Development is not yet authorized. The frozen CAA/ActAdd, top-5000 approximation, causal cache-reset/cache-only, dose matching, layer/regularization grid, and compatible attention-side control have not all passed their own smoke tests.

## Provenance and failed attempts

The source packet is `artifacts/development/cacheback_v1_data/data.json`, SHA-256 `bd7a72ba87d90e4d6df128ea3b60b66b49d2c14e4a7e85644fc4b44fc4f6e1ae`. It contains 81 globally document- and context-disjoint states; this audit opens only the frozen eight-state sentinel.

Four attempts are retained rather than overwritten:

| Attempt | Result | Diagnosis | Scientific use |
|---|---|---|---|
| `cacheback_v1_sentinel` | 0/8 pass | partial GPT-2 replay dropped the explicit causal mask passed positionally by Transformers | none; implementation-control failure |
| `cacheback_v1_sentinel_attempt2` | 8/8 executed | replay fixed, but realized KL accumulated in float32 and produced a negative cancellation artifact | none; numerical-control failure |
| `cacheback_v1_sentinel_attempt3` | 8/8 executed | float64 KL valid, but stored quadratic cost included solver regularization | realized-KL debugging only |
| `cacheback_v1_sentinel_attempt4` | 8/8 pass | causal replay exact, float64 KL nonnegative, physical metric separated from solver regularization | valid exposed premise sentinel |

The valid analysis is `artifacts/development/cacheback_v1_sentinel_attempt4/analysis.json`, SHA-256 `2e04167e6c0a2101a94058d27d54f92b6e35236930072f22dae5b16336b1a571`. Its runner SHA-256 is `fd006090a1fc7987e613af52ef33d9015d0617586e4141e615f8b7eca06b6380`. All four shard receipts report two passed cases, zero failed cases, and `pilot_states_observed = 0` under PyTorch `2.9.1+cu128` on H100 80 GB GPUs.

## Exposed sentinel results

All reductions compare realized cumulative teacher-forced `H=8` forward KL with the same-state exact `H=0` action. Intervals are fixed-seed state bootstrap diagnostics, not confirmatory intervals.

| Action | Mean KL reduction | Diagnostic 95% interval | Median | Worst state | Minimum immediate-effect retention |
|---|---:|---:|---:|---:|---:|
| CacheBack `H=2` | 89.05% | [86.65%, 91.64%] | 88.81% | 84.08% | 99.04% |
| CacheBack `H=4` | 93.09% | [90.76%, 95.40%] | 93.38% | 87.50% | 99.03% |
| CacheBack `H=8` | 95.62% | [94.19%, 97.12%] | 95.26% | 92.17% | 98.97% |
| Euclidean | -427.30% | [-759.28%, -122.18%] | -201.90% | -1217.42% | 98.96% |

The result is not driven by a single morphology concept. Mean `H=8` reductions are 94.16% for progressive, 97.28% for past tense, and 95.97% for third-person singular. These are exposed descriptive strata with only two or three states each.

The action changes are well above numerical noise: median cosine with `H=0` is `0.746` at `H=2`, `0.651` at `H=4`, and `0.588` at `H=8`; the corresponding observed ranges are `[0.526, 0.771]`, `[0.439, 0.674]`, and `[0.398, 0.625]`. Nested metric increments are PSD in every state; the smallest recorded increment eigenvalue is positive up to numerical precision.

The unregularized full-trajectory quadratic prediction ranks realized KL perfectly across the eight states for each candidate action in this exposed packet. The worst relative Taylor error is 7.93% for `H=2`, 8.90% for `H=4`, and 10.88% for `H=8`. This supports local calibration at the frozen smoke dose only; it does not establish calibration across doses.

## Severe-case and mechanism diagnostics

The largest single-offset share for `H=8` is `0.7993448`, only `0.0006552` below the frozen `0.80` ceiling. The aggregate direction is favorable, but this margin is too fragile to treat the concentration gate as established. Development must compute the frozen paired-reduction definition rather than reuse the absolute within-action share used by this premise analyzer.

The stored `cache_reset_forward_kl_mean` and `cache_only_forward_kl_mean` values are output-offset decompositions, not the frozen causal cache interventions. They cannot establish CB.6. A real cache-reset path must preserve the immediate steered logits while restoring every affected current-token key/value entry before the next token; cache-only must preserve the altered key/value path while clamping the immediate output to base. Their Jacobians and realized KL must be tested independently.

The exact construction produces 55,296 derivative rows per state and 442,368 across the sentinel. Geometry takes 11.30-43.30 seconds per state, with median 18.00 seconds. Median peak allocation is 6.74 GB and the maximum is 16.81 GB, below the 40 GB memory ceiling. The frozen `H=8/H=0` wall-time ratio is not identified by a run that computes nested horizons together and remains open.

## Next experiment

Before opening any development state:

1. implement causal cache-reset and cache-only paths and prove them on a synthetic two-step decoder and pretrained GPT-2 replay;
2. add CAA/ActAdd and top-5000 FishBack under the same realized-effect matcher;
3. audit whether a budget- and equation-compatible attention-side control is implementable; record an explicit incompatibility rather than silently dropping it;
4. run one-state comparator smoke and an `H=0`-only timing smoke;
5. only then open the 24 frozen development states for layer, regularization, and dose selection.

The valid conclusion at this checkpoint is: future-aware local geometry is a strong premise candidate on exposed GPT-2 morphology states. Cache-mediated mechanism, generalization, practical efficiency, and frontier superiority are unverified.
