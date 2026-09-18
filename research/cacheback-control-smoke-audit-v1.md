# CacheBack control and efficiency smoke audit, version 1

Date: 2026-09-06. Status: the complete comparator preflight passes; development is authorized but remains unopened.

## Decision

The exact future Jacobian should be constructed with forward-mode automatic differentiation over the 768-dimensional intervention state. On the frozen exposed smoke state, this preserves the exact realized outputs while reducing the `H=8/H=0` geometry-time ratio to `4.49` and peak memory to 14.32 GB. It passes the one-state versions of the frozen 12x and 40 GB limits.

Top-5000 Fisher is a high-fidelity approximation on this state but does not reduce geometry time or memory. It remains a diagnostic and cannot replace the exact primary method.

The differentiable cache-path split passes algebraic tests, and explicit incremental key/value replay passes on the exposed pretrained GPT-2 smoke state with the matched exact H8 action. CAA/ActAdd is now fit from all 633 frozen third-person construction pairs and evaluated at the same realized effect. The separate GCAD audit finds that GCAD is not compatible with this frozen data and actuator contract; its absence forbids any attention-side or frontier-superiority claim. Development states remain unopened at the time of this audit.

## Performance lineage

All attempts use the same frozen smoke state, layer 6, target 0.05, eight base continuations for `H=8`, exact full-vocabulary realized KL, and H100 80 GB GPUs.

| Packet | Jacobian strategy | Result | Geometry time | Peak allocation | Decision |
|---|---|---|---:|---:|---|
| `cacheback_v1_control_smoke` | reverse mode, one offset and trajectory at a time | complete | 29.41 s exact | 11.56 GB | correct but honest cost ratio unavailable because H0 was redundantly repeated |
| `cacheback_v1_control_smoke_batched` | all trajectories and offsets in one reverse Jacobian | stopped without exact receipt | >143 s | 76.15 GiB observed | reject: exceeds 40 GB and is slower |
| `cacheback_v1_control_smoke_offsetbatched` | all offsets in one reverse Jacobian, trajectories sequential | complete | 32.90 s exact | 65.50 GB | reject: exceeds 40 GB and is slower |
| `cacheback_v1_control_smoke_forwardad` | all offsets in one forward Jacobian, trajectories sequential | complete | 4.09 s exact | 14.32 GB | accept for the next comparator smoke |

The valid `H=0` forward-mode measurement deduplicates the trajectory-independent immediate geometry: 0.9102 seconds, 13.28 GB, and 768 derivative rows. Exact `H=8` takes 4.0864 seconds, 14.32 GB, and represents 55,296 derivative rows. The time ratio is `4.4895`.

The accepted exact result reproduces the bounded reverse-mode smoke numerically. Exact `H=8` realized cumulative forward KL is `1.69498e-7`, immediate semantic effect is `0.0502665`, and unregularized quadratic prediction is `1.48203e-7`. The earlier bounded reverse-mode values were `1.70030e-7`, `0.0502669`, and `1.48203e-7`; the small realized difference is consistent with backend numerical ordering and does not alter the premise decision.

## Top-5000 diagnostic

With the same forward-mode Jacobian, top-5000 takes 4.0453 seconds and 14.32 GB, or `0.990x` the exact time. Relative to exact full-vocabulary geometry:

- its `H=8` action norm is `0.99243x`;
- its `H=8` immediate effect is `1.00015x`;
- its realized `H=8` cumulative KL is `1.00477x`;
- its `H=0` action norm is `1.00147x` and its realized cumulative KL is `0.95935x`.

This is promising approximation fidelity on one exposed state, but the full-vocabulary Fisher is not the runtime bottleneck. Top-5000 earns no efficiency claim and remains secondary.

## Matched-effect comparator smoke

Accepted results are in the `attempt4` packets. Attempt 1 completed the numerical work but failed JSON serialization because horizon and baseline metadata used mixed key types. Attempt 2 repaired serialization but retained ambiguous horizon identifiers. Attempt 3 normalized identifiers and established the numerical result. Attempt 4 supersedes it by also recording the automatic-differentiation backend, full runner and case time, exact derivative counts, model-forward counts, Jacobian calls, JVP/VJP counts, action evaluations, and zero pilot access.

All actions use the same local target 0.05, scale bracket `[0,1]`, at most 24 evaluations, no extrapolation, and realized-effect tolerance `1e-4`.

| Method | Realized effect | Cumulative forward KL | Reduction versus H0 | Matching scale |
|---|---:|---:|---:|---:|
| exact FishBack H0 | 0.0500025 | `1.20561e-5` | reference | 0.998047 |
| CacheBack H2 | 0.0500913 | `1.12995e-6` | 90.63% | 0.996094 |
| CacheBack H4 | 0.0500401 | `2.89761e-7` | 97.60% | 0.996094 |
| CacheBack H8 | 0.0500646 | `1.68137e-7` | 98.61% | 0.996094 |
| Euclidean | 0.0500953 | `1.11987e-5` | 7.11% | 0.996094 |
| CAA/ActAdd | 0.0499994 | `5.54085e-4` | -4495.88% | 0.933594 |

CAA incurs 45.96 times the H0 KL and 3295.43 times the H8 KL on this exposed state. This large margin survives realized-effect matching and cannot be explained by the earlier 7% endpoint-effect mismatch. It is nevertheless one state and supplies no powered superiority result.

Top-5000 H8 produces `1.68194e-7` at realized effect 0.0500731, only 0.034% more KL than exact H8. Its full runner time is 11.08 seconds versus 11.34 seconds for exact in separate concurrent processes, which is not evidence of a reliable speed advantage.

The accepted exact packet records forward-mode AD, 55,296 derivative rows, eight Jacobian calls, 6,144 JVP directions, zero VJPs, 52 action evaluations, 7.99 seconds for the complete case after model loading, 11.34 seconds for the complete runner, 14.32 GB peak allocation, and zero pilot-state access.

## Cache-path scope

`CachePathDownstream` now represents the frozen first-order path partition. Reset retains the immediate Jacobian and zeros every future Jacobian; cache-only zeros the immediate Jacobian and retains every future Jacobian; their sum exactly reconstructs the full Jacobian in the synthetic test. This verifies the algebraic decomposition expected from a one-shot causal decoder intervention.

`GPT2IncrementalDownstream` reconstructs the downstream layer-7-to-11 cache from the prefix, substitutes the intervention state at the source token, and consumes frozen future layer-6 token states one at a time. On the pretrained GPT-2 smoke trajectory, its maximum representation error against full parallel replay is `7.63e-5` at base and `9.16e-5` under the intervention.

The original arbitrary norm-0.01 wiring perturbation is retained only as implementation lineage. The accepted matched packet constructs the exact H8 action from eight frozen trajectories, scales it to realized effect 0.0500646, and preserves the resulting 768-dimensional action by hash.

With that matched H8 action, the explicit cache-reset construction preserves 99.94% of immediate KL and removes 99.07% of future KL. Cache-only leaves 0.0075% of immediate KL and recovers 98.31% of future KL. The base and steered incremental replay errors are both `7.63e-5`. Raw values, the action, and the runner receipt are preserved in `artifacts/development/cacheback_v1_cache_path_smoke_matched`.

This passes the one-state wiring smoke for the frozen 80% removal and 50% recovery thresholds. It remains structurally expected for a one-shot causal decoder and is not independent evidence that cache mediation is a novel mechanism. Powered CB.6 inference still requires matched CacheBack actions across independent states.

## Development boundary

The frozen pre-development requirements now have the following dispositions:

1. CAA/ActAdd fit and matched evaluation: pass.
2. Attention-side compatibility audit: complete; GCAD is incompatible and is recorded as a claim limitation.
3. Matched exact horizons, Euclidean, CAA/ActAdd, and top-5000: pass.
4. Backend, compute-count, wall-time, memory, and pilot-access receipts: pass.
5. Matched-action cache removal and recovery: pass.

The 24 sealed development states may now be opened for frozen layer, regularization, and dose selection. The strongest current statement remains a local exposed-state premise. No pilot, powered mechanism, behavioral, architecture, or SOTA conclusion is authorized.
