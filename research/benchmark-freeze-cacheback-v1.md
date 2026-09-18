# CacheBack pilot freeze, version 1

Freeze date: 2026-09-06. Frozen before implementation or model output.

## Model, source states, and rollouts

The only model is the pinned local GPT-2 Small checkpoint at `.external/hub/gpt2`. Targets are the same three verb-morphology concepts used by the FishBack audit: third-person singular, progressive, and past tense. New source documents are hash-disjoint from the exposed version-18 and version-20 development contexts.

The exposed smoke uses one state. The sentinel uses 8 different states with fixed concept counts `3/3/2`. Development uses 24 different states, balanced across concepts. The untouched pilot uses 48 further states, balanced across concepts. The independent statistical unit is the source-document state; rollout trajectories are repeated measurements inside a state.

For each state, sample 8 continuations from the frozen base model at temperature 1, top-p 1, no top-k truncation, and maximum 9 new tokens. Philox seed is `sha256(state_id || rollout_index || 20260906)` reduced to 64 bits. The sampled tokens are teacher-forced for every horizon, method, cache mode, and dose. Horizons are `{0,2,4,8}` and use nested prefixes of the same trajectories.

## Geometry

The primary forward KL is `KL(P0 || Pu)` at each teacher-forced offset. Softmax Fisher uses the full vocabulary. Jacobians, Fisher matrices, horizon sums, eigendecompositions, and Cholesky solves use float64 after model forward values are produced in the checkpoint's native inference dtype. Symmetry is enforced only by `(G + G^T)/2`; negative eigenvalues below `-1e-8 max(1, ||G||_2)` fail the group.

Candidate layers are `{3,6,9}`. Regularization multipliers are `{0.01,0.1,1.0}` times the statewise median positive eigenvalue of `G_0`, with the same resulting `lambda` used across all horizons for that state. The smallest positive eigenvalue floor is `1e-12` times the largest eigenvalue. Development selects one layer and multiplier by primary KL subject to every gate below; ties within `1e-6` choose the earlier layer and larger regularization.

## Methods and calibration

- exact FishBack `H=0`;
- CacheBack at `H=2,4,8`;
- Euclidean gradient action;
- CAA/ActAdd direction from the same fit examples;
- future metric under cache reset;
- cache-only action evaluation;
- top-5000 FishBack approximation diagnostic;
- GCAD-style attention-side control only if an equation- and budget-matched implementation passes the sentinel; its absence cannot be used to claim frontier superiority.

The first premise sentinel is intentionally smaller than the development comparison. It runs exact `H=0,2,4,8`, Euclidean, cache-reset, and cache-only diagnostics only. CAA/ActAdd, the top-5000 approximation, and any compatible GCAD control must be implemented and pass their own smoke before development, but they cannot block learning whether the future metric is distinct. No primary superiority or frontier statement can be computed from the premise sentinel.

Every action first satisfies the same local constraint `q^T u = rho`. Doses `{0.125,0.25,0.5}` in normalized local target units are evaluated in development. A deterministic bisection then matches the realized immediate semantic log-odds effect to the largest common dose reached by at least 80% of states for every required method. Bisection uses at most 24 evaluations, an absolute effect tolerance of `1e-4`, no extrapolation, and the same bracket for every horizon. The selected dose is frozen before pilot-test access.

## Outcomes

The primary outcome is the per-state percentage reduction in cumulative teacher-forced forward KL against exact `H=0` FishBack at matched realized immediate effect. Zero-denominator states are excluded by the prospective floor `1e-8` and counted against coverage.

Secondary outcomes are per-offset KL, horizon action cosine, `G_0`-metric angle, immediate target retention, future target retention, target reachability, free-generation morphology rate and perplexity on a fixed 16-token diagnostic continuation, quadratic-versus-realized Taylor error, temporal-externality ratio, cache-only recovery, eigenvalue spectra, condition number, exact wall time, model forwards, Jacobian/JVP/VJP count, and peak memory.

## Uncertainty and strata

BCa intervals resample the 48 independent states 10,000 times with seed `20260906`; rollout trajectories remain nested inside their state. The primary horizon contrast uses Holm correction across `H=2,4,8`. The route requires at least two corrected contrasts to pass. Required strata are concept, base confidence quartile, source length quartile, layer, horizon, action-angle quartile, condition-number quartile, reachability, and rollout entropy quartile.

Quadratic-prediction correlation is computed across method-dose pairs within state and summarized by state-level Spearman coefficients before bootstrap. Offset concentration is the largest absolute paired KL reduction at one future offset divided by the total absolute future-offset reduction.

## Gates

- CB.1: algebra, full-vocabulary Fisher, replay, cache-mode, allocation, hashing, and serialization tests pass; pilot-test count is zero during selection.
- CB.2: PSD nesting passes every required state; at least two future horizons have median action cosine below `0.999` relative to `H=0` and a paired angle lower interval above the `1e-3` numerical floor.
- CB.3: at least two Holm-corrected horizon contrasts have a one-sided 95% lower interval above 15% cumulative-KL reduction.
- CB.4: immediate-effect-retention lower interval is at least `0.98`; reachability and matched coverage are at least 80% overall and within concept; offset concentration upper interval is below `0.80`.
- CB.5: Spearman lower interval exceeds `0.50`; median smallest-dose Taylor relative error upper interval is below `0.20`.
- CB.6: cache-reset removal lower interval is at least `0.80`, cache-only recovery lower interval is at least `0.50`, and `TER_H` lower interval is above zero on at least two horizons.
- CB.7: `H=8/H=0` wall-time ratio upper interval is at most `12`, peak allocated memory is at most 40 GB, and all derivative and pass counts are complete.

All gates pass or Pilot B closes or is explicitly narrowed to a local mechanism result. No instruction-tuned scaling, PathwayBack, joint branding, paper result, or SOTA claim is authorized by this freeze.
