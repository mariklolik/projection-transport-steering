# Claim-relative implementation discovery, version 1

Discovery date: 2026-09-06. Status: completed before new code.

## Existing components to reuse

- `amortized_pullback.py` already provides a replay-checked `GPT2Downstream`, exact explicit pullback geometry, a matrix-free pullback product, top-k softmax Fisher factors, and covector pullback.
- `amortized_pullback_runner.py` already provides pinned GPT-2 model loading assumptions, concept mappings, context-group identifiers, deterministic development allocation, model replay checks, target localization, per-method timing, memory accounting, source hashes, and shard receipts.
- `amortized_pullback_experiment.py` already provides concept probability, off-target KL, target localization, and trajectory helpers.
- `statistics.py` already provides deterministic paired and general BCa intervals with configurable alpha, resample count, and seed.
- `transport.py` and `torch_transport.py` already validate SPD metrics and implement exact metric-constrained updates for equality targets.
- The manifest, JSONL result, source-hash, and atomic-directory conventions used by versions 16-20 are the caller contract for new runners.

## Missing components

InvariantBack needs one small exact convex-QP primitive accepting an SPD metric, a matrix of view covectors, a common target, and an optional predeclared L1 slack cap. It must return the primal action, dual coefficients, margins, slack, objective, feasibility status, and KKT residuals. There is no existing multi-inequality solver or active-set interface in the repository.

CacheBack needs an autoregressive GPT-2 replay object that exposes one intervention state, fixed teacher-forced base tokens, and explicit cache modes. The existing `GPT2Downstream` ends at the same token and cannot represent future offsets or cache replacement. CacheBack also needs exact full-vocabulary representation Fisher and accumulation of nested horizon metrics. The current top-k helper is retained only for comparison and cannot define the new primary estimand.

## Wiring contracts

- New algebra lives in focused modules under `src/projection_transport_steering`; runners call it rather than duplicating linear algebra.
- Invalid shapes, nonfinite values, non-SPD systems, infeasible hard constraints, cache replay mismatch, non-PSD increments beyond tolerance, and incomplete receipts fail loudly.
- Comparator failure is a row-level status. It does not terminate other groups unless the failed comparator is required for the primary gate.
- Every output row carries model revision, source hashes, group ID, layer, method, encoding or horizon, target calibration, derivative counts, wall time, memory, and status.
- No version-20 solver, certificate, product cap, or partial packet is imported into either pilot.

## Minimal code decision

The first implementation slice is pure algebra only: robust SPD-QP and nested sequence-metric utilities with deterministic synthetic tests. Model hooks, dataset materialization, and GPU runners are separate later slices and are not justified until the algebra tests pass.
