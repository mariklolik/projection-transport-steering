# Outcome-score transport implementation discovery, version 1

Discovery date: 2026-09-07. Status: completed before code.

## What already exists

- `splits.py` supplies stable content-derived group IDs and deterministic group allocation.
- `transport.py` supplies the fitted empirical quantile map and the existing batch minimum-metric lift.
- `statistics.py` supplies paired and resampled BCa intervals.
- `torch_runtime.py` supplies model-layer resolution, additive actions, layer hooks, same-forward controllers, exact replay checks, and runtime accounting.
- Existing experiment runners already implement atomic shard outputs, source hashes, immutable receipts, one model load per worker, and fail-closed merge checks.
- Existing Fisher experiments contain pooled metric construction and regularized linear solves; those primitives must be called or minimally generalized, not copied.

## Missing minimum surface

Only these pieces are not already represented by a reusable contract:

1. exact AIME answer extraction and correctness with adversarial parser tests;
2. deterministic materialization of the frozen AIME split packet;
3. causal prefix feature capture and one prefix-value model whose training and query inputs match;
4. a per-row minimum-metric score increment using an already fitted metric;
5. an outcome-score controller that composes the existing transport, hook, metric, and receipt APIs;
6. one shard runner and one analyzer for the frozen stage sequence.

The scorer is deliberately AIME-only at first. MATH-500 remains unopened until a pinned official symbolic-equivalence scorer passes its own source and environment audit.

## Wiring and caller contract

The data materializer writes immutable JSONL rows keyed by source commit, source row ID, content hash, and allocation. The rollout worker consumes only an explicitly listed allocation and writes raw completion, extracted answer, correctness, activation trace metadata, model revision, decoding configuration, counts, timing, and atomic receipt. It must not discover or load evaluation paths implicitly.

The prefix fitter consumes only `basis`, `fit`, and `calibration` receipts. It serializes basis, observer, calibration, transport, and metric identities. The controller is constructed from that frozen artifact and a declared method ID. It returns either an additive action or an explicit no-op plus a reason. The analyzer joins only exact keys and fails on duplicates, missing rows, hash drift, nonfinite values, or undeclared methods.

No new source loader, interval implementation, generic hook framework, or duplicate transport class is justified.
