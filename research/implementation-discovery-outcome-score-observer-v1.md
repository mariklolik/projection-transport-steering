# Outcome-score observer implementation discovery, version 1

Discovery date: 2026-09-07. Status: completed before code.

## Existing reusable surface

- `outcome_score_runtime.py` already owns prompt construction, deterministic rollout seeds, generation, strict scoring, and per-token layer traces.
- `outcome_score_sentinel.py` already owns frozen protocol lookup, allocation-bound row selection, receipt validation, and fail-closed promotion decisions.
- `run_outcome_score_sentinel_shard.py` already owns exact packet hashes, one resident model per worker, zero-action replay, trace serialization and reload, resource accounting, and receipts.
- `analyze_outcome_score_sentinel.py` already owns eight-shard discovery, row and trace hash checks, key joins, and finite trace validation.
- `statistics.py` already supplies group-resampling BCa intervals. `transport.py` already supplies empirical monotone transport and minimum-metric lifting.
- The immutable allocation packets already contain every authorized group. No source download, relabeling, or data materializer is needed.

## Minimum missing surface

The source pipeline lacks only protocol entries for basis indices 2-3 and fit indices 0-3, allocation-aware packet validation, and a source-stage aggregate. The observer pipeline later needs one basis constructor and two small causal-prefix model families under one grouped selection function. No generic training framework, hook system, tokenizer wrapper, evaluator, statistics package, or transport implementation is justified.

## Wiring

Extend the existing protocol registry with `basis_completion` and `fit`. The unchanged shard runner reads the stage-bound allocation hash and exact sorted group list, calls the existing rollout function, and emits the existing row, trace, replay, and resource fields plus allocation/source identity. The analyzer continues to verify every raw file before a source-specific summary checks expected groups, rollouts, classes, and progress support.

After the complete basis packet passes, the basis constructor consumes Stage C indices 0-1 and basis-completion indices 2-3. It emits center, ordered orthonormal basis, weighting receipt, source listing, and hashes. The observer fitter consumes that frozen basis and fit outputs only. It emits all fold predictions and metrics, the selection decision, and the selected refit artifact. Calibration and validation paths are absent from these callers.

## Caller contracts

Source-stage callers expect exactly eight pass receipts, the frozen packet and model identities, unique complete generation IDs, declared rollout sets for every declared group, exact trace-key equality, finite token-aligned traces, strict evaluator identity, exact replay, no cap or parse failure, and frozen resource limits. The aggregate fails on any missing, duplicate, unexpected, nonfinite, hash-drifted, or unsupported object.

Observer callers expect a 3,584-dimensional center, a 3,584 by 128 finite orthonormal basis with canonical signs, group-disjoint folds, complete checkpoint predictions, deterministic selection, and serializable finite parameters. A selected model must expose both online probability and its gradient with respect to the current layer-14 state under the same causal hidden-state contract used in training.

The existing source analyzer and accepted v2-v5 analyses remain regression fixtures. No new output is authorized until their byte-exact reproduction and the new RED/GREEN tests pass.
