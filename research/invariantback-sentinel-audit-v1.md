# InvariantBack sentinel audit, version 1

Audit date: 2026-09-06. Decision: Pilot A closes at the prospectively frozen mapping-comprehension gate. Development, powered pilot, multi-model scaling, combination with CacheBack, and paper promotion remain closed.

## Immutable data and isolation

The public packet is `artifacts/development/invariantback_v1_data/public.json` with SHA-256 `472cc968c8bce46e8e2a59c7cdd8bb99cae6b8c674bd6995d2be402dca0ce259`. It contains 80 construction, 8 sentinel, and 24 development groups per dataset. The separately serialized pilot packet has SHA-256 `540b71a5872d59c9bfc538223a450c606603d18faa2cafd43c22e96555497d37` and contains 48 groups per dataset.

All 480 group identifiers and all 1,440 endpoint identifiers are unique across allocations. Eligible pre-model-output pools contain 10,068 NormBank groups, 128,065 MNLI premise groups, and 37,391 SC101 action triplets. The source revisions, filters, construction size, allocation seed, row-order split, and held-out vocabularies were frozen in `benchmark-freeze-invariantback-v2.md` before materialization.

Only the public packet was copied to `avi-gn-fsk40`. The remote pilot path is absent. No development or pilot group was used to construct the QP-sentinel covectors.

## Replay and robust-QP sentinel

The one-cell Gemma replay smoke receipt is `artifacts/development/invariantback_v1_replay_smoke/receipt.json` with SHA-256 `1a89f82d8d6fae604bfcbc4adc54e60b9c60b85e9829cfc44db56fd0b1107e15`. A zero action gives maximum absolute logit error `0.0`; the semantic covector is finite with norm `0.14519`. Peak allocated memory is 20.04 GB.

The accepted eight-shard QP packet is `artifacts/development/invariantback_v1_qp_sentinel_attempt1`; its 51-file listing has SHA-256 `6edb2cbc7530c7c82ca4ff1c947b63dd343634aca05c6adb9fa420b4a1c8e220`. All shard receipts and every result/tensor hash validate.

- 27 of 27 dataset-contrast-layer cells pass and are hard feasible;
- all cells contain 54 construction-view covectors and have reduced rank 54;
- maximum zero-action replay error is 0;
- minimum normalized-view margin is 0.999999812;
- maximum complementarity residual is 1.23e-7 and stationarity residual is 0;
- total model forwards are 5,886;
- aggregate GPU-cell time is 441.04 seconds, the slowest shard wall time is 70.67 seconds, and maximum peak memory is 23.87 GB.

This passes the algebra, replay, and sentinel-feasibility premise only. The identity-metric sentinel is not the Fisher candidate and contains no steered performance comparison.

## Mapping-comprehension failure

The accepted eight-shard mapping packet is `artifacts/development/invariantback_v1_mapping_sentinel_attempt1`; its 24-file listing has SHA-256 `fd291149112be41b56a066f66913b6a27f420abdf61a11764693c1f1512fafa9`. All eight receipts and result hashes validate. The run uses 90 unsteered cells and 90 model forwards, takes 10.54 aggregate GPU-shard seconds, and peaks at 20.05 GB.

Eighty-nine cells pass. The sole failure is SC101 mapping `ok/bad/good` under held-out identifiers `one/two/three`. The cell accuracy is 8/9. For phrasing 2 and semantic label `bad`, the model predicts `ok`: correct `two` score -1.16237 versus incorrect `one` score -0.66237, a correct margin of -0.5. This is not a rounding or tie case.

The frozen protocol requires every one of the 90 cells to have at least 8/9 accuracy and every query to have positive mapped-identifier margin. Changing the verbalizer, dropping SC101, weakening the per-query margin requirement, or competence-gating after seeing this cell would be post-result rescue.

## Claim boundary and next research decision

Claim C52 is contradicted as the prospectively frozen Pilot A route because its necessary held-out mapping-comprehension condition fails before development. The result does not show that robust semantic synthesis is impossible; it shows that this frozen cross-dataset, cross-verbalizer design cannot identify semantic control on all declared held-out views with the pinned model.

CacheBack independently closed at development coverage and InvariantBack independently closes here. They are not combined under a joint method name. The existing local CacheBack mechanism evidence remains unchanged. A future route requires a new claim and a new pre-model-output freeze; it cannot reuse this failure to edit Pilot A.

