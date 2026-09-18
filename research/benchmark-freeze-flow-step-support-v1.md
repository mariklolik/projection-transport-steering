# Flow-step support benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before flow-step-support output.

## Allocation and sources

Use only the 12 already exposed SemanticOutcomeBack/ResidualFlowBack design concepts and their byte-identical matched continuations. They are post-hoc design evidence. The next 24 held-in IDs `74,76,82,84,85,87,91,114,118,123,131,132,134,140,141,142,143,144,157,169,174,182,194,204`, the remaining 64 held-in concepts, all 100 held-out concepts, and all 9B allocations remain unmaterialized.

Use released Gemma-2-2B-IT FLAS at layer 20, flow time 2, checkpoint `N=3`, source commit `720ef8a67697d9b94130b374b5b3a1522a782566`, configuration SHA-256 `d5414215889017d76686bda35e00e4399ea7efa66815eceb322e18e3e7c7a46a`, and weights SHA-256 `bca45f7fa5abe11d607407b11ba0f00bdbf7936fa0a104b988cfac765446148e`. No learned parameter changes.

## Endpoints and analysis

For construction and held-out matched rows, report positive-minus-negative length-normalized continuation contrast relative to base. For the 16 neutral rows, report mean likelihood change and forward tokenwise KL relative to base. The independent unit is a concept. Fixed-seed 10,000-draw bootstrap intervals resample concepts. All 14 steered conditions, all failures, all subset effects, and exact three-step Shapley allocations are reported.

The candidate is equal-time late support `{1,2}`. It passes only if at least 11/12 concepts complete, every invariant and resource receipt passes, FS.2, FS.3, and FS.4 all pass, and no severe numerical or output failure is present. No other subset can advance. Failure closes version 1 without selecting another support, altering the `3/|S|` rescale, using a per-concept rule, changing flow time, weakening safety, or introducing a corrective action.

## Resource plan

Run one concept as an integration smoke, then six two-concept shards on six H100s. Each worker loads the base model and flow once. No gradients, new continuations, generations, or judge calls are needed. The expected full design cost is 45 teacher-forced base-model forwards per concept plus two exact replay calls per shard; each steered forward uses only the selected number of FlowBlock evaluations.
