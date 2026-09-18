# Semantic-outcome source receipt, version 1

Receipt date: 2026-09-07. Status: admitted for one-concept smoke only; no model forward observed when this receipt was written.

## Model

Remote path: `/home/mekashirskiy/projection-transport-steering/.external/hub/gemma-2-2b-it` on `avi-gn-fsk40`. The downloaded directory has no embedded `_commit_hash`, so identity is defined by exact bytes:

| File | SHA-256 |
|---|---|
| `config.json` | `eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55` |
| `generation_config.json` | `a543a5d299bc2b20c52bd87ed174f561266510b57a392e12b5b5d758d798ce05` |
| `model-00001-of-00002.safetensors` | `532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf` |
| `model-00002-of-00002.safetensors` | `6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20` |
| `model.safetensors.index.json` | `ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a` |
| `special_tokens_map.json` | `baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351` |
| `tokenizer.json` | `3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922` |
| `tokenizer.model` | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |
| `tokenizer_config.json` | `cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89` |

The configuration declares Gemma2, hidden size 2304, 26 decoder blocks, bfloat16, and the recorded Gemma chat template. The smoke layer is block 20.

## Data

The admitted sentinel packet is `artifacts/source/semantic_outcome_v1_sentinel.json`, file SHA-256 `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`, content SHA-256 `e7f78438c7d1ee7dfb7f39d52722853e93f4614eb83960ae312d1efbfbc47464`. It contains 12 concepts, eight construction and eight disjoint latent-evaluation positives per concept, 16 global neutral negatives, and 208 unique source IDs. The exact upstream file hashes are embedded and match the benchmark freeze.

The AlpacaEval prompt file has SHA-256 `d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51`. Only tuning input ID 0 for concept 1 is authorized in the smoke. No AxBench test prompt is scored.

## Runtime and capacity

`avi-gn-fsk40` reports PyTorch `2.9.1+cu128`, CUDA runtime `12.8`, Transformers `5.16.1`, pandas `3.0.5`, and SciPy `1.18.1`. At the capacity check all eight H100s reported zero allocated memory and no compute processes. `avi-gn-fsk44` is excluded because all eight cards are occupied by another user's Blender jobs. Capacity is volatile and must be rechecked immediately before launch.

## Gate

This receipt authorizes exactly one concept-1 smoke on one idle H100. It does not authorize the 12-concept sentinel, development, official judging, performance language, or SOTA. Source mismatch, occupied GPU, zero-action replay error, nonfinite gradient, failed hard QP, KKT failure, empty generation, or incomplete receipt fails closed.
