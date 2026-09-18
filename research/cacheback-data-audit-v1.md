# CacheBack source-data audit, version 1

Materialization date: 2026-09-06. Status: valid frozen source allocation after one preserved invalid attempt; no sentinel or pilot-test model output.

## Attempt 1

`artifacts/development/cacheback_v1_data_attempt1_invalid_cross_concept_overlap/data.json` has SHA-256 `0331632dc623cb7b9668380fae4e131dea6d609121d5a72586634ba4da4793f3`. It contained the requested 81 concept rows but only 30 unique context hashes because a C4 prefix could qualify for several morphology concepts and was allocated independently. This created cross-concept dependence and possible development/pilot reuse. The packet is preserved and excluded from every analysis.

The root cause was per-concept duplicate tracking. A failing regression was added before the correction. The materializer now requires global uniqueness of both context hash and C4 document index and rejects any duplicate across concept or stage.

## Valid packet

`artifacts/development/cacheback_v1_data/data.json` has SHA-256 `bd7a72ba87d90e4d6df128ea3b60b66b49d2c14e4a7e85644fc4b44fc4f6e1ae`. It contains 81 rows, 81 unique context hashes, and 81 unique C4 document indices. It has zero overlap with every version-18 development and validation context hash.

The allocation is:

- third-person singular: 1 smoke, 3 sentinel, 8 development, 16 pilot;
- progressive: 3 sentinel, 8 development, 16 pilot;
- past tense: 2 sentinel, 8 development, 16 pilot.

The source starts at C4 validation document index 4096 and screened 320 documents under pinned dataset revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. Screening ran on `avi-gn-fsk40`, one H100 80GB, in 3.26 seconds. The materializer SHA-256 is recorded with the run manifest.

Decision: the 8 sentinel states may be opened. The 24 development and 48 pilot states remain output-free; pilot rows are not accessed by sentinel runners.
