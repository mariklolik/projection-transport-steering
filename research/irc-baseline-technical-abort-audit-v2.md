# Baseline v2: technical abort, not an efficacy result

The frozen eight-candidate comparison is technically incomplete. Two workers failed naturally with the same native compiler error; six were administratively stopped under the recorded resource-conservation decision. No candidate is admitted, ranked, or promoted to response collection. The scientific hypothesis is not refuted by this infrastructure failure.

## Terminal evidence

`irc-baseline-terminal-inventory-v2.json` binds the original Docker IDs, immutable mounts, GPU assignments, terminal states and every output-file hash. `irc-baseline-result-audit-v2.json` independently checks their dispatch lineage and charges all eight container lifetimes: **7.605374254722222 H100-hours**. The 48-hour stage ceiling was not exhausted; its unused allowance is not transferred to engineering or an automatic retry. The original engineering allowance has **9.198053592 seconds** remaining.

| Candidate | Exit / reason | Fit updates | Saved zero rows | Saved selection rows |
|---|---|---:|---:|---:|
| c00 | 143 / administrative stop | 0 | 16 | 0 |
| c01 | 1 / native compiler failure | 0 | 0 | 0 |
| c02 | 1 / native compiler failure | 0 | 0 | 0 |
| c03 | 143 / administrative stop | 0 | 24 | 0 |
| c04 | 143 / final-fit archival stop | 564 | 32 | 0 |
| c05 | 143 / administrative stop | 0 | 8 | 0 |
| c06 | 143 / final-fit archival stop | 564 | 32 | 0 |
| c07 | 143 / final-fit archival stop | 564 | 32 | 16 |

The 48 native output files total 213,153,439 bytes and have been copied locally with byte/hash verification. Four full natural-failure stdout/stderr logs also match the remote hashes. Raw and scored row counts are equal: 144 of 256 common-zero rows and 16 candidate-selection rows, not a complete paired comparison. Correctness values have not been summarized or used to choose survivors. All eight devices were idle at the terminal inventory snapshot; unrelated containers were not touched.

The prospective stop decision listed c02 among possible cancellations, but it had already failed before the stop operation checked it. Its actual classification is a natural failure, not a seventh administrative stop. c07 generated two eight-row selection chunks before cancellation was observed. These rows and their cost are retained; claiming that selection never ran would be incorrect.

The three adapter files are final 12-epoch raw states, not intermediate-selected checkpoints. Entering `selection/` proves that the source-ordered save, strict reload and saved-state equality path was reached. Administrative cancellation may precede the final frozen-base-version check and terminal receipt; none is a full worker PASS, an optimizer-resumable state, or a baseline-eligible winner.

## Failure mechanism and limits

Both failure receipts report `InductorError`: no valid Triton configuration, required kernel resource 528,384 bytes versus hardware limit 232,448. Docker reports `OOMKilled=false`. This is a per-kernel resource failure, not evidence that GPU global memory was exhausted.

The archived native source selects FlexDecoding when query length is below 128 and its other shape conditions hold. Grouped heads share a block, whose default row extent rounds `query_length * GQA_SHARED_HEADS` to a power of two. The failed generated source records FP32 Q/K/V, IEEE precision, `GQA_SHARED_HEADS=4`, `BLOCK_M=512`, `BLOCK_N=64`, `num_stages=3` and `SPLIT_KV=4`. The traceback and generated source support this specialization as the failure path. They do not establish a universally safe alternative or an end-to-end speedup.

Native source locators are retained under `artifacts/selection/irc_baseline_v2/runtime_source/`: Torch FlexAttention SHA256 `647de8711e39660127e712bff893fd18283d326d148ce25a36f87a2757651fb4`, FlexDecoding `f147ceca751a436a3ea9dd201b77ec7a50e92089e93be9f85e2b38a9e835cb4e`, HF integration `46655f4b8b528a75cdaa418e2fc6690fa4c2dd5e0180fe78900f4db0a7c83650`, and the generated failed kernel `cecc437f0516611d732aff47d9ae74b05d07a7f01e89f6be70eb792857ec1ec0`.

The earlier 16-question runtime pass remains true for its fixed workload. It did not cover every inference specialization encountered by the full selection allocation. Warming the training cache likewise did not qualify the dynamic short-prefill generation path. The practical failure was expanding to eight resident workers before covering the relevant shape-dependent compiler routes, thereby replicating costly CPU compilation while holding GPU allocations.

The native source-only inventory finds eight of 32 selection batches below the 128-token boundary. c01 and c02 start with widths 113 and 103, respectively, both rounding to a 512-row grouped-head tile. The saved zero prefixes of c00/c03/c05 stop immediately before their first short batch (widths 101/112/121). c04/c06/c07 have no short batch in their zero shares and complete all four. This alignment supports the shape-dependent diagnosis; the administratively stopped workers are not additional observed natural failures. Zero generation occurs before adapter construction, so the survival of three full-policy fits is **not an application-policy effect**. c07's 16 selection rows precede the first short batch at selection offset16.

## CPU-only archival analysis scope

The declared archival analysis reconstructs all 564 logged updates for each surviving final fit against the original 1,500 targets, epoch groups, learning-rate schedule and position policy. It reuses `audit_fit` with explicitly reconstructed expectations, not a fabricated native worker receipt. Its timing field uses the sum of recorded update intervals solely to satisfy the helper's lower-bound contract; it is nested update time, never full fitting duration. Actual checkpoint tensors are checked separately on CPU. The incomplete selection fragments are not regraded or ranked.

`irc-baseline-archive-fit-cpu-v2.json` passes that restricted contract for all three archives: 1,692 updates and 12,842,748 target-token instances over the same 1,500 unique training questions. All checkpoint tensors are finite. The 4,096-square FP32 orthogonal-parametrization base explains most of each roughly67-MB raw checkpoint; it is not 67MB of independently trained low-rank coefficients.

| Archive | Epoch1 token-weighted loss | Epoch12 token-weighted loss | Sum of update intervals, s |
|---|---:|---:|---:|
| c04, rank4/site17/full | 0.756879 | 0.487978 | 2603.190173 |
| c06, rank8/site17/full | 0.746379 | 0.466264 | 2623.215436 |
| c07, rank8/site23/full | 0.771146 | 0.491629 | 2146.884920 |

These training losses are descriptive within-epoch averages, not held-out accuracy or evidence for choosing a rank/site. Parameter ownership and the final base-version check retain their separate evidence limits. No optimizer/scheduler/RNG restart is supported.

`irc-baseline-partial-cost-decomposition-v2.json` accounts for 5,530.475958 saved generation seconds, 7,373.290529 logged update seconds and1.592228 scoring/check seconds within27,379.347317 charged container seconds. The14,473.988601-second residual includes compilation, loading, unfinished generation, collation and shutdown; it is **not a measured compilation time or GPU-utilization percentage**. Full container cost remains authoritative.

The baseline outcome analyzer itself passes 48 native-runtime CPU tests, including eight real tiny-model producers and the complete audit, plus a deliberately corrupted-score rejection. Native input preflight rechecks all 52 frozen producer hashes and the exact 1,500/256 allocations. The full local suite passes 462 tests with one CUDA skip. These are tool/contract results, not scientific outcomes. A first source-transport archive was rejected for macOS AppleDouble members before extraction; an eight-file metadata-free transport succeeded without changing a frozen producer.

The first CPU archival-analysis process exited0, but its tool-output parser rejected leading native warnings before the JSON result. A second identical CPU execution recovered the packet; no scientific configuration or GPU workload changed. Full CPU container lifetime was not separately instrumented. The independent critic subsequently verified352 identity, lineage, output-hash, coverage and archive assertions without reading correctness fields. Exact nanosecond lifetime reconstruction agrees with the main audit's declared microsecond resolution.

## Single next dependency

The pinned HF callable already forwards `kernel_options`; Torch already provides `FORCE_USE_FLEX_ATTENTION`. A narrowly scoped, separately named inference backend that forces regular FlexAttention only for FP32 multi-token prefill is a source-supported repair hypothesis, **not implemented or GPU-qualified here**. Do not lower precision, drop masks, change the 8,192 cap, tune the candidate product, replace questions or silently reopen v2.

A new GPU qualification requires an explicit prospective engineering resource decision. Before eight-way scientific dispatch, qualify short-prefill boundary shapes and the actual source-only batch-width inventory in one bounded worker, retain cold compilation and failures in its cost, and preserve existing numerical/EOS/branch contracts. Only a passed engine qualification can support a separately declared complete comparison. Archived fits do not authorize a three-candidate shortcut. C67, S5/S7, strong-comparator, multi-architecture/construct and SOTA gates remain unpassed.
