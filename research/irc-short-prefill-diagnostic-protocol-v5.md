# Short-prefill diagnostic v5: shape history, not admission

## Decision and discovery

The previous turn made progress: immutable v4 Gate A failed on Q93 after Q113/Q2, and source inspection localized the reported tensor size to K/V gradients. C83 stays contradicted; the original v4 protocol and its failed receipt are immutable. The claim that compiler history causes the failure is unverified. Native reverse BlockMask materialization and an intrinsic Q93 kernel failure remain alternatives.

`tests/test_attention_backend.py::test_native_fp32_prefill_backward_matches_grouped_fp64` already supplies the exact fixed-seed inputs, native mask, FP64 oracle, output/no-grad/Q/K/V ordering, tolerances and fail-fast contract. Call this unchanged helper through runpy. `attention_backend.py`, native runtime files and scientific producers remain unchanged. Existing `materialize_outcome_score_data.file_sha256` supplies file hashing. The only new code is an external diagnostic runner and its CPU tests; no new attention, mask, optimizer, sampler, scorer or reference equations are needed.

The runner temporarily wraps real `torch.testing.assert_close` for calls from this exact helper. It delegates identical arguments first, records labels only for early successful comparisons, and captures caller locals only after final V success. On exception it obtains helper locals from the traceback after unwinding. Tensor copies, metrics and serialization happen only after the helper finishes. Restore the original function in all cases and release GPU/frame references before the next history geometry. No global tracing or profiler is installed. This limits but cannot eliminate observer timing, allocator and lifetime effects; a non-reproduction cannot prove the original failure was transient or repaired.

## Prospective design

Two planned arms, run sequentially on one freshly verified idle FSK42 H100:

| Arm | Fresh process sequence | Purpose |
|---|---|---|
| fresh93 | 93 | Test the failing shape without preceding compiled shapes |
| history113_2_93 | 113, 2, 93 | Test the original failing history with labeled telemetry |

Each arm has separate initially empty Inductor/Triton/CUDA caches; no cache reset within the history arm. Both use the frozen v4 backend, native image/runtime, B8/H32/KV8/D128, training=False, grad and detached no-grad forwards, seed20260908, rtol0.005/atol0.0005. Freeze runner/test/config/protocol hashes and commands before dispatch. Record compiler recompilation logs and retain terminal containers and cache contents. No model weights, prompts, labels, selection correctness or sealed-test data are accessed.

Save each attempted cell's inputs, upstream VJP, native/oracle outputs and available gradients, padding/dense masks, validity and exact native `kv_num_blocks`, `kv_indices`, `full_kv_num_blocks`, `full_kv_indices`, `q_num_blocks`, `q_indices`, `full_q_num_blocks`, `full_q_indices`, plus block sizes. Save a named row for each actually attempted assertion. Assertions not reached remain unattempted even if tensors permit later explicitly labeled analysis. Require exact Q93 Q/K/V, VJP, mask and reference hashes across arms before a paired conclusion.

The independent unit is one fixed-seed process sequence, not tensor elements or independent statistical replications. All conditions are exploratory diagnostics chosen after v4 failure. There is no accuracy endpoint, population confidence interval, tuning winner, serving benchmark or superiority claim. The complete denominator is two planned arms/four planned geometry calls, with five numerical assertions per call when reached. Masks, finite gradients and masked-query zeros retain the unchanged helper's additional assertions.

## Interpretation fixed before outcomes

- fresh93 passes and history Q93 fails, with paired inputs and matching reference: evidence of history-dependent numerical behavior under instrumentation, not identification of a particular compiler defect.
- both Q93 fail: intrinsic-shape failure remains plausible; history is not necessary in these two observations.
- both Q93 pass: original failure not reproduced under instrumentation; no repair or engine admission follows.
- fresh93 fails/history passes, history fails before Q93, missing tensors, differing Q93 hashes, runtime mismatch or timeout: ambiguous/incomplete diagnosis; retain every result.
- Audit compressed forward/reverse metadata by deriving expected block occupancy from each saved dense mask and its logical lengths. Q93/Q113 have one partial block per row and zero full blocks. At Q2, row0 has both keys masked and therefore zero partial blocks; the other seven rows have one partial block. Incorrect reverse materialization supports a narrower metadata defect; correct saved metadata rules out only that captured representation, not compiler consumption or later mutation.

No outcome reopens v4 B/C, admits v4, changes scientific caps or tolerances, resumes the three baseline survivors, or establishes C67/S5/S7/SOTA. A repair and a new full qualification would require a new source-bound decision; this protocol authorizes neither.

## Resource amendment and stopping

Under existing autonomous FSK authority, reserve at most600 container-seconds from the unspent3553.471520511 seconds of the separately allocated v4 engineering envelope. This is an explicit prospective reallocation to diagnosis, not an automatic qualification retry, new funding, transfer from baseline, or a user-approved numerical budget. At least2953.471520511 seconds remain unreserved before diagnostic consumption. Old engineering remainder9.198053592 and baseline-v2 cost7.605374254722222 H100-hours remain separate. B/C stay closed regardless of unused allowance.

Each planned arm has a300-second container-lifetime ceiling, a260-second child timeout and10-second forced-termination grace, reserving30 seconds for startup/shutdown. Charge actual total lifetimes including failures, compilation and serialization. No retry after any numerical, compiler, resource, packaging or timeout failure. A failure stops later geometries in that arm; the other independently planned arm is still attempted if transport, idle-GPU and remaining resources pass. Stop the increment after both terminal arms or an external/resource block. Never relabel an unlaunched arm as attempted.

Native CPU preflight and a fresh successful SSH/idle/name/source check are launch prerequisites. Current turn's initial SSH banner timeout does not authorize launching blindly. Local tests and prepared configuration do not imply native qualification or remote deployment.

## Skill and scientific boundaries

Route B: scientific-claim-evidence → scientific-results-and-figures → scientific-research-strengthening; experimental-setup-writer is invoked for this changed design, not submission prose. Reuse the completed literature/SOTA and benchmark contracts rather than repeat the survey. Scientific maturity stays EVIDENCE_AVAILABLE / EVIDENCE_THIN: C83 fails, no complete matched behavioral effect or strongest-baseline result supports S5/S7. Scientific prose, downstream section/global/review/compliance gates are not reached. Responsible-author verification and disclosure classification remain unverified. An independent critic reviews protocol/telemetry before launch and claims after results.
