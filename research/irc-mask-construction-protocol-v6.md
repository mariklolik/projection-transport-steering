# Native mask-construction counterfactual v6

## Decision and discovery

The preceding goal turn made progress: v5 paired identical Q93 inputs across fresh and 113→2→93 processes, exposed a native K-gradient failure and malformed saved metadata, and located omitted reduction bounds in generated mask-construction source. The current v5 checkpoint and all 26 bound files matched at this turn's start. C84/C85 remain narrowly supported; v4 qualification and scientific efficacy remain failed or unpassed. The complete three-architecture/three-construct SOTA objective is unchanged.

The reviewer-relevant prerequisite is trustworthy execution of the matched baseline. The immediate question is whether changing only native mask construction removes the observed compressed-metadata and numerical defect. A separate backward defect and observer/allocator effects are alternatives. This is not a proposed novel steering method or a replacement scientific endpoint.

Reuse `scripts/run_attention_diagnostic.py` for immutable tensor packets, named numerical comparisons, fail-fast history and source/runtime validation. Reuse the unchanged `tests/test_attention_backend.py` helper, FP64 oracle, tolerances, input generator and masked-query checks. Native HF already implements padding, offsets and the predicate; native Torch already implements eager block construction. Do not duplicate either implementation.

The only new code is `scripts/run_mask_diagnostic.py` and its CPU tests. A scoped context temporarily substitutes HF's `create_block_mask` binding with a delegating wrapper. Automatic mode preserves every argument; eager mode changes only `_compile=False`. The original Torch function itself and its global binding are not patched. Guard policy, helper identity, call count and restoration on exceptions. This is a single-process diagnostic context, not a thread-safe production integration.

For the two diagnostic arms, copy all eight compressed-metadata tensors into CPU lists immediately after the native builder returns. Reuse the original late-capture tensor packet for the after-helper snapshot. The same early observation is applied in both arms. It synchronizes and changes timing/allocator state, so failure disappearance is ambiguous, not a repair. Metadata validity is analyzed after the arm, not used to truncate a numerically passing Q2 before Q93. Missing or failed capture is a technical failure. Runner `pass` denotes its numerical/capture contract, not mask validity or engine admission.

## Prospective experiment and admission

| Stage | Fresh-process workload | Container ceiling | Decision |
|---|---|---:|---|
| A | Automatic native mask construction; 113, 2, 93; pre/post metadata | 300s | Contemporary instrumented comparator |
| B | Eager native mask construction; 113, 2, 93; identical observation | 300s | Paired construction-only counterfactual |
| C, contingent | Eager mask construction; all original 35 ordered v4 geometries; no early/late tensor observer | 600s | Original 175-comparison operator contract, not full engine admission |

A and B each execute once on one freshly verified idle FSK42 H100 with separate cold Inductor/Triton/CUDA caches. A numerical failure stops subsequent geometries within that arm; the other independently planned arm still runs if external state and remaining resources permit. No cache resets within an arm. All source and runtime hashes, exact commands and paths are frozen before dispatch. No model weights, question text, labels, outcomes or sealed data are mounted or opened.

Both diagnostic arms keep B8/H32/KV8/D128, module training=False, both grad and detached no-grad forwards, the v4 attention callable, seed 20260908, BF16 inputs with the unchanged FP32 prefill adapter, FP64 reference, rtol 0.005 and atol 0.0005. No global static-attention switch, generated-source patch, dependency upgrade or precision relaxation is allowed.

The primary diagnostic estimand is the paired difference in native numerical validity and compressed-mask validity under these two construction policies. The independent unit is a fixed-seed process history; tensor entries and assertion counts are not replications. The denominator is two planned histories, six planned cells and 30 potential native comparisons. No population confidence interval, multiplicity-adjusted superiority test or behavioral effect is estimated. Report all reached and missing comparisons.

Before interpreting a paired difference, require exact hashes, dtypes, shapes and strides of actual Q/K/V, upstream VJP, padding/dense masks, FP64 inputs/reference output and equality of reference gradients for every reached shared cell. Compare all eight pre/post compressed tensors with dense-mask-derived block occupancy and active indices. For these short shapes each direction has one index slot per batch: Q113/Q93 require one partial and zero full block; Q2 requires zero partial blocks in fully masked batch row 0 and one elsewhere. Counts must fit capacity. Only in-capacity active indices are interpretable. Saved inactive entries do not describe kernel accesses.

Stage C may open only after independent evidence review confirms: B completes all 15 native comparisons; B's pre/post metadata is correct and unchanged; all reached shared cells are paired; and A contains at least one numerical or metadata failure on a shared cell. An A failure before Q93 limits the history-specific inference and must remain visible. If both arms pass every numerical and metadata criterion, if pairing/capture is incomplete, or if B fails, the causal comparison is ambiguous or adverse and C stays closed.

For C, reuse the original v4 ordered lengths and unchanged helper directly inside the tested eager-construction context with capture disabled. This removes the additional v6 observer while preserving the mask-only intervention. Require all 35 geometries and 175 original numerical comparisons, mask predicates, finite gradients and exact masked-query zeros. One failure or missing cell fails C. Preserve complete compiler logs and caches; do not substitute selected successful shapes. No C result rewrites v4 or opens actual-weight B/C gates or a baseline rerun. Those require a separately declared successor decision.

## Implementation and verification order

- Write CPU tests for policy validation, exact argument preservation, real eager padding/offset semantics, immutable before/after snapshots, missing/duplicate-call rejection and restoration on native/helper exceptions. Observe expected RED failures before implementation.
- Implement the smallest scoped context and diagnostic adapter. Reuse the existing CLI rather than copy its parser, loops or serialization.
- Run focused local tests with the existing environment (`uv run --no-sync pytest`) and critical Ruff checks. Preserve local/native runtime distinctions; do not update dependency locks.
- Obtain independent source/design criticism. Package an isolated copy of the frozen v5 source with only the new files/configs overlaid; preserve v4/v5 snapshots.
- Verify the exact native CPU tests, runtime versions and five native source hashes, actual command mounts, permissions, unique absent names, empty cache/output paths and idle GPU. A ready package is not a launched experiment.
- Execute A/B, retrieve terminal evidence and verify the completed archive before analysis. Independently review pairing and metadata before deciding C. Archive and report C if admitted.
- Append all attempts and AI assistance; update claims, failure history, strength assessment and the next decision without promoting scientific prose or SOTA.

## Resources and failure policy

Prospectively reserve at most 1,200 container-seconds from the existing additional-engineering remainder of 3475.563100399 seconds under the user's autonomous FSK authority. This is an assistant allocation, not a user-approved numerical budget, extra funding or transfer from baseline. The original engineering remainder of 9.198053592 seconds and baseline-v2 expenditure of 7.605374254722222 H100-hours remain separate. Only actual lifetimes are charged; unused reservation does not admit a later stage.

A/B child timeouts are 260s and C's is 560s, each with 10s termination grace and 30s startup/shutdown margin. Count compilation, failed work, serialization and shutdown. No numerical, compiler, resource or timeout retry is allowed. At most one packaging-only correction is allowed before a container is created and before numerical exposure, after preserving the failed argv, independently checking the exact corrected argv and rechecking mounts; it consumes the same reservation. Do not relabel a compiler failure before the first comparison as packaging-only. SSH observation timeout is not terminal state and never justifies duplicate launch. Unrelated jobs are never stopped.

## Research and skill boundaries

Route: `scientific-claim-evidence` → `scientific-results-and-figures` → `scientific-research-strengthening`; `experimental-setup-writer` specifies this changed design, not submission prose. Non-superpowers TDD and modern-python apply to code; the research protocol supplies the implementation plan under the explicit autonomous/no-superpowers instruction. Generic templates requiring Superpowers workflows, additional approval loops or unsolicited commits are not used.

The completed literature/SOTA/benchmark reset is reused with its existing scope and unresolved novelty boundary. No new steering novelty claim or broad survey is needed for this numerical prerequisite. Empirical-target strength before this increment remains S1–S9 = 3,2,0,3,0,0,0,4,3 (15/36; source anchors in the v5 audit), EVIDENCE_THIN. Presentation is separate. Fair-strong admission, paper skeleton, section/global checks and compliance remain closed. Responsible-author verification and AI disclosure classification remain unverified.
