# Short-prefill v4: failed numerical gate and bounded diagnosis

## Decision

Gate A is **FAIL**. The v4 callable is not admitted; Gate B, Gate C and a scientific baseline rerun remain closed. No numerical tolerance, generation cap, candidate product or earlier result changed. No further GPU job was launched. This failure concerns engine qualification, not the efficacy of intervention-response control.

The `research-paper-writing` route remained evidence-first: claim/evidence and result/cost audits preceded the next engineering decision, with an independent critic. Maturity stays EVIDENCE_THIN. C67/S5/S7/SOTA and the three-architecture/three-construct requirement remain unpassed. The completed literature reset remains the scientific design reference; this turn did not redo it.

## Frozen execution and accounting

`irc-short-prefill-qualification-protocol-v4.md` prospectively allocated one additional H100-hour under the autonomous FSK authority. It did not claim a user-approved numerical budget or transfer unused baseline/old-engineering time. Its Gate A used35 ordered geometries and required175 numerical comparisons, native masks, B8/H32/KV8/D128, module.training=False, fixed seed20260908, grad-enabled and detached no-grad forwards, and rtol0.005/atol0.0005.

The wrapper adds the existing native `FORCE_USE_FLEX_ATTENTION` option only for1<Q<128. Caller options are copied; original mask, dtype and gradient contracts and the old callable body are retained. The new alias is separate. Existing scientific producers were not changed or launched. Thirteen tests failed before implementation and passed afterward. The final native CPU preflight passes18 tests with one CUDA skip,11 package versions, three native runtime-file hashes and four new source-file hashes. All52 old immutable snapshot sources remain unchanged; the new snapshot differs in the one intentional attention module.

The full local suite reports475 pass/one CUDA skip. It started before the final GPU-only no-grad oracle extension, so it is not a CUDA qualification or an exact-final-file full-suite claim. The final source received its separate native18-test CPU preflight and the actual Gate A execution.

Container `5a84065016bf3ac05ee6f49049813a4ea9427b0f59cdd7187b3f24262aa1b635` on FSK42 GPU0 ran from00:26:13.174443039 to00:26:59.702922528 UTC on2026-09-08. It exited1, without Docker OOM or timeout, after **46.528479489 container seconds =0.012924577636 H100-hours**. The additional allocation has3553.471520511 seconds unspent, but B/C are closed; the old9.198053592-second remainder and baseline-v2's7.605374 H100-hours remain separate.

## What passed and what failed

| Geometry in fixed order | Observed cell result | Meaning |
|---|---|---|
| Q113 | pass | All five numerical comparisons and associated checks passed |
| Q2 | pass | All five numerical comparisons and associated checks passed |
| Q93 | fail | One K/V gradient comparison fails; exact K versus V label is unavailable |
| Remaining32 geometries | not attempted | No missing cell is counted as a pass |

The Q93 error reports761650/761856 mismatches, maximum absolute difference1199.5983157205703 and infinite relative error at an expected-zero entry. Output/Q-gradient tensors each contain3047424 elements; K/V gradients each contain761856. By the frozen assertion order, both output comparisons, masked-query zeros and the Q-gradient comparison passed before the K/V assertion. Those are **source-order inferences**, not separately emitted measurement rows. The receipt's10 comparisons in fully passed cells stays unchanged; it does not claim13 or14 directly recorded passes.

The harness preserved the exception text but not the inner traceback, per-comparison label or failing tensors. Consequently, K versus V and actual internal buffer values cannot be recovered from this receipt. The failed cell's0.041 seconds is detection wall time without the exception-path final synchronization, not a kernel-speed estimate. Peak allocated memory442632704 bytes is a synthetic-operator figure, not model VRAM use.

## Deeper source-only diagnosis

The complete338-file Inductor/Triton cache, two native result files and two CPU artifacts are copied and hash-verified:342 files,35747155 bytes. `irc-short-prefill-archive-files-v4.jsonl` binds every remote/local file. Original generated sources are preserved unedited.

The archived graph has a staticQ113 backward and a symbolic backward generated forQ2. The latter's generated benchmark fixes the initial lengths to2 while its actual wrapper accepts symbolic Q/KV strides and a symbolic launch grid. The Q93 failure is consistent with reuse of this path; graph-selection telemetry was not retained, so compiler-history dependence is not yet demonstrated.

Both grad-enabled and detached no-grad calls retain the first autograd graph. The wrapper creates fresh FP32 casts and has no explicit input mutation. No Python-level lifetime defect is established.

The generated backward separates dQ, which uses forward KV block metadata, from dK/dV, which uses reverse Q block metadata. A correct `block.mask_mod` predicate does not independently prove correct `q_num_blocks/q_indices/full_q_*` materialization. Reverse metadata and DK/DV launch/store behavior are therefore concrete diagnostic targets; neither is asserted to be the root cause. At Q93 the dense reference implies one partial block in both directions for every batch row and zero full blocks. Actual GPU block metadata was not saved.

An independent CPU FP64 check of the existing grouped-oracle equations against native SDPA with explicit repeated KV heads passes output and all three gradients at the sameB8/H32/KV8/Q93/D128 geometry. The largest difference is1.60e-14. This uses newly CPU-generated inputs and local Torch2.14.0, not the failed H100 tensors or pinned CUDA runtime. It supports the oracle algebra and does not overturn Gate A.

A bounded primary-source search found related upstream gradient reports, but no verified matching fix. [PyTorch issue158212](https://github.com/pytorch/pytorch/issues/158212#issuecomment-3911714813) traces its failure to negative floor division in a mask; this native causal-plus-padding mask has no such operation. That issue is not treated as our diagnosis or permission to update dependencies.

The independent critic passes23 integrity checks and confirms the K/V attribution limit, exact cost and absence of an established Python mutation defect. It identifies reverse block metadata as a distinct alternative to compiler-buffer aliasing. The archive and CPU algebra checks are separately reproducible.

## Preserved preparation and next decision

CPU-only source preparation also verifies all32 original selection widths and derives a16-row fit-only packet by filtering stored fit order for1<prompt_tokens<128 and taking the first16. There are1237 eligible fit rows; the two batches have widths119 and111. No outcome or difficulty filter is used. This is source preparation only: B/C conditions and deadlines were never dispatched or admitted after failure, and the derived rows are not new independent scientific observations.

The next diagnostic, if separately declared, must distinguish an intrinsicQ93 failure from shape-history/compiler reuse and from reverse-mask materialization. It should preserve the failed oracle and tolerances, label each output/Q/K/V comparison, save input/gradient tensors and forward/reverse sparse metadata, and compare freshQ93 with the declared113→2→93 history. This is a proposed **diagnostic counterfactual**, not an automatic retry, engine admission, or a way to replace the failed receipt. Its exact contract and resource decision must precede any GPU work.

Do not switch to a forward-only acceptance rule, admit archived baseline survivors, assert a general FlexAttention bug from this one sequence, or claim a useful steering method. The next scientific experiment remains the complete matched baseline only after a separately valid engine decision.
