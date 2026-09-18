# Short-prefill v5: paired history effect and a malformed mask path

## Decision

The diagnostic made progress: identical Q93 tensors pass in a fresh process and fail after113→2→93. The native failing assertion is now explicitly **K gradient**. Saved forward and reverse block counts are malformed in the history arm. The archived symbolic mask-construction source contains a concrete missing reduction-bound defect. These results prioritize mask construction over an undifferentiated attention-kernel investigation.

No repair was implemented or admitted. Original v4 Gate A remains FAIL; B/C, the baseline rerun, C67/S5/S7 and SOTA remain closed. This is an exploratory numerical counterfactual on one fixed seed, not a behavioral experiment or independent statistical replication. The completed literature reset and full three-architecture/three-construct objective are unchanged.

## Execution, deviations and cost

The unchanged v4 helper/backend were called by an external observer. Early successful comparisons retain labels only; tensors are captured after final V success or exception unwind. Eleven local and eleven exact-native CPU tests pass; independent review also checked tensor-release behavior. The native preflight verifies58 source hashes,11 versions and three runtime-file hashes. Neither attention mathematics nor rtol0.005/atol0.0005 changed.

The original first Docker command was rejected before a container existed because the generated output mount duplicated an absolute path prefix. The second known-invalid command was not attempted. This is a recorded orchestration failure, not a numerical result. The original no-packaging-retry rule was not retroactively declared satisfied. An explicit reactive pre-execution amendment under the broader autonomous FSK mandate corrected only container names and complete output mounts, preserving both arms and the600-second reservation. An independent40-check argv audit and fresh actual-mount/path/permission checks preceded the corrected dispatch.

The two corrected containers used one H100 sequentially, separate initially absent caches, the same pinned image and no model-weight mount. All four planned geometry calls ran. Of20 planned native numerical assertions,19 were attempted:18 passed, one K assertion failed, and the following V assertion was unattempted. Plain mask/finite/zero assertions remain additional checks in the unchanged helper.

| Corrected arm | Native result | Container lifetime |
|---|---|---:|
| fresh93 | Q93: both forwards and Q/K/V gradients pass | 29.830838849s |
| history113_2_93 | Q113/Q2 pass; Q93 forwards/dQ pass, dK fails; dV assertion unattempted | 48.077581263s |
| Total | Both containers terminal; no retry or model run | **77.908420112s =0.021641227809 H100-hours** |

The original failed create consumed zero container-seconds, verified by the no-object inspection, not inferred from exit125 alone. The v4 engineering allocation now has3475.563100399 seconds unspent;522.091579888 of the diagnostic reservation was unused. Old engineering9.198053592 seconds and baseline-v2's7.605374254722222 H100-hours remain separate. The total v4-plus-diagnostic expenditure is124.436899601 seconds. Unused time does not authorize B/C.

Iteration wall time also included implementation, review, orchestration mistakes and archive transport; it is not represented by H100-hours. No serving-speedup, GPU-utilization percentage or end-to-end efficiency claim follows. A failed unprivileged archive and an inspection during an unfinished download were retained as failures; only the final complete archive passed verification.

## Paired tensors and failure anatomy

The actual Q93 Q/K/V, upstream VJP, padding/dense masks, FP64 inputs and oracle output have equal content hashes, dtype, shape and stride across arms. All three saved reference gradients are also identical. Input RNG mismatch and a different reference calculation therefore do not explain this paired difference.

A separately labeled post-hoc local CPU comparison finds both Q93 forward outputs bit-identical across arms. The Q gradients are not bit-identical:16726/3047424 entries differ, maximum absolute difference0.001953125, while both pass the same FP64 tolerance. Passing dQ is therefore not a bitwise-replay claim. Code and the computed detail are retained in the closure checkpoint.

The native K failure reproduces761650/761856 mismatches and maximum absolute error1199.5983157205703, the same reported error statistics as v4. The original v4 receipt still lacks its own K/V label and tensors; this new labeled observation does not rewrite it. A separate CPU check in the pinned native image finds the saved V gradient also wrong:761657/761856 mismatches, maximum absolute error1695.9453173250301. That is a post-capture check, not a second native failed assertion. All saved gradients remain finite, so a finite-only gate would miss the defect.

| Captured metadata | Fresh Q93 | History Q93 | Correct Q93 value |
|---|---|---|---|
| partial KV counts | eight1s | eight73s | eight1s |
| partial Q counts | eight1s | 124,123,…,117 | eight1s |
| full Q counts | eight0s | 116,117,…,123 | eight0s |
| full KV counts | eight0s | eight0s | eight0s |

All saved index entries are zero; each index array has capacity one per batch and direction. History Q2 also has malformed saved counts despite passing the numerical comparisons: partial KV19,19,18,…,13 and partial Q8,7,…,1. Its correct first-row count is zero because both keys are masked; other rows require one partial block. This prelaunch criterion was corrected by independent criticism before observing GPU data.

The observer captures metadata after the helper completes or fails. It does not measure initial materialization, intermediate mutation or exactly what each compiled kernel consumed. In particular, Q2's numerical pass with malformed saved metadata makes timing of corruption an unresolved question. A correct `mask_mod` predicate and a passing forward/dQ do not validate compressed metadata.

## Concrete source-level defect

The symbolic generated mask module `history113_2_93/compiler_archive/inductor/f2/cf2cubbrg22v2c2ej5zktrtire4aeivang66gpdtzxapq7qoqmgl.py`, SHA256806c3ab43a7d08d9998362289e2767eabec035fc25f711232361c317beb303ac, supplies the next source-bound target:

- Forward count reduction: lines 188/196/203 use R0_BLOCK=128, an all-true reduction mask and loads masked only by batch index. The caller at line 473 supplies logical reduction length ceil(KV_LEN/128)=1 for these short shapes.
- Reverse reductions: lines414/422/429 have the same omitted reduction bound; callers511/523 supply length1.
- CPU address enumeration for the exact short-shape strides yields1024 logical load positions per named reduction:8 intended,28 crossing into other rows and988 beyond the logical source tensor. These are not measured hardware transactions or a GPU memory-sanitizer trace.

The native HF builder unconditionally requests `_compile=True` in this Torch version; Torch's implementation invokes `torch.compile(create_block_mask)` without a dynamic override. Recompilation logs record the113→2 shape transition; archived symbolic source accepts those sequence lengths. This connects an actual source-level reduction defect to the failing path, but does not establish its sole causal responsibility or prove absence of a separate backward defect.

## Next increment and lessons

Do not modify or hand-patch generated Triton files. The next smallest counterfactual should vary **only native mask construction**, holding attention computation, input tensors, RNG, tolerances and shape history fixed. Preserve the native mask predicate and offsets; compare the automatic symbolic builder with an eager or explicitly shape-specialized native builder. Record compressed metadata immediately after construction and again after attention/backward. A candidate repair must then pass the entire original35-shape operator contract before actual weights or a full baseline are considered. This is a next research decision, not a GPU launch authorization in v5.

This failure extends the practical failure story: shape-history coverage and compressed-mask integrity are prerequisites that forward-only or single-shape checks cannot establish. GPU work should stay at the smallest operator until those invariants pass. Reuse the validated exact-argv/mount check before future launches; check completed transfers before hashing/extraction. Only a qualified engine justifies reopening the full eight-candidate baseline and parallel GPU work.

## Research strength and skill receipt

Route: scientific-claim-evidence → scientific-results-and-figures → scientific-research-strengthening, with experimental-setup-writer for the changed design. Claim/result audits support only the scoped observations and static source property; the strengthening increment resolved a concrete alternative at low GPU cost. Fair-strong scientific admission remains BLOCKED. Non-superpowers TDD and modern-python governed the minimal external observer; production/helper files remain unchanged.

Current empirical target rubric,0–4, before→after this diagnostic: S1 question3→3 (`benchmark-reset-v1.md`); S2 novelty boundary2→2 (`literature-reset-v1.md`, efficacy/priority still unproved); S3 trusted execution0→0 (v4 FAIL and current corrupted path); S4 selection separation3→3 (`irc-data-and-cuda-audit-v1.md`, no new question outcomes); S5 behavioral effect0→0 and S7 complete comparators0→0 (`irc-baseline-result-audit-v2.json`); S6 required scope0→0 (no admitted current IRC multi-family/construct result); S8 adverse integrity4→4 and S9 reproducibility3→3 (frozen failed/corrected dispatches, complete native archive and analysis). Total15/36, **EVIDENCE_THIN**. This is a current-target operational assessment, not a replacement for the historical source-paper rubric. Progress lies in a stronger diagnostic comparison and source-bound defect, not a higher scientific headline score.

Presentation is separately assessed as claim fidelity3, information density3, boundary visibility4; it cannot offset failed scientific gates. No submission prose, paper skeleton, section/global gate or compliance gate is promoted. Responsible-author verification, AI disclosure classification and ICLR compliance remain unverified.

Source precedence is immutable native packets/terminal state, deterministic CPU reconstruction, generated/native source, then this audit. The complete archive binds495 files/485529188 bytes; its tar SHA256 is0a58e890979372ee4ad49c098d1f9166b55b4c41742e467b5638bd2afcdb739e and remote/local canonical inventory SHA256 isbfa072acad4065d42a0520598e718343b41b38b59d98770f5f4eb8c69e3ac384. Exact terminal, analysis and source-audit records contain commands, code, hashes, exceptions, cost and the provenance boundary. AI-assistance ledger and independent closure locators are recorded in the matching checkpoint.

The independent evidence audit passed 741 checks: 667 tensor/provenance checks and 74 terminal/source checks. It reproduced the complete local archive inventory, saved-tensor comparisons, metadata defects, container accounting and source-address enumeration. This is an integrity and interpretation audit, not 741 experimental replications or a second GPU execution. Local tensor verification used CPU Torch 2.14 without new forward/backward computation; native-image CPU reconstruction is recorded separately. The evidence review does not certify this final Markdown file byte-for-byte or admit an engine, behavioral result or SOTA claim.
