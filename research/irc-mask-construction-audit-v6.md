# Native mask construction v6: paired repair candidate passes the original operator matrix

## Decision

Changing only the native mask builder's `_compile` argument from true to false removes the observed mask-count and numerical failure on the fixed 113 → 2 → 93 history. The separately admitted observer-free run then passes all 35 original geometries and their 175 numerical comparisons. This is a successful numerical-engine prerequisite, not a validated steering method or a qualified production engine.

The next dependency is minimal durable engine integration and actual-weight qualification. Original v4 remains failed; its actual-weight Gates B/C and the eight-candidate baseline remain unrun here. Baseline-v2, all earlier scientific failures and v20 remain closed. No shortened cap, reduced candidate family, dependency upgrade or tolerance change rescues those records.

## What changed and what stayed fixed

The 88-line adapter in `scripts/run_mask_diagnostic.py` reuses the unchanged v5 observer/CLI, native Torch builder and HF mask function. It temporarily replaces only HF's builder binding, forwards every other argument unchanged and restores that binding on exit. Torch's function/global binding, the v4 attention callable, original helper, input generator and FP64 oracle are unchanged. This scoped single-process diagnostic context is not a thread-safe production integration.

A/B use the same early CPU metadata snapshot and late tensor capture, but separate fresh processes and cold caches. Both retain batch 8, H32/KV8/D128, training false, seed 20260908, BF16 inputs, the original FP32 prefill adapter, two forward modes and Q/K/V gradients. The fixed tolerances remain rtol 0.005 / atol 0.0005. No weights, outcomes or sealed questions are mounted.

Fourteen new tests were observed failing before implementation. Final focused local and exact-native CPU suites each pass 43 tests with one expected CUDA skip; critical Ruff checks and formatting pass. Independent implementation review passed 25 focused tests and 28 retention/restoration checks. These CPU checks are not native GPU numerical qualification. The complete repository suite was not rerun in this increment.

## A/B result and failure anatomy

| Arm and length | Native numerical result | Metadata at builder return | Change after attention/backward |
|---|---|---|---|
| Automatic Q113 | 5/5 pass | valid | none |
| Automatic Q2 | 5/5 pass | 15/32 counts exceed capacity | none |
| Automatic Q93 | both forwards/dQ pass; dK fails; dV unattempted | 24/32 counts exceed capacity | none |
| Eager Q113 | 5/5 pass | valid | none |
| Eager Q2 | 5/5 pass | valid, including fully masked row 0 | none |
| Eager Q93 | 5/5 pass | valid | none |

All three shared cells have exactly matched actual Q/K/V, upstream VJP, padding/dense masks, FP64 inputs and oracle output: ten fields per cell, checked by content hash, dtype, shape and stride. All nine reference-gradient pairings also match.

Automatic Q93 K has 761,650/761,856 tolerance violations and maximum absolute error 1199.5983157205703. Its saved V also fails a separately labeled native-image CPU reconstruction: 761,657 violations, maximum error 1695.9453173250301. The native helper stopped at K; that V result is not a second native failed assertion. Eager K/V have zero tolerance violations, with maximum absolute differences 0.015573456286689158 and 0.029624263699831843 respectively. These maxima need not be below the absolute tolerance alone: the original elementwise rule also includes relative tolerance. All saved values remain finite.

The A/B denominator is two planned histories, six executed geometries and 30 potential numerical comparisons: 29 attempted, 28 pass, one K failure and one unattempted V comparison. The independent unit is a fixed-seed process history, not each tensor entry or assertion. No population interval or behavioral superiority test is estimated.

The pre/post snapshots resolve one uncertainty left by v5: malformed counts are already present immediately after the automatic builder returns. Corruption exclusively after attention/backward cannot explain these observed snapshots. Q2 is especially informative: even all five numerical comparisons can pass with malformed compressed metadata. Predicate correctness, a successful forward and finite gradients therefore do not establish compressed-mask integrity.

The result still does not establish sole causal responsibility for the K failure. Construction policy changes compiled execution, allocation and timing together; the early observation also synchronizes execution. The completed C run addresses whether the added observer is required for the eager numerical pass, not every possible allocation or compiler mechanism. The source-bound omitted reduction bound from v5 remains a static implementation property, not a memory-sanitizer trace.

## Observer-free original matrix

After independent review of the actual A/B packets passed, `irc-mask-qualification-admission-v6.json` admitted C before dispatch. Fresh source/config hashes, GPU identity, absent container name and empty output were checked. C used the exact frozen command, unchanged helper and all original 35 ordered lengths, 2–558 tokens, with both early and late observers disabled.

The container exited 0: all 35 helper calls passed, mapping to all 175 original numerical comparisons, plus native predicate equality, finite QKV gradients and exact masked-query zeros. The receipt and complete cell log agree. Unlike A/B, C does not separately log comparison labels or compressed-metadata snapshots; 175 is source-mapped from complete helper returns. This does not validate all 35 compressed metadata arrays or arbitrary future shapes.

The C bootstrap explicitly wraps the unchanged matrix body in eager construction with `capture=False`. Running the config's inner body alone would not apply that policy; the exact dispatched bootstrap is part of the evidence. Peak Torch allocation was 4,669,630,976 bytes on the named H100, not an end-to-end serving memory claim.

## Cost and archive integrity

| Container | Terminal status | H100 container-seconds |
|---|---|---:|
| Automatic history | exit 1, native K failure | 46.797233546 |
| Eager history | exit 0 | 41.419271578 |
| Eager full matrix | exit 0 | 67.678780264 |
| Total | all three terminal | **155.895285388** |

Total v6 cost is 0.043304245941 H100-hours. Of its prospective 1,200-second reservation, 1044.104714612 seconds were unused. The additional engineering allocation has 3319.667815011 seconds remaining; cumulative expenditure is 280.332184989 seconds. The original engineering remainder of 9.198053592 seconds and baseline-v2's 7.605374254722222 H100-hours remain separate, with no transfer.

Container lifetime includes compilation, failed work and shutdown. It is not GPU busy time or end-to-end iteration wall time. Local implementation, review and archive transport also took time. A/B's complete immutable packet was independently reviewed on FSK while local transfer was pending; the completed local copy was subsequently hashed and safely extracted. No unfinished tar was inspected. No numerical, resource or packaging retry occurred in v6.

The diagnostic archive contains 515 files / 667,225,206 bytes; tar SHA256 `4472da82b29c5b53a9d06ed290dfa04ca0c2b8b2c67e692cab5337d334a8b806`. C contains 253 files / 65,129,857 bytes; tar SHA256 `bc63bd03a91b905aa663760aab8276a00513e94c18c63ed39ba938d462f51e09`. Both complete remote/local canonical inventories match. Original containers, results and cache files are retained; native permissions were not changed.

A closing check at 2026-09-08 02:42:52 UTC verified 61 unchanged producer sources, exactly three terminal v6 containers and all eight GPUs idle. This is a timestamped snapshot, not a continuing monitor.

One historical prose precision fix is explicit: the v5 report now says that all saved index entries are zero and each array has capacity one; it no longer calls out-of-capacity entries valid active indices. The old closure hash remains historical. Its original 26 bindings were verified before this correction; they are not claimed to match after it.

## Next research increment and practical lessons

1. Reuse the native eager builder and existing registration/runtime wiring in a separately declared engine. Do not ship the diagnostic observer or silently mutate a shared builder across concurrent calls. Bind the effective policy at the actual model call site and test restoration/identity.
2. Reuse original actual-weight B/C workload definitions and limits: all 32 source-ordered selection prefills with production first-forward kwargs, followed only on pass by the 16 fit-only question runtime with complete 8,192-token conditions and unchanged replay/EOS checks. Freeze any new source hashes and exact command before execution.
3. Only a fully qualified engine can justify a separately admitted complete eight-candidate baseline. Use the independent GPU lanes there; do not select the three archived survivors or interpret the failed grid's missing results as behavioral losses.
4. After a useful action is measured under the frozen selection rule, continue intervention-response collection and the planned controller/strong-baseline comparison. The full three-architecture/three-construct target, uncertainty, stratification and mechanism/protection checks remain required.

The efficiency lesson is to resolve a shared invalid engine once before parallelizing expensive fits. Archive transport and CPU review should overlap when immutable evidence is already available remotely. These are workflow decisions, not measured speedup claims. The research aim remains a positive, correctly interpreted steering result; this diagnostic is not proposed as the paper's contribution.

## Research-strength and skill receipt

The `research-paper-writing` router was reapplied for result closure: claim evidence → results and figures → research strengthening, with experimental-setup-writer for the prospective changed design. Non-Superpowers TDD and modern-python governed the minimal adapter. The results audit supports only the scoped counterfactual and original operator contract; the independent review is recorded separately in `irc-mask-independent-review-v6.json`.

The current empirical target remains EVIDENCE_AVAILABLE / EVIDENCE_THIN. S1–S9 on the 0–4 rubric remain 3,2,0,3,0,0,0,4,3 = **15/36**. Question/novelty anchors remain the reset benchmark/literature; source separation remains the data/CUDA audit; S3 remains blocked by missing actual-weight engine qualification; S5/S7 by the technically aborted baseline; S6 by absent admitted multi-family/construct results. S8/S9 retain adverse-result and reproducibility evidence. The numerical prerequisite improved without upgrading scientific efficacy.

Fair-strong, paper skeleton, section/global, full-paper review and submission gates remain blocked or not yet reached. Presentation is assessed separately as claim fidelity 3, density 3 and boundary visibility 4; it cannot offset absent scientific effects. No paper prose was produced. Responsible-author verification, disclosure classification and current ICLR compliance remain unverified.

Primary evidence is immutable native outputs and source-bound reconstruction. Reviewer interpretations and the next-step recommendation are distinct. The completed broad literature reset, original SOTA definition and benchmark are reused; no new novelty or freshness claim is made. `irc-mask-construction-result-v6.json` preserves exact commands, producer code, costs and archive verification; the closure checkpoint binds final document/ledger hashes and specialist receipts.
