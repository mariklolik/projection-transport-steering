# Isolated eager engine: v7 failure and v8 qualification

Status: v7 is closed before GPU admission. All four prospective v8 stages complete successfully, and the root audit verifies the complete raw/compiler archive. This qualifies the declared isolated inference workload, not the static-fit composition or a method comparison. The complete baseline and controller remain unopened.

## What failed, and what changed

Two native CPU admissions failed in v7. First, the probe config interpolated an absent top-level model revision into `/model/snapshots/undefined`. A recorded pre-GPU path amendment fixed only that administrative error. Second, the actual producer rejected the original frozen input inventory: v4 hashes canonical JSON lists, whereas the new probe and its local test fixture both hashed raw NumPy bytes. Sharing the same wrong convention made the local green suite insufficient evidence of native compatibility.

Independent tokenizer-only reconstruction recovered every original prompt, identity, width and length across all 32 batches. Canonical-list input and mask hashes matched 32/32; raw-byte hashes matched 0/32. This is a serialization-contract failure, not token drift, numerical failure or a negative behavioral result. Both original failures and the original source archive remain. No v7 GPU stage started; its two CPU containers cost 26.72699762135744 seconds.

v8 was separately declared before GPU execution. Its minimal repair calls the existing `payload_sha256(tensor.tolist())` and corrects the fixture to the immutable inventory convention; the two positive first-forward tests fail before the producer repair and pass afterward. The stronger native CPU boundary executes the real producer against all 32 original input records and stops before real model forwarding. The native suite passes 105 tests with one CUDA skip. These are software checks, not GPU results.

The isolated alias `fp32_prefill_flex_eager` retains the original FP32-prefill attention callable. It clones the native mask function's code/defaults/closure and changes only its private global builder binding to native `create_block_mask` with compilation disabled. It does not change the shared Transformers mask function, masks, dtype policy, model, tokenizer, generator, scorer, cap or numerical tolerances. The alias is qualified only in the pinned runtime; its name is not a general Transformers capability guarantee.

## Prospective and observed gates

| Gate | Frozen workload | Observed status | Inclusive H100 seconds |
|---|---|---|---:|
| A | Original 35 geometries, 175 source-mapped FP64 comparisons | 35/35 helper returns pass | 67.927741841 |
| B | Original 32 batch-8 selection prefills, no sampling | 32/32 pass, all 256 source identities | 50.658860020 |
| C-profile | Same 16 short-prompt fit questions, batch 8, cap 512, original replay/action/cache checks | Complete pass; all 16 cap-censored | 164.259135166 |
| C-full | Same 16 questions and seeds, batch 8, cap 8192, no replay checks | Complete pass; 14 EOS and 2 cap failures | 669.565432562 |

A uses the original helper with the exact registered attention and mask bindings, highest FP32 precision, unchanged tolerances and no tensor observer. Its 175 comparisons are inferred from 35 normal returns through unchanged source; they are not 175 separately logged labels or independent statistical replicates.

B verifies each prompt/input/mask identity before forwarding real Qwen3-8B weights. All logits are finite BF16 tensors of shape `[8,1,151936]`; every cache length equals its padded input width, and base parameter versions remain unchanged. The 256 prompts contain 25,836 unmasked tokens and 55,384 padded token instances; 8/32 batches are shorter than 128. The synchronized first-forward sum is 32.44352578371763 seconds. Logits are not archived, so this is not retained-logit parity or a full base-weight byte-identity proof.

C-profile preserves exact raw/scored/source joins, prompts, tokenization, seeds, decode, EOS/cap accounting and original-grader replay. The saved zero and split token arrays equal the references. Every row is at risk at token 256 and advances 64 tokens through the nonzero path; 6/16 action suffixes differ from the reference. Identical remaining suffixes are not exclusions and do not refute path coverage. RNG equality and unchanged parent-cache keys/values are producer observations; no complete cache-metadata snapshot was saved. The reported action-forward count is width-derived.

All 16 profile responses reach 512 without EOS; both graders report parse failure and there are no evaluator exceptions. This does not estimate complete-answer correctness. Reference generation costs 60.0575875043869 seconds, checks 85.61989083141089, and grading 0.09476519376039505. Nested components are not added again to container time. Independent reviews pass 201 A, 408 B and 361 profile assertions, alongside the 173 predispatch source/argv checks; these check counts are not scientific sample sizes.

C-full retains all 16 questions and 65,653 emitted tokens, with 13 primary successes, 14 sensitivity successes, 14 EOS responses and two cap failures. No evaluator exception or row exclusion occurs. The existing full-condition auditor reconstructs raw/scored/source, seed, prompt, token, decode and original-grader joins through a temporary read-only file view. Full-generation replay/action checks were not requested; profile checks retain their shorter scope. A separately labeled post hoc diagnostic finds that all 16 full answers begin with exactly the profile's 512 tokens. It is not an added success criterion.

The one grader disagreement is `train:algebra:1486`: the output boxes `Sunday`, while the reference boxes `\text{Sunday}`. The pinned author normalization preserves the unequal strings; Math-Verify parses the day name as a product of symbols and reports equivalence. Both displayed answers name Sunday, but the registered primary score remains false. Do not relabel the row, replace the primary metric or describe this formatting discrepancy as demonstrated faulty reasoning.

All four v8 container lifetimes total 952.411169589 seconds (0.2645586582191667 H100-hours), leaving 2367.256645422 seconds in the separately recorded additional engineering allowance. v7's two CPU failures and v8's 115.94559252634645-second CPU container remain separate; neither consumes GPU time. Original engineering and baseline accounting are unchanged. The full run's reference rate is 100.8752 emitted tokens/s and its inclusive-container rate is 98.0532, not a measured speedup.

The frozen dense batch path performs 131,072 advance-step row slots for 65,653 emitted tokens: 65,419 slots follow row absorption. This is source-derived slot accounting, not a profiler estimate of wasted FLOPs or GPU utilization. Both batches contain one capped row, so both run all 8192 steps. It identifies a future scheduling opportunity, but does not authorize post hoc row reordering, cap reduction or active-row compaction in this frozen benchmark.

The first archive command failed because the copied CUDA-cache directories were root-owned mode0700. Its incomplete tar and error remain. Reading the existing copies as root, without recopying caches or changing permissions, produced a new complete tar. All 721 files and 135,918,174 content bytes match the native canonical inventory; all 26 separately copied raw stage files match that archive. The complete tar is 137,236,480 bytes, SHA256 `bbcd94e34cb06409001f2df89974ffb10c8638a04630bf62c89a5fa3055b8274`. No GPU retry, original-file overwrite or data repair occurred.

## Interpretation and next decision

The mechanism evidence is separated from qualification: the earlier v6 paired experiment found malformed automatic mask counts before attention; v8 tests a reusable isolated implementation. Neither establishes sole causal responsibility for every historical failure or correctness across arbitrary models, contexts and training compositions.

The v3 and v8 full-runtime packets share only one question. Their durations and correctness totals therefore cannot identify a speedup or accuracy change. v8's 16 questions are source-selected short-prompt fit examples, not held-out test data. No confidence interval from these engineering rows would establish benchmark generalization.

With C-full receipt, raw-output and archive verification complete, the next dependency is the changed static-fit/eager-mask composition, followed by a separately frozen full eight-candidate baseline recovery. Reuse the existing epoch loop, exact targets, accumulated optimizer, position-mask context, saved-state loader, generator and outcome analyzer. Do not implement another trainer or use only the three archival survivors. Preserve all 1,500 fit questions, 256 selection questions, 12 epochs, ranks 4/8, sites 17/23, prompt/full policies, seeds, unit strength, cap and primary metric. Only the needed registration/engine binding and an explicitly accounted bounded execution contract may change.

The independent source-only successor review identifies one registration-condition change in `run_irc_baseline.py`; no new trainer is needed. The existing old-mask/static FP64 gate covers B2 buckets 384/896/2816, the old-mask/static accumulated gate covers rank8/site17/full at384/896, and v8 A covers eager-mask/default compilation at B8 widths through558. These receipts do not establish their new static-plus-eager composition. The next bounded bridge should exercise that exact binding at the original training buckets and the accumulated/raw-checkpoint path with unchanged tolerances. Reuse prior normalization, target and checkpoint evidence at its original scope; do not demand another broad suite or claim all-rank/site/maximum-length accumulated coverage.

The baseline's old 7.605374254722222 H100-hours remain charged against its 48-hour allowance. A complete eight-worker five-hour ceiling would reserve 40 of the remaining 40.394625745277778 hours; this is budget arithmetic, not proof of affordability or launch permission. The changed-composition bridge and end-to-end cost still require separately recorded admission decisions. No baseline, response collection, development or sealed test is opened by this report.

C67, useful-adapter evidence, S5/S7, architecture/construct breadth and SOTA remain unpassed. Scientific strength stays 15/36; the paper is EVIDENCE_AVAILABLE but EVIDENCE_THIN. Submission prose and compliance promotion remain blocked by missing scientific evidence, not by writing quality.

## Evidence locators

- `irc-eager-engine-result-v7.json` and `irc-eager-engine-independent-review-v7.json`: immutable failure history and independent source reconstruction.
- `irc-eager-engine-protocol-v8.md`, config SHA256 `bccc5f0aa5e98fb1d1188cbc0053ab529fbdd9ecdfac42e892107f19f367af47`, and dispatch SHA256 `ce1c74de71d83eab67bbe19422583d360bee09176888d0e53c7dfb5ef79e4c69`: prospective workloads, exact commands and fail-closed budgets.
- `irc-eager-engine-predispatch-v8.json` and stage admission receipts: native CPU proof, preceding raw-packet gates and guarded launch code.
- `artifacts/engineering/irc_eager_engine_v8/{a,b,profile,full}`: complete raw native stage artifacts; `complete_archive` contains the separately extracted complete raw/compiler archive.
- `irc-eager-engine-root-audit-v8.json`, SHA256 `dba104f32cd2f4ee11c703fcfafc60a703e85e3215eb71124b8f65fe33cfa734`: exact root analysis/transport code, outputs, failures and grader diagnostic.
- `irc-eager-engine-archive-files-v8.jsonl`, SHA256 `97c65b499235eff50bbd15866c69708629a9c2068052113dd30cc0eb183340bc`: complete canonical native/local inventory. Final result, independent-review and closure receipts bind the human report and its scope.
