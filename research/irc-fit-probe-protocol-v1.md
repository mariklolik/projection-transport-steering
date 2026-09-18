# IRC actual-weight fit qualification

2026-09-07. Prospective implementation and engineering design, before any actual-weight IRC adapter update. DRAFT_FROM_EVIDENCE; no scientific fitting or selection admission. The earlier runtime, benchmark, source-length and backward protocols remain immutable.

## Decision and reuse

Can the pinned Qwen3-8B runtime train only the upstream LoReFT parameters with the intended prompt/full-generation position contract, finite gradients, correct accumulated-token loss and bounded memory/time? A failed numerical, source, parameter-ownership or completeness check blocks grid fitting. Decreasing teacher-forced loss is not an efficacy gate and cannot select the recipe.

The code index finds the pinned `LoreftIntervention` and its tiny-model optimizer/roundtrip tests in `tests/test_reft_conformance.py`; `TorchLayerAction` owns post-block hooks; Transformers `DataCollatorForSeq2Seq` owns padding; installed `ForCausalLMLoss` owns shifted token loss and sum/`num_items_in_batch` normalization. Installed loss source SHA256 is `83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe`. Reuse these contracts. No general trainer, replacement objective, native attention kernel or package patch is justified.

The upstream data-module helper adds EOS by default and truncates. Its intervention-location helper encodes first/last prompt positions but not the proposed full-generation policy. Calling these helpers unchanged would not satisfy this packet. Add one small `reft_training.py` integration module for native-target construction and an explicit masked action; a bounded CLI wires it to the existing runtime and upstream optimizer. Leave all closed runners and the v3 runtime snapshot unchanged.

## Frozen target and position semantics

Use every admitted fit question and exact public solution; no selection, response, development or sealed payload. Construct the unchanged native Qwen thinking prompt and a native assistant message with exact solution in `reasoning_content` and the author-extracted boxed answer in `content`. Require exact prompt-token prefix identity and one assistant EOS. Retain native tokens through that EOS. Permit only tokenizer-decoded whitespace after EOS and omit that suffix from training; fail on another EOS or non-whitespace. This prospectively excludes a template separator, not any solution or terminal-answer token. The previous 508975/358243 counts describe the untrimmed template and must not be silently reused as the new training counts.

Mask prompt and padding labels with -100. Reuse the model's shifted causal cross-entropy with `num_items_in_batch` equal to the total number of supervised target tokens in the complete effective optimizer batch. Do not average unequal microbatch means. The final incomplete effective batch uses its actual target-token count.

Prompt-only applies one tied upstream action to the first seven and last seven nonpadding prompt positions, using `min(7, prompt_length//2)` at each end as in the upstream location helper. Full-generation applies the same prompt mask plus all nonpadding response input positions. The terminal EOS input has no supervised successor, so its intervention cannot affect this causal loss. At inference apply the same prompt mask during prefill; subsequently use the action only for the full-generation variant. This is a labeled Qwen/single-layer/full-generation adaptation, not exact reproduction of all-layer paper ReFT.

## Tests before implementation

1. Exact native prefix, one EOS, whitespace-only suffix, source preservation and overflow rejection without dropping rows.
2. Correct first/last positions under both left and right padding; distinct prompt/full response masks; invalid dimensions/values fail.
3. Only masked hidden positions change; unmasked states and gradients preserve their contract; shape mismatch fails.
4. On existing tiny Qwen3/Llama fixtures and actual upstream LoReFT, accumulated token-normalized gradients match one combined batch, model weights stay frozen, and only adapter parameters update.
5. Native cached prefill/decode and teacher-forced position policies agree numerically on the same tiny inputs, with independent model tolerances rather than a GPU bitwise portability claim.

## Bounded GPU measurement

After the full v3 packet's independent audit and the frozen synthetic native-backward pass, measure one rank8/site17 adapter on actual weights for each application policy. Use BF16 base/learned-source weights, upstream FP32 orthogonal rotation, eval-mode frozen base, zero dropout, AdamW learning rate0.0009, betas(0.9,0.999), epsilon1e-8 and weight decay0. Select three pairs of fit examples solely by the precomputed native target lengths: adjacent pairs nearest the median and p95, and the two longest. Store exact identities and lengths in the launch config before GPU dispatch. Right-pad each pair to a multiple of128, with no truncation and a4096 ceiling.

For each policy and pair run three forward/backward/optimizer steps, recording each separately. Label the first step first-in-cell: compiler caches persist within the resident process, so the second policy is not independently cold for an already visited shape. Reset the adapter to the same seed20260907 before each policy. No checkpoint from this diagnostic is eligible for the later baseline. Require finite nonzero adapter gradients, exact optimizer ownership, no base gradients, unchanged base parameter versions, at least one adapter parameter change and complete18-step accounting. Record loss, active positions, supervised tokens, observed dtypes/backend, per-step time, peak memory and source/config hashes. Parameter-version checks plus explicit optimizer ownership are not a cryptographic before/after hash of every base-weight byte. The probe performs one optimizer update per pair; the separate tiny-model accumulation test validates the upstream denominator, not an as-yet-unimplemented accumulated trainer.

Use one resident H100 worker, image/environment identical to v3, network disabled and read-only source/environment/model mounts. Deadline240 seconds plus20 seconds kill grace. Cold loading, compilation, validation and failed work count. This maximum260-second interval fits the original one-H100-hour engineering ceiling when added to the completed v1/SDPA, operator, profile, full and synthetic-backward intervals and the0.10-hour historical-overhead reserve. No automatic retry is authorized after a numerical failure or timeout.

## Next decision

Use this measured cost and complete long-generation accounting to freeze the eight-H100-hour baseline fitting/selection schedule before any baseline outcome. Preserve ranks4/8, sites17/23, both application patterns, all1500 fit examples and256 selection questions. More GPUs reduce wall time, not total allocation. Batch-size qualification or a prospective equal-budget screen may be needed to make the declared packet affordable; do not change the8192 inference cap or present a partial selection packet as complete. C67, useful positive full-generation action, response-control efficacy, strong-comparator and broader-paper gates remain unpassed.

## Independent prelaunch corrections

The source-only critic found two helper edges before any GPU update: Qwen also stops on151643, and its template strips edge newlines from supplied reasoning. The fit helper now counts the entire frozen generation-EOS set and fails if the exact source solution is absent from the assistant rendering. No source row is excluded or repaired. The whole1500-row ending audit must pass these stronger checks before launch. Noninteger/NaN prompt lengths fail the input contract. The cached-policy test uses two successive one-query decode calls, matching the actual runtime path. These corrections strengthen the prospective instrument without changing an observed scientific outcome.
