# IRC mask-preserving runtime qualification v2

2026-09-07. Prospective engineering follow-up to the immutable v1 full-budget timeout. The previous goal turn was progress, not a repeated external blocker. The central IRC efficacy claim remains EVIDENCE_THIN and unverified.

## Decision and discovery before code

Can the installed Transformers FlexAttention/GQA implementation execute the unchanged real-question workload with correct masks, within-condition replay, isolated parent cache, complete output accounting and acceptable total cost? A backend may not be admitted from operator speed alone.

The installed public `flex_attention_mask` converts the native 2D padding mask plus causal offsets to BlockMask. `flex_attention_forward` enables grouped-query attention for Qwen3's power-of-two query-head count and calls the existing compiled wrapper. Reuse these functions; do not write an attention kernel, replace their masking logic, register an unknown backend or modify installed packages. Exact inspected SHA-256 values are flex integration `46655f4b8b528a75cdaa418e2fc6690fa4c2dd5e0180fe78900f4db0a7c83650`, masking utilities `c159cd91c2a7fcafce04a8b6cbca55c320ce904b8ebf634383c97da5d9313ce3`, and SDPA integration `53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f`.

The CLI already accepts a hash-bound configuration containing `attention`, but hardcodes SDPA in `from_pretrained`. The only production change needed is to pass the existing field, defaulting to SDPA for older callers, and record the actually loaded model configuration in the receipt. Extend the existing tiny-model CLI test across SDPA/eager; no mocked model or new loader abstraction. Preserve the old executable in the already frozen remote v1 snapshot and a local evidence snapshot before editing. Existing branch state, sampler, hooks, math scoring, data allocation and raw-before-check persistence remain unchanged.

## Prospective sequence and stop rules

1. Synthetic operator qualification: Qwen3-8B-shaped bfloat16 attention, batch8, query heads32, KV heads8, head dimension128, query length1, key lengths512/2048/8192. Test all-true, left-padding0..7 and left-padding plus trailing absorbed-row mask patterns. Use the installed mask builder and attention callable. Check the actual BlockMask predicate against the exact expected Boolean mask. Also check causal prefill predicates at length128 with left padding and absorbed tails.
2. Check zero-query masked means and random-query outputs against an explicit grouped FP64 softmax oracle. The predeclared pointwise tolerances are rtol0.005/atol0.0005, unchanged from the prior numerical diagnostic. Preserve every failed comparison, complete remaining cost cells if safe, and require all semantic checks to pass before actual-weight Flex qualification. Numerical disagreements with old SDPA are not silently equated with oracle error.
3. Record first-call mask/compile/attention wall time per cell, 32 warm operator calls, eight warm mask-plus-operator calls, and temporary peak memory. No repeated-run CI or whole-model speedup is inferred. No weights, task outputs or selected adapter enter this operator job.
4. If and only if operator semantics pass, run the existing real-question runner under a new source/config/runtime receipt with `attention=flex_attention`. Reuse all16 fixed runtime fit questions, their hash order, native prompt and tokenizer, per-row seeds, temperature0.6/top-p0.95/top-k20, bfloat16, EOS/pad, two intra-op threads and one inter-op thread. Run batch8/512 with all original identity/split/cache checks, and separately batch8/8192 with complete output scoring. Only batch8 is being qualified in this version; the v1 batch1/4 receipts do not apply to Flex.
5. Each condition has its own detached, deadline-bounded container and exclusive output directory. Full workload still requires16/16 unique expected IDs, all scorer calls/errors recorded and a final pass receipt. Retain wrong, capped and partial outputs. No question replacement, mask removal, cap reduction, grading retry or v1 overwrite. A failed short conformance condition prevents admission even if the separate full workload finishes.
6. No baseline fit, response, selection, development or sealed test is opened by the operator job. After complete runtime admission, freeze actual-weight fit-step cost and the exact baseline recipe before fitting.

## Runtime and budget

Use the unchanged isolated Torch2.9.1+cu128/Transformers5.16.1/Pyvene0.1.8 runtime and immutable image `sha256:edc40b8cd5d95f053f05b0f7807e256038cd604c08aee981535c8e44d72180cb`. Python -S and the existing read-only overlay remain mandatory. Add explicit writable temporary TorchInductor/CUDA cache paths and limit compilation workers to2; compilation belongs to measured cold process time. Do not use the rejected native Torch2.13 environment.

The runtime-stage ceiling remains1 H100-hour. The measured four v1 containers and two later SDPA diagnostics used about0.50273 hours. Reserve a further0.10 hours for earlier engineering and unmeasured historical overhead; this is conservative planning allowance, not a reconstructed exact lifetime bill. New deadlines are operator180s, profile8 180s and full8 900s, plus20s kill grace each: maximum1320s=0.36667 hours, leaving about0.0306 hours before the planning ceiling. Before retries, reconcile actual elapsed intervals; never reset the allowance by renaming a job. Changing the backend creates a new engineering version, not a scientific metric/cap amendment.

## Evidence and source boundary

The same data/lexical-allocation contract and unchanged branch code can reuse matched prior gates. Backend-dependent numerical, replay, masking and throughput claims require new evidence. C71 remains SDPA-short-only; C72 remains a measured SDPA operator cost; no new useful-controller claim or SOTA statement follows.

[PyTorch's inference article](https://pytorch.org/blog/flexattention-for-inference/) describes an existing short-query decoding implementation with GQA. [Transformers' attention interface](https://huggingface.co/docs/transformers/en/attention_interface) documents backend-specific mask conversion. Installed source and measured outputs, not a web description alone, decide admission.

The current primary-source novelty/benchmark contracts are unchanged. For the empirical strength rubric, S5 effect/uncertainty and S7 strongest-comparator execution remain0; the engineering increment can improve practical measurement and S9 provenance but cannot pass the central fair-strong gate. An independent empirical critic must audit resulting masks, output completeness, stopping and cost before opening baseline fitting.
