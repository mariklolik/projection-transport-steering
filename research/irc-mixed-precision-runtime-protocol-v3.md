# IRC FP32-prefill Flex runtime qualification v3

2026-09-07. Prospective actual-weight engineering qualification, after the v2 synthetic outcomes and before any v3 model output. DRAFT_FROM_EVIDENCE, not submission prose. Native BF16 Flex v2 remains failed; this named backend is a distinct candidate, not a retroactive pass.

## Evidence and discovery

The v2 decode operator passed nine fixed cells. Its supplemental causal prefill failed the unchanged FP64 pointwise tolerance in 30230 of 4227072 output elements. Identical BF16 inputs also failed with default SDPA (30909 elements); SDPA MATH and losslessly upcast FP32 Flex, each cast back to BF16, had zero violations. All these prefill implementations returned finite values and exactly zero for the 28 fully masked queries. These are single synthetic input ensembles, not independent trials or model-level evidence. A probability-rounding mechanism is source-consistent but not isolated experimentally.

The first FP32 control failed compilation because a raw kernel option emitted an unquoted Triton constant. The retained retry uses public `torch.set_float32_matmul_precision("highest")` and the installed default that emits the quoted IEEE constant. It verifies all four original tensor hashes. No tolerance, tensor draw, mask or oracle changed. Evidence is in `irc-flex-prefill-followup-v2.json`, both precision-control specs, and their retained logs under `artifacts/engineering/irc_flex_runtime_v2`.

The codebase has no attention-backend registration helper. The existing runtime CLI now consumes its pre-existing attention configuration field; existing branch/cache, sampler, outcome scorer and training-allocation checks own the rest of the workload. A 25-line `attention_backend.py` reuses the installed attention callable and mask formatter. Public AttentionInterface and AttentionMaskInterface both register the explicit name `fp32_prefill_flex`. Query length greater than one casts Q/K/V to FP32; query length one retains BF16. Returned output and optional log-sum-exp retain the original query dtype. No custom kernel, mask, KV repetition, cache conversion or installed-package patch is introduced. The separate helper is shared-capable for the later fit runner; no training admission is implied.

## Frozen runtime decision

Use the same Qwen3-8B revision, isolated Torch2.9.1/Transformers5.16.1 runtime, image, GPU class and original 16 fit-only runtime questions as v1/v2. Preserve packet order, native prompts, tokenizer, sampling, per-row seed, EOS/pad, BF16 weights and cache, and raw reference persistence. The registration sets IEEE float32 matmul precision; configuration, source hashes, actual loaded backend and observed precision enter the receipt. V1 raw results and executable snapshot remain immutable.

First run batch8/512 with every original identity, split replay, parent-cache and action-path check. Only if this condition has 16/16 unique expected IDs, all checks passing, no evaluator errors and a final pass receipt, run batch8/8192 with complete scoring. The full condition must finish all 16 original questions with a final pass receipt. Wrong, capped or partial outputs remain failures or observations under their original definitions. No cap reduction, replacement question, scoring retry or threshold change is permitted. This version does not qualify batch1/4, cross-backend exact tokens, real-weight backward or fit throughput.

Use one isolated container per condition, network disabled, a hash-bound read-only source snapshot, read-only weights/environment, unique output directories and two compilation workers. Cold compilation, model loading, checks and scoring belong to elapsed process cost. Deadline is 180 seconds plus 20 seconds kill grace for profile8, then 900 plus 20 for full8. A profile failure stops this sequence; a full failure prevents runtime admission. Do not spend the remaining allowance on automatic undocumented retries.

## Accounting and inference

Six operator attempts consumed 137.356475079 seconds of measured single-GPU container elapsed time, including the setup and compiler failures. Together with about 0.50273 H100-hours of previously measured v1/SDPA jobs, a separate conservative 0.10-hour historical-overhead planning allowance, and the maximum 1120 seconds of these two actual-weight conditions, planned cumulative use is about 0.9520 H100-hours. The unchanged runtime ceiling is one H100-hour. This is allocation time, not integrated GPU activity; the allowance is not an exact lifetime bill.

CPU tests check the small precision adapter's argument/dtype/gradient plumbing through an intercepted native call and check exact registry identity. They do not test a GPU Flex kernel. The v2 native operator evidence and the paired FP32 control establish only the listed inputs; actual-weight runtime checks remain mandatory. Native GPU backward and fit-step cost require a separate prospective qualification before fitting.

S5 effect/uncertainty and S7 strongest-comparator execution remain zero. No useful IRC controller, benchmark SOTA, quality preservation, whole-model speedup or paper readiness follows from this engineering candidate. An independent critic audits output completeness, numeric scope, mask registration and cumulative cost before the next scientific stage.
