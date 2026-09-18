# Separate short-prefill engine qualification v4

## Decision, authority and preserved history

The previous goal turn made progress: it closed baseline-v2 technically, preserved all48 native output files, identified the short-prefill compiler route and verified the archive independently. It did not establish a useful action or SOTA. The current evidence class remains EVIDENCE_THIN; C67/S5/S7 and the three-architecture/three-construct goal remain unchanged.

The user goal authorizes autonomous experiments on existing FSK machines and supplies no numerical compute ceiling. Under that authority, prospectively allocate **one additional H100-hour (3,600 container-seconds)** to this separately named engineering increment. This is an assistant planning decision, not a claim that the user approved a numerical budget. The original one-hour engineering ledger and its9.198053592-second remainder remain unchanged. Baseline-v2's7.605374 H100-hours and its unused allowance are not transferred or erased. No purchase, new infrastructure or interference with unrelated jobs is authorized.

Use at most one verified-idle FSK42 GPU at a time. Count every new container lifetime, including compilation, loading, failed work and termination. No automatic retry after numerical failure, no tolerance or precision relaxation, and no baseline rerun is authorized by this engineering protocol. A packaging-only failure may receive a separately recorded correction only before numerical outputs, within the same3,600-second aggregate ceiling. Compiler, resource, numerical and timeout failures are never packaging-only failures, even before the first numerical output.

## Discovery before code

`attention_backend.fp32_prefill_flex` already casts multi-token Q/K/V to FP32, delegates to native HF FlexAttention and restores the original output dtype. Its registration already reuses the native mask builder. HF forwards `kernel_options`; pinned Torch exposes `FORCE_USE_FLEX_ATTENTION`. The archived failure and source-only32-batch inventory locate the bad path at1<Q<128, with Q=113/GQA4/BLOCK_M512 directly source-bound for c01.

Reuse that wrapper and native option in a separately registered `fp32_prefill_flex_v4` callable. Override the option only for1<Q<128, copy rather than mutate caller options, and preserve mask identity, other kwargs, dtype and gradient flow. The original callable's body remains unchanged. Register the new alias through the existing registration entry point. Existing scientific producers and frozen v2 snapshots are not changed or relaunched in this increment.

`tests/test_attention_backend.py` already contains the grouped FP64 output/QKV-gradient oracle with fixed rtol0.005/atol0.0005 and masked-query zeros. Generalize only its existing geometry/callable arguments; do not write another attention oracle. `branch_runtime.start_branches/advance_branches`, `run_math_batch`, `load_packet`, source validators and current runtime launch receipts already provide the model, generation, mask, EOS, replay and artifact contracts. Extend only the missing wiring, with failing tests first. No new trainer, sampler, mask implementation, grader or candidate family is needed.

## Ordered qualification

| Gate | Workload | Maximum container lifetime | Required result |
|---|---|---:|---|
| A | Synthetic native FP64 output/VJP and mask checks, B8/H32/KV8/D128, module training=False, grad-enabled and detached-input no_grad forwards, all distinct actual selection batch widths plus2/127/128/129 | 600s | Every declared comparison passes unchanged tolerances, finite gradients and exact masked-query zeros; no kernel failure |
| B | Actual Qwen3-8B prefill on all32 original selection batches with native masks and tokenizer; no sampling, grading or fitted adapter | 600s | All exact prompt/input hashes and widths, finite last-valid logits, unchanged base versions; no missing batch |
| C | Native actual-weight short-prompt runtime on16 fit-only questions, source-order selected by prompt length before outcomes; paired branch checks and complete8192-token condition | 2,400s | Original source/seed/token/EOS/replay/action checks pass; all16 full-condition raw/scored rows and terminal receipt retained |

These are total envelopes, not runtime estimates. Exact child process timeouts must leave termination/startup margin inside them. The sum of actual lifetimes may never exceed3,600 seconds. Gate B opens only after A; C opens only after B. A/B are new shape qualification, not repetition of unchanged passed workloads. C's short profile is an engineering check, not a reduced scientific generation cap; its complete condition retains8,192 tokens. Fit-only source selection and exact IDs/config/code hashes are frozen before C dispatch. No sealed-test access, new labels or candidate correctness ranking occurs.

Gate A explicitly binds the v4 callable, native mask registration and highest FP32 matmul precision. Its ordered35 geometries require175 successful numerical comparisons: grad-enabled output, detached no-grad output, and Q/K/V gradients for each geometry. Exact mask predicates and masked-query zeros are additional checks. Gate B uses the exact production first-forward kwargs: left-padded input IDs/attention mask, cumsum position IDs with padded positions zeroed, `past_key_values=None`, `use_cache=True`, `logits_to_keep=1`, model.eval and torch.no_grad. A generic uncached forward is not an adequate substitute for the production graph.

The actual selection prompt shapes are already exposed source metadata, not a new outcome-selected subset. Gate B returns finite-logit checks and hashes only, not model answers, likelihood-based selection scores or labels. Gate A uses one fixed synthetic seed (20260908), not independent statistical replicates. The comparison is numerical/operator qualification against the existing FP64 reference, not a superiority experiment or a strongest-baseline comparison. Cold timings are charged; no serving-speedup claim follows.

## Pass, failure and next action

Freeze exact source/config/command hashes and the hardware snapshot before dispatch. Keep partial output and every failure. A single numerical, source, mask, nonfinite, missingness or timeout failure blocks engine admission; do not change the failed contract. On complete pass, obtain independent integrity/claim criticism and decide whether a separately declared full scientific baseline can run within its own recorded remaining resource envelope. Do not evaluate only the three archived survivors or claim optimizer-resumable training state.

This increment resolves a prerequisite to measuring the scientific effect, not the effect itself. The closest-work/SOTA boundary in `literature-reset-v1.md` and `benchmark-reset-v1.md` is reused; no new novelty claim requires another survey. Responsible-author verification, disclosure classification and ICLR compliance remain unverified.
