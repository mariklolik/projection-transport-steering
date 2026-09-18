# IRC implementation discovery

2026-09-07. This is an execution note for the prospective reset, not submission prose or a behavioral result. Existing closed protocols are unchanged.

## Reuse and contracts

`outcome_score.py` already owns Math-Verify dispatch. The new general-MATH caller accepts a reference solution and the pinned author's last-box extractor; it does not modify the existing integer scorers. The author helper resolves multiple boxes. Symbol-preserving normalization is needed because treating variable names as units can change mathematical meaning. The complete training-reference audit remains a separate admission dependency.

`TorchLayerAction` already installs and reliably removes a shape-preserving post-block intervention. The pinned upstream `LoreftIntervention` is directly callable on `[batch, tokens, hidden]`; its own parameters and serialization suffice. The project therefore adds conformance tests and the pinned `pyvene` dependency, not another ReFT implementation or trainer. The zero action uses the existing identity module.

`materialize_outcome_score_data.py` already supplies file and canonical-payload hashing. The source audit calls those helpers, checks pinned download metadata and Git revisions, reconciles all four MATH record fields, and records training-reference failures without excluding rows. Model metadata is not a weight or CUDA receipt.

## Branch-continuation discovery before code

`outcome_score_runtime.generate_rollout` delegates to Hugging Face generation, seeds global Torch state, handles one row, and retains a dense per-token trace. `semantic_outcome_runtime.generate_many` supports left-padded greedy batches but no isolated stochastic branches. `cacheback_runner._sample_continuations` demonstrates explicit per-trajectory generators but recomputes the entire prefix without a KV-cache. None implements the required callable contract; changing these closed-path callers would add risk without reuse.

The new narrow `branch_runtime.py` will use Transformers' `DynamicCache` and public temperature/top-k/top-p processors, Torch's per-row generators, and the existing intervention hook. It will not implement attention, caching internals, a sampler distribution, tokenizer formatting, a transport method, or a training framework.

The state contains left-padded token IDs, attention mask, cache, per-row generated lengths, EOS termination flags and RNG states. After a generation step, the final sampled token is pending: its forward pass is not yet in the cache. Thus a branch applies its intervention while processing that token, before sampling the next one. Every continuation deep-copies the incoming state once. It may mutate its own copy, never another branch or the parent. EOS rows are absorbing and do not consume further random draws; truncation is distinct from EOS.

Callers advance the shared reference prefix to a declared cut, continue a cloned state for exactly 64 steps under an action hook, then continue under the fixed reference. A repeated controller can choose within the same post-block hook without an extra full-model pass. Correctness of that later policy, intervention duration, response joins and real-weight batching are separate tests; a standalone continuation function does not establish them.

Tests precede implementation: exact same-engine zero-action replay, per-row RNG isolation and row-order equivariance, parent/branch cache isolation, cached versus uncached conditional logits, left-padding, EOS absorption, invalid-input rejection, and unchanged global RNG. Tiny randomly initialized Qwen3/Llama models exercise the real architecture code on CPU. These checks cannot establish 8B/H100 throughput or behavioral improvement.

## Current boundaries

No source-reference failure is a failure of IRC efficacy. No passing source or runtime check is evidence for IRC efficacy. Dataset exposure/deduplication, the final heterogeneous-answer scoring contract, action fitting/selection, GPU batching and all behavioral comparisons remain required. The prospective success thresholds in `benchmark-reset-v1.md` are unchanged.
