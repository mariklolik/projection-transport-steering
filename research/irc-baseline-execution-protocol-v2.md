# Complete fixed-adapter baseline execution v2

Prospective implementation contract, before any eligible baseline fit or selection outcome. The baseline-only 48 H100-hour resource amendment governs this packet; it does not reopen the one-hour engineering budget or change inference cap, allocations, candidates or scientific claims.

## Discovery and integration

`run_irc_fit_probe.load_examples` already verifies the immutable fit packet, author extraction and exact native supervised targets. `reft_training.fit_step` already implements adapter-only, globally target-token-normalized accumulation, clipping and one AdamW step. `DataCollatorForSeq2Seq` supplies padded targets. `run_math_batch` already owns the native prompt, independent per-question RNG, cached generator, raw-output persistence, EOS/cap policy and both graders. `TorchLayerAction` owns the removable decoder hook. Reuse these functions and the pinned upstream `LoreftIntervention`.

Missing wiring is a complete epoch loop, a cached-generation position-mask context, a final raw checkpoint, and fit/selection accounting. Add these contracts without replacing attention, sampling, scoring, source labels or the earlier diagnostic entry point. Callers must retain all questions, reject source/hash mismatches, persist failed and completed chunks, never overwrite an output directory, and distinguish selection completion from positive action admission. No training resume is offered.

## Fit schedule

Eight independent candidates are ranks 4/8 × sites 17/23 × prompt/full. Each uses all 1,500 fit questions for 12 epochs, native unit strength, seed 20260907, effective batch 32, microbatch 2, AdamW learning rate 0.0009, betas (0.9,0.999), epsilon 1e-8, weight decay 0, adapter dropout 0 and gradient clipping at norm 1. The base remains frozen and in evaluation mode; this does not disable adapter autograd.

For epoch e, use a separate CPU Torch generator seeded with 20260907+e to permute original fit-row indices. Preserve each consecutive logical batch's membership; sort its indices by decreasing exact target length, then index, only for adjacent microbatch pairing. There are 47 updates per epoch, with 28 questions in the final update, and 564 updates overall. Every update uses the total unmasked shifted target count across its microbatches, one zero-grad, one clipping operation and one optimizer step. Use native Transformers linear scheduling with ceil(0.1×564)=57 warmup updates; record the learning rate used before each update and step the scheduler once afterward. The valid initial zero-learning-rate update still advances optimizer state.

Right-pad each pair to the smallest of 384,896,2816 that contains both complete examples. These are finite compiled shape buckets, not truncation; every admitted target is at most2689 tokens. This differs from padding to a multiple of128 and must have a source-only cost audit before dispatch. Use the qualified static native Flex callable for fit, with unchanged IEEE FP32 prefill arithmetic. Restore the original dynamic native callable for generation, whose long-budget runtime was separately qualified. A failed static/dynamic training-trajectory comparison stays failed; this is an explicit engine contract, not equivalence or a speedup claim.

Save only the final adapter using unbound `torch.nn.Module.state_dict`, including the orthogonal chart, and reload the saved raw state before selection. Do not claim complete optimizer/scheduler/RNG resume. No intermediate-checkpoint, strength, learning-rate, epoch or position selection is permitted.

At the zero-to-fit and fit-to-selection engine boundaries, reset the process-local Dynamo graph cache with `torch.compiler.reset()`; retain the immutable disk kernel cache. This prevents unrelated dynamic inference guards and three static fit shapes from sharing an accumulated guard-limit history. Generation uses the original Transformers default auto-dynamic callable, not an explicitly forced `dynamic=True` callable. The reset has no precision, weight, mask or RNG change; its compile/reload cost stays in the worker budget.

## Generation and selection

Each worker first generates one disjoint contiguous32-question share of the common zero reference, then fits its candidate, then evaluates that final candidate on all256 selection questions, in original selection order, batch8. Zero is unmodified; both adapters affect only first/last seven unpadded prompt positions at prefill. On subsequent cached one-token forwards, full additionally intervenes on generation positions; prompt-only does not. Left padding is never selected. Absorbed rows may incur padded forward work, which remains in the cost.

Reuse unchanged temperature0.6, top-p0.95, top-k20, per-question rollout seed0, EOS/pad configuration and8192-token cap. Persist raw outputs before grading, retain cap failures and disagreements, and stop on evaluator exceptions without dropping rows. Selection maximizes exact correctness separately over four full and four prompt candidates; ties prefer lower rank, then lower site. Full must strictly exceed the shared zero mean before response data collection can open. The winner is exploratory and is not a SOTA result.

## Admission and stopping

Before launch bind exact source/config/data/model/runtime hashes, verify all native targets and full epoch/bucket schedule on CPU, exercise saved-state generation masks and the actual full runner on tiny CPU models, and obtain an independent implementation review. Do not repeat the completed native accumulation diagnostic. Fresh GPU occupancy must permit each exact worker. Each worker's entire container lifetime, including termination grace, must stay below six hours; all eight and any explicitly admitted technical recovery stay below48 H100-hours. No automatic failure or unfavorable-result retry. Incomplete coverage is budget-incomplete, never a completed comparison.

The engineering pass is limited to two accumulated updates on four unique fit questions plus raw-state/logit reload. General rank/site/length training and steered long generation are risks to be observed in this scientific stage, not already-passed claims. No response, development, final-test, stronger-comparator or three-family/three-construct gate is opened merely by launch.
