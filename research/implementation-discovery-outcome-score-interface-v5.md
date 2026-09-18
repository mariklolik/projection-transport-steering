# Outcome-score interface implementation discovery, version 5

Discovery date: 2026-09-07. Status: completed before dependency or code changes.

## Existing reusable path

- `outcome_score_runtime.generate_rollout` already owns prompt formatting, generation, and scorer dispatch.
- `outcome_score_sentinel.sentinel_protocol` already owns model, layer, cap, mode, and answer-format identities.
- The existing runner already validates model config before CUDA and records a hash-bound receipt.
- The existing analyzer already validates packet hashes and tensors before stage decisions.
- Version-4 outputs provide immutable conformance fixtures without becoming version-5 results.

## Minimal addition

Add `math-verify==0.9.0` to the existing dependency set and lockfile. Add one scorer wrapper with the audited extraction configuration and fail-closed exceptions. Add `answer_format=math_verify_strict` and evaluator identity to version 5. Pass the evaluator identity through the existing receipt and validate it in the existing summaries.

Do not add another regex, runner, analyzer, allocation, or output schema. Rows retain parse status, extracted answer, exact correctness, and generation metadata. If Math-Verify returns a noninteger mathematical candidate, serialize its canonical string and score it false against the integer gold.

Versions 2 through 4 must remain byte-reproducible. Observer, transport, steering, baselines, and downstream data remain outside this diff.
