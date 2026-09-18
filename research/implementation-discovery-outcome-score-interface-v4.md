# Outcome-score interface implementation discovery, version 4

Discovery date: 2026-09-07. Status: completed before code changes.

## Existing implementation

- `outcome_score.py` already separates immutable line-only and terminal `[0,999]` scorers.
- `outcome_score_runtime.py` already routes prompt/scorer behavior by `answer_format` and routes `enable_thinking` into the tokenizer.
- `outcome_score_sentinel.py` already owns versioned interface fields and version-aware termination and class-mix gates.
- The existing shard runner already loads one causal LM, records model files and config hash, captures layer states, replays a zero action, and writes hash-bound artifacts.
- The existing analyzer already validates every row, trace, receipt, and packet hash.

## Reuse and wiring

Add one terminal bounded-integer scorer beside the old scorers. Add a version-4 protocol entry with `answer_format=reward_integer`, layer 14, model config hash, non-thinking mode, and the unchanged 2,048 cap. The runner reads layer and model identity from the protocol, passes the answer format through the existing runtime, and records the same fields in its receipt.

The analyzer and allocation do not fork. Version-2 and version-3 artifacts must continue to reproduce byte-for-byte under their frozen analyzers. No observer, transport, action, baseline, or downstream evaluation code is justified before Stage C passes.

## Caller contract

Rows retain their exact existing schema. A version-4 parsed out-of-range or negative integer has `parse_status=pass`, `correct=false`, and its integer in `extracted_answer`. Receipts retain all existing provenance and add no alternate result channel.

The runner must fail before model load when the layer or config hash differs from the frozen protocol. The smallest correct diff touches only the scorer, runtime dispatch, protocol registry, runner validation, and tests.
