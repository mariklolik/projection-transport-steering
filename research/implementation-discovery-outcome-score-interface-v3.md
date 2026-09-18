# Outcome-score interface implementation discovery, version 3

Discovery date: 2026-09-07. Status: completed before code changes.

## What already exists

- `outcome_score.extract_aime_answer` and `score_aime_completion` implement the immutable version-1 and version-2 line-only contract.
- `outcome_score_runtime.format_aime_prompt` and `generate_rollout` own prompt construction, chat-template invocation, seeded generation, objective scoring, and trace alignment.
- `outcome_score_sentinel.sentinel_protocol` owns versioned row counts, rollout indices, and token caps.
- `outcome_score_sentinel.summarize_termination` owns the eight-shard termination decision.
- `run_outcome_score_sentinel_shard.py` owns model loading, one-model-per-GPU execution, exact replay, serialization, and receipts.
- `analyze_outcome_score_sentinel.py` validates hashes and traces before calling the stage summary.

## What can be called instead of rewritten

- Preserve the current allocation, group selection, seeds, trace hook, generation loop, no-op action, model manifest, packet hashing, and analyzer merge.
- Preserve the line-only scorer for old versions and add one terminal boxed-integer extractor beside it.
- Extend the protocol registry with interface fields and pass those fields through the existing runtime.
- Make termination validation read its expected values from the existing protocol registry rather than creating a version-3 analyzer.

## Wiring point

`sentinel_protocol(3, stage)` supplies `max_new_tokens`, `rollout_indices`, `enable_thinking`, and `answer_format`. The shard runner passes those values into `generate_rollout`, which chooses the matching existing or benchmark-aligned scorer. The receipt records the interface values. The unchanged analyzer validates the packet and asks `summarize_termination` for a version-aware gate.

## Caller contract

Every generated row must retain `completion`, `generated_tokens`, `generation_id`, `group_id`, `rollout_index`, `seed`, `trace_length`, `parse_status`, `extracted_answer`, and `correct`. Every receipt must retain existing hashes, model identity, environment, timing, memory, replay, and stage fields while adding explicit inference and answer-format fields.

Legacy version-2 tests and analyses must remain valid. Version 3 may not mutate old raw artifacts, allocation files, hashes, or score their rows under a new primary contract.

## Minimal justified diff

No new production module or runner is needed. The only justified code is one narrow parser and scorer, two runtime parameters, version-3 protocol entries, receipt fields, version-aware termination validation, and their tests. All other method, observer, transport, and evaluation code remains unopened.
