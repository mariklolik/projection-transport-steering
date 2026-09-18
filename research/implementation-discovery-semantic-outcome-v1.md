# Semantic-outcome implementation discovery, version 1

Discovery date: 2026-09-07. Status: completed before code changes.

## What already exists

- `claim_relative_geometry.robust_metric_action` implements the exact SPD hard or L1-slack robust QP and returns action, dual variables, margins, slack, objective, feasibility, and KKT residuals.
- `claim_relative_geometry.reduced_identity_robust_action` solves the same problem in the covector span and avoids an ambient dense identity solve.
- `invariantback_runner.candidate_mean_log_likelihood` and `encode_candidate_batch` implement differentiable, length-normalized multi-token continuation scoring with prefix-preserving tokenization checks.
- `torch_runtime.AdditiveAction` and `TorchLayerAction` apply one vector to every causal token position at a supported decoder layer across Gemma/Llama/Qwen-style, GPT-2, and GPT-NeoX layer collections.
- `statistics.py` contains deterministic BCa primitives; prior runners contain source hashing, atomic shard receipts, replay checks, resource measurement, and failure serialization.
- AxBench already implements the prompt sampling, five/five tuning-test split, generation schema, and judge rubric. FLAS and Steer Like the LLM provide exact comparator generation paths.

## Smallest missing part

The only new algebra needed before a model runner is a diagonal-SPD robust-action adapter. It whitens covectors by the positive square root of the metric diagonal, calls the existing reduced identity solver, and unwhitens the action. It must preserve the existing return contract and add the metric cost and reduced rank. A dense diagonal matrix and ambient `O(d^3)` solve are unnecessary.

The model runner then wires existing batch encoding, continuation likelihood, layer hook, robust solver, statistics, and receipt conventions. It does not introduce a new hook framework, tokenizer path, optimizer abstraction, judge wrapper, bootstrap implementation, or manifest format.

## Caller contracts

- Metric diagonals are finite, one-dimensional, strictly positive, and match the covector dimension.
- Covectors are finite nonzero columns; every view is retained.
- Hard infeasibility and KKT failure are explicit statuses.
- Zero action reproduces unhooked logits before gradients or generations are admitted.
- A concept is the atomic failure and inference unit; partial methods never silently reduce its denominator.
- Construction, tuning, test, and neutral rows carry stable hashes and cannot cross allocations.
- Every output records model snapshot, tokenizer hashes, source hashes, concept and prompt IDs, method, factor, layer, seeds, derivative and pass counts, wall time, peak memory, status, and raw failure.

## Minimal edit decision

First add and test the diagonal adapter only. Next add a one-concept source-and-gradient smoke that reuses existing modules. A multi-concept runner is justified only after the smoke receipt passes. Analysis and open-generation evaluation remain separate scripts so failed generation or judge calls cannot corrupt construction artifacts.
