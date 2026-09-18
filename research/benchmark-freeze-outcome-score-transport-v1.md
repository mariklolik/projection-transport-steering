# Outcome-score transport benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before result-generating implementation or GPU output.

## Sources and immutable allocation

Historical AIME comes from `Pandores/aime-1983-2025`, commit `292f1a1b4f041918afced4435127c7eda48a2597`, parquet SHA-256 `046308535b5fc06b501bd3311c7c5080eaaef00239e7069bb0181b73b88b2d63`. Its 1,035 rows contain 30 questions per year from 2020 through 2025 and 975 questions through 2023.

AIME 2026 comes from `math-ai/aime26`, commit `79037aebdb6580008fb960d17cb21fd3099083e3`, JSONL SHA-256 `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee`. Its 30 questions have no exact problem-string overlap with the historical source.

MATH-500 comes from `HuggingFaceH4/MATH-500`, commit `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, JSONL SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`. It is held closed until symbolic scoring is separately pinned.

For historical rows through 2023, derive the group ID with the existing `stable_group_id` contract from the source SHA, config `aime`, split `historical-through-2023`, and the exact `year`, `index`, `problem`, and `answer` fields. Apply the existing `allocate_groups` function with salt `ost-v1` and ordered quotas `basis=32`, `fit=256`, `calibration=128`, `validation=128`, and `reserve=431`. The 32 `basis` questions form the exposed engineering sentinel and remain eligible only for basis construction, not observer or metric fitting. Years 2024, 2025, and 2026 are never available to a fit worker.

- AIME 2024: one-shot development.
- AIME 2025: untouched temporal pilot.
- AIME 2026: untouched temporal confirmation.
- MATH-500: breadth confirmation only after a temporal pass and scorer audit.

No question moves between allocations. Question text, answer, source row, content hash, and split-manifest hash are recorded before rollout.

The materialized basis packet SHA-256 is `8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7`; its embedded content SHA-256 is `883723fdea812d0646685eeffed968c8ad571e6c8f18c20268ee9ccf31477fd3`. Fit, calibration, validation, reserve, development, and pilot packets remain distinct files and are never passed to a sentinel worker.

## Sentinel model and generation

Use `Qwen/Qwen3-8B`, snapshot `b968826d9c46dd6066d109eabc6255188de91218`, bfloat16, residual layer 18 of 36. The sentinel uses two stochastic rollouts per question, temperature `0.7`, top-p `0.95`, maximum 1,024 new tokens, and seeds derived from `SHA-256("ost-v1|question_id|rollout_index")`. The prompt requests worked reasoning followed by `Final Answer: <integer>`.

The exact scorer accepts only a final integer in `[0,999]` from the declared terminal-answer form. It records ambiguity rather than choosing among conflicting terminal answers. Correctness is exact integer equality. No LLM judge is used.

Sentinel promotion requires all 64 rows, exact key uniqueness, deterministic seed/config receipts, zero empty or ambiguous parses, both correctness classes with at least eight rows each, finite activation traces, exact no-op replay, peak memory below 70 GiB, and estimated eight-shard fit wall time below six hours. Failure closes or narrows the route before observer fitting.

## Fit and steering stages

If the sentinel passes, run four stochastic rollouts per `basis` and `fit` question and two per `calibration` question under the same decoder. Use one resident model on each of eight GPUs and deterministic disjoint question shards. The validation allocation is opened only after a frozen observer artifact exists.

The observer is selected only on fit data; calibration fits probability calibration, transport maps, metric ridge, score support, gate threshold, and one intervention-count budget. Validation estimates OST.1 once. No steering condition runs on validation.

After OST.1 passes, AIME 2024 permits one common-protocol method packet. One candidate configuration may be selected using the declared primary and safety ordering. That exact artifact is then sealed for AIME 2025. AIME 2026 opens only after the AIME 2025 decision and cannot alter the artifact. No evaluation result changes layer, rank, prompt, gate, dose, or baseline identity.

## Baselines, metrics, and analysis

The required methods are those listed in `method-hypotheses-outcome-score-transport-v1.md`. Published LRS values remain unmatched markers; the common-protocol LRS-like baseline is explicitly labeled clean-room.

Primary outcome is per-question exact correctness under paired method seeds. Report transition counts, paired difference, one-sided exact McNemar test, question bootstrap interval, and Holm adjustment across the three post-development datasets. Secondary analyses report baseline-correctness, year, difficulty proxy, progress quartile, response length, observer score, intervention count, output-KL mean and 95th percentile, recoveries, degradations, parse events, numerical failures, wall time, tokens, memory, and derivative counts.

The strongest feasible baseline is chosen by the frozen development rule, not by test outcome. SOTA wording remains governed by `sota-definition.md` and cannot follow from this one model-behavior route alone.
