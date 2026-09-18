# Outcome-score interface audit, version 5

Audit date: 2026-09-07. Decision: pass Stage T and Stage C; open observer fitting.

## Frozen questions

Stage T asked whether Qwen2.5-7B-Instruct with pinned Math-Verify 0.9.0 produces eight complete AIME outcome labels before 2,048 generated tokens. Stage C separately required 64 complete labels from two rollouts on each of 32 basis questions, with at least eight correct and eight incorrect outcomes. Both stages also required exact model, layer, evaluator, row, seed, replay, trace, memory, and runtime identities.

The evaluator source and strict configuration were audited in `math-verify-source-audit-v1.md`. The model, layer 14, prompt, rows, seeds, sampling, cap, sharding, gates, and stopping rules were frozen in `benchmark-freeze-outcome-score-interface-v5.md` before dependency integration or new output.

## Accepted packets

| Stage | Raw packet | Packet-listing SHA-256 | Analysis SHA-256 | Decision |
|---|---|---|---|---|
| T | `artifacts/development/outcome_score_v5_termination_attempt1` | `9429bc6b4c42fc55045d1efe8e1d22a48226f0fd952dc5252a8f8b36711cc5be` | `e2cd3d9a703c7452afd501ff950e5d1d6bf82d977bc05c91609f404dbb2c4155` | `pass_open_class_mix` |
| C | `artifacts/development/outcome_score_v5_class_mix_attempt1` | `17aa113d2930c6c6e625add7ce18eff5b72fb9c2ad04a43ac9ed34af39c68c5c` | `98b0312fa0127a141a81ec388d625344da5d072dac7fb2e0474c42050438e0f4` | `pass_open_fit` |

Both remote analyses reproduce byte-for-byte locally. Every receipt identifies `math-verify==0.9.0:latex-anchored-no-fallback-first-match`, the frozen Qwen2.5 configuration, and layer 14. The installed remote parser and grader hashes equal the audited source hashes. No retry or excluded scientific row exists.

## Stage T result

Stage T has 8/8 accepted parses, 8/8 pre-cap terminations, 8/8 finite aligned traces, 8/8 exact zero-action replays, three correct outcomes, and five incorrect outcomes. Generated lengths are `[521, 559, 574, 582, 625, 698, 956, 981]`, with median 603.5, mean 687, and 5,496 main-generation tokens.

All eight v5 completions, seeds, generated-token counts, and trace lengths are exactly equal to version 4. The intervention code therefore did not change generation while the prospectively frozen standard evaluator resolved the prior custom-parser attrition.

## Stage C result

Stage C has 64/64 accepted parses, 64/64 pre-cap terminations, 64/64 finite aligned traces, and 8/8 exact zero-action replays. All 32 frozen groups appear exactly twice with rollout indices zero and one. Sixteen rows are correct and 48 are incorrect, passing the frozen minimum of eight outcomes in each class.

The paired question patterns are five correct/correct, 21 incorrect/incorrect, four correct/incorrect, and two incorrect/correct. Thus 11 questions have at least one correct rollout and 27 have at least one incorrect rollout. Rollout zero has 9/32 correct and rollout one has 7/32 correct.

The row accuracy is 0.25. A deterministic 200,000-resample question-cluster bootstrap with seed 6505 gives a descriptive 95% interval of `[0.125, 0.390625]`. This is not a benchmark accuracy estimate or a method-effect interval; it quantifies uncertainty in the exposed basis packet used only to establish class support.

Decade strata are 3/10 correct in the 1980s, 4/14 in the 1990s, 0/16 in the 2000s, 8/16 in the 2010s, and 1/8 in the 2020s. These small, imbalanced strata are diagnostic only. Correct rows have median length 594.5 and mean 759.9 tokens; incorrect rows have median 607 and mean 656.4 tokens. No stratum contains a parse or cap failure.

The eight Stage-T rows recur as the fixed rollout-zero rows for their groups in Stage C. All eight completions, generation fields, and saved activation tensors match exactly across stages.

## Cost and utilization

Stage T uses 238.708 summed worker-seconds, 38.668 seconds parallel wall, 0.0663 H100-hours, 10,992 instrumented forward calls, 77.2% parallel efficiency, and a serial-to-parallel ratio of 6.17. Maximum allocated memory is 15,373,015,040 bytes.

Stage C uses 806.554 summed worker-seconds, 124.593 seconds parallel wall, 0.2240 H100-hours, and 49,164 forward calls. Main rollouts account for 43,668 calls and the eight required replays for 5,496. Parallel efficiency is 80.9%, the serial-to-parallel ratio is 6.47, and maximum allocated memory is 15,396,486,656 bytes. The frozen fit-wall estimate is 1,771.99 seconds.

## Interpretation and boundary

Version 5 supports only the interface claim C65: the pinned evaluator and Qwen2.5 inference contract provide complete outcome labels with both classes under the frozen basis packet. It does not show that an outcome observer is calibrated, that a steering direction is causal, that transport improves correctness, or that any method is state of the art.

The next authorized stage is observer fitting from the frozen basis traces. Calibration, validation, AIME 2024 development, AIME 2025 pilot, AIME 2026 confirmation, MATH-500, multi-model scaling, and manuscript efficacy claims remain closed until their own gates pass.

## Skill receipt

- Router state: `EVIDENCE_AVAILABLE + REVIEW_MODE + SUBMISSION_MODE`.
- `scientific-results-and-figures`: `PASS` for packet reconstruction, paired and temporal strata, cluster-aware uncertainty, resource accounting, and byte-exact reproduction; method effect remains unestimated.
- `scientific-claim-evidence`: `PASS` for C65 at interface scope and `BLOCKED` for outcome-steering efficacy or SOTA.
- `scientific-research-strengthening`: evidence advances from interface-thin to fit-eligible; fair-strong remains blocked by observer, causal, matched-comparator, protected-degradation, temporal, and multi-model gates.
- `experimental-setup-writer`: `PASS` for instrument and compute reporting; downstream experimental setup remains prospective.
