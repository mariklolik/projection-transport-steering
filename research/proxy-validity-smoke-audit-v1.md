# Teacher-forced proxy validity smoke audit, version 1

Date: 2026-09-07. Decision: close before the full packet and retire the frozen proxy-first funnel.

## Scope and stopping rule

The hypotheses, 12 exposed concepts, 14 Flow-step conditions, eight prompts, generation settings, Qwen3-14B judge, exact upstream prompts and parser, completeness gate, correlations, bootstrap, and thresholds were frozen before generation. The plan authorized one engineering smoke containing concept 1, prompt 0, and all 14 conditions. Full generation was permitted only if that smoke was valid.

This study validates an instrument. It cannot select a Flow-step condition, reopen Flow-step support version 1, establish behavior improvement, or support a paper claim.

## Attempt lineage

The generation smoke passed. It contains 14 nonempty, uniquely keyed outputs, exact JSON round-trip serialization, 14 generation calls, 1,794 counted language-model forwards, 43.4795 seconds elapsed time, and 5,482,067,968 peak allocated bytes. The packet listing SHA-256 is `5e6e38f3b0aeb88a8cdf7246f4877b448c115adac6bf7283d91418f44bb5fc35`; its receipt SHA-256 is `99c3caf3ac48bbb1cbb81979813612645653e7cb9bfb135bced7f96c48f01b37`.

Judge smoke attempt 1 stopped before model load or any judgment because importing the hash-locked upstream judge module also imported the unavailable optional `openai` package. Its only artifact is `failure.json`, SHA-256 `506caf30ef45a0ad206a9735ad82f884a2163f98b84f58815775a0821f8cfb91`; the packet listing SHA-256 is `b418f0a69da0f9cb65329b072f8e1e1f58c4cd2e9b6b1031b1c3c3b029b3bda4`.

The infrastructure repair loads only the three prompt constants and two scoring functions from the same hash-verified upstream abstract syntax tree. It neither copies nor changes their source. Judge smoke attempt 2 used the frozen model, batch size 8, greedy decoding, and maximum 160 new tokens. It preserved 42 raw completions and exact serialization, used 916 counted judge-model forwards, took 58.1574 seconds including model load, and peaked at 30,996,022,784 allocated bytes. Its packet listing SHA-256 is `9bb493defdb19dc2b1516c8bcd9c386ff1048e8afdc6bb43efb78029c081152b`; its receipt SHA-256 is `9b60f57b4bc4df24bb68793ebfd3765f9a5a3871dab553670519909d77d22434`.

## Frozen-gate result

Only 13 of 14 condition rows and 40 of 42 individual judge completions parsed. The failed row is `1:equal_01:0`. Its Concept completion reaches exactly 160 tokens without emitting `Rating:`. Its Fluency completion also reaches exactly 160 tokens and ends at `Rating: [[`. The Instruction completion parses normally. Across all 42 completions, token length ranges from 66 to 160 with median 116; exactly the two unparsed completions reach the frozen limit.

PV.1 therefore fails through incomplete, condition-specific parse coverage. This is observed evaluator behavior under the frozen configuration, not the missing-package defect from attempt 1. Selective retry, a longer judge limit, moving the rating before the explanation, changing the parser, or mapping missing values to zero would alter the preregistered instrument after observing its failure and are disallowed.

The full 1,344-generation and 4,032-completion packet was not opened. PV.2--PV.5 were not estimated, no condition-level performance comparison was computed, and the fresh 24-concept, pilot, held-out, 9B, official-judge, model-scaling, and paper gates remain unopened.

## Interpretation

The failure exposes a mismatch between explanation-first judge prompts and the 160-token local-proxy budget: a fluent, concept-rich response can induce a long explanation that truncates the required terminal rating. Because truncation depends on generated content, its missingness can be condition-dependent. The teacher-forced screen is therefore not retained as the primary method-selection instrument under version 1. A future route must begin from a prospectively validated behavior-level evaluator rather than repair or reuse this opened smoke.
