# IRC data admission and CUDA audit

2026-09-07. Mode: `DRAFT_FROM_EVIDENCE`, audit notes only. Data admission is complete within the declared historical-coverage boundary. The scientific engine, fitted baseline, response policy and behavioral gates are not complete. C67 remains unverified; no SOTA or submission-readiness claim follows.

## Measurement admission

The prospective `irc-measurement-and-allocation-v1.md` amendment explicitly changes the new experiment's primary endpoint from Math-Verify to the unchanged MATH authors' normalized answer score, with nonempty-reference/prediction guards and required termination. This change preceded allocation and new model outcomes. It does not relabel an old experiment or change its cap or success threshold. The existing general-MATH Math-Verify function is unchanged and supplies a fixed sensitivity endpoint on its pre-audited supported-reference subset.

All 12,500 references were mechanically audited, without generating model answers or printing test solutions. The author scorer admits 7,496/7,500 training references and all 5,000 test references. The four exclusions are `train:algebra:888`, `train:algebra:1011`, `train:number_theory:661` and `train:number_theory:663`: two braceless and two empty boxed references. Training exclusion is 0.0533%; test exclusion is zero, both below the prospectively specified 1% ceiling. Reference self-agreement tests coverage, not universal semantic validity.

The author's grader can reject algebraic equivalents and ignore distinctions such as units or earlier multipart boxes. Math-Verify has different extraction and symbolic-comparison failures. These are retained measurement limits. Report both graders' effects on the same supported subset and the primary score on the full admitted panel. Prediction exceptions are technical failures, not permission to regenerate or omit a row; Math-Verify's `TimeoutException` inherits `BaseException` and needs its explicit catch in the future runtime scorer. The materializer handles it, but an actual prediction-side timeout has not yet been exercised end to end.

## Allocation and exposure

`artifacts/data/irc_v1_allocations/manifest.json`, SHA-256 `e489f1273b1c667ac5f5bbc35963a5ed480a01224f155cf52ee56bbd79083156`, binds all 14 source parquets, 144 historical input sources, 189,095 retained/conservatively included historical texts, source code, evaluator versions, component membership and packet hashes. The complete materialization took 42.469 seconds on the local CPU and inspected zero new model outcomes.

| Original split | Fit | Response | Selection | Development | Reserve | Sealed test | History exclusion | Train-overlap exclusion | Duplicate | Reference exclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train, 7,500 | 1,500 | 512 | 256 | 256 | 4,532 | 0 | 387 | 0 | 53 | 4 |
| Test, 5,000 | 0 | 0 | 0 | 0 | 0 | 4,359 | 593 | 35 | 13 | 0 |

All 11,415 active representatives have distinct component IDs. The 128-question response pilot is exactly `manifest.pilot_cluster_ids`, selected by the frozen salted group order; it is not the first 128 rows of the component-sorted response JSON. Test components intersecting any official training question, including reserve and reference-ineligible rows, are excluded. The sealed packet and manifest contain source locators, hashes and metadata, not test problems or solutions. Training workers must receive only their designated training payloads.

The independent measurement critic reconstructed every MATH row identity, checked allocation quotas and representatives, all packet/source hashes, history exclusion and test-versus-all-training separation, and confirmed the pilot order. No readiness-blocking allocation defect was found. Tests also verify that restricting neighbor queries to MATH vertices preserves eligible MATH component identity and history exclusion; historical-only connectivity is deliberately not reconstructed.

One wording correction is necessary and is recorded here without rewriting the frozen amendment: its representative rule says "independently of answer content," but the existing stable source-row ID hashes the complete raw record, including solution bytes. The executed rule is the predeclared minimum stable row ID, with no model-outcome-dependent choice. It is deterministic, not literally independent of answer bytes. Component construction and allocation order use problem text only. No allocation or hash is regenerated to improve the wording.

The history inventory includes full source frames for MMLU, MMLU-Pro, GSM8K, ARC-Challenge, GPQA, TruthfulQA, BBQ, ETHICS and FaithEval, all MATH-500/AIME source questions, available old input packets and decoded retained GPT-2/C4 contexts. Full-source inclusion does not mean every source row was actually evaluated. Discarded unsaved C4 screening inputs and unavailable unrecorded prompts remain unreconstructed. The cosine-0.95 character-5-gram screen is lexical: it does not exclude all semantic paraphrases or establish absence from model pretraining. GSM8K remains previously exposed generalization, not a fresh confirmatory endpoint.

| Allocation | Admitted primary references | Math-Verify supported references |
|---|---:|---:|
| Fit | 1,500 | 1,497 |
| Response | 512 | 509 |
| Selection | 256 | 255 |
| Development | 256 | 256 |
| Reserve | 4,532 | 4,527 |
| Sealed test | 4,359 | 4,356 |

All 15 unsupported but primary-admitted references retain `ValueError` diagnostics; they are not excluded from the primary endpoint. Subject and difficulty counts are available for every allocation in the manifest. The response set contains one source `Level ?` record: preserve an unknown-difficulty stratum rather than guessing a label. The sealed test has 386/787/998/1,053/1,135 questions at levels 1–5. These are population descriptions, not measured per-stratum model performance.

The 4,359-question target is the unexposed, deduplicated MATH population defined by this manifest, not all 5,000 official test records. Reference-exclusion bounds do not undo this deliberate population change. Nor does its size alone guarantee power: at paired discordance 0.25, an 80%-power normal approximation gives minimum detectable effects of 2.12 points for one two-sided 5% comparison and 2.71 points for eight Bonferroni-adjusted comparisons; at discordance 0.5 the eight-comparison value is 3.83 points. These are planning assumptions, not empirical estimates. Freeze the actual contrast family and development-derived variance before opening test outcomes, as already required by the benchmark.

## Runtime failure and recovery

`irc-cuda-runtime-check-v1.json` records exact commands, immutable image/source identities, package-tree hashes and cost exclusions. The seven unchanged upstream ReFT tests pass on actual CUDA tensors in both tested environments. They exercise real tiny Qwen3/Llama classes, finite adapter gradients, frozen base weights, saved/reloaded logits and bfloat16 output. They do not measure a useful trained 8B adapter.

The native image's Torch 2.13.0+cu129 / Transformers 5.12.1 passes those tiny tests but fails the actual Qwen3-8B synthetic branch probe: identity replay is false and split/resume replay is false, while the parent cache remains unchanged. The failed receipt is preserved at `artifacts/engineering/irc_native_branch_probe_v1/batch8.json`, SHA-256 `f36d824a9162a77313f55085aadb25ab8ca354b19f9202dd77c23936c08bb9d3`. The cause is not localized within the changed software stack. Tiny-test success does not admit this engine. An earlier native tiny-forward cache-directory permission error is separately retained in the runtime record.

The previously validated Torch 2.9.1+cu128 / Transformers 5.16.1 environment, isolated with Python `-S`, now imports Pyvene 0.1.8 through a narrow dependency overlay. It passes all seven CUDA ReFT tests in 9.51 seconds and the unchanged actual-weight batch-8 branch probe. Identity, split/resume replay and parent-cache preservation all pass. The transferred receipt hash is `4427203a4a213c8e285a4b6d9124d07679359e40d3ba588bd84186c4ab4f8095`, matching the remote file. No native Torch/torchvision is placed on that isolated import path.

The successful branch check generated 512 synthetic tokens in a single timed reference segment of 1.6913 seconds, or 302.73 tokens/second, with 16.76 GiB peak allocation. It is not a real-question throughput estimate or a comparison with the failed native run's cold timing. The two branch-process timers sum to 0.01513 H100-hours; this excludes container startup, imports before the timers, tiny-test setup, deployment and hashing. The full experiment cost is not estimated from that partial denominator. All owned workers exited, their containers were removed, and the final fsk42 compute-process list was empty.

## Minimal implementation and verification

The guarded author-score wrapper reuses pinned upstream functions; the narrow loader checks source hashes without executing historical top-level API configuration. The materializer reuses stable identity/allocation and SHA helpers, extending `splits.py` with SciPy/scikit-learn clustering instead of another nearest-neighbor implementation. The materializer is 293 lines and the score module 267. No new comments/docstrings or historical scorer changes were introduced. Failing tests were observed before each new callable was implemented.

The final local suite passes 341 tests in 29.87 seconds; critical Ruff checks pass. Four warnings are retained: one existing Torch JIT deprecation and three unchanged upstream invalid-escape warnings. The independent critic's reconstruction adds an audit beyond the unit tests but is not an independent behavioral replication. The data manifest's `ready_for_fitting` refers only to data admission; its scientific-engine/behavioral flag remains false.

## Updated execution order

1. Reuse this allocation and the passing upstream ReFT/branch contracts. Do not rerun source admission or the same synthetic probes without a changed input.
2. Freeze a complete isolated scientific-runtime receipt and the 16-real-training-question runtime workload; use the same fixed workload at batches 1/4/8, record all outputs, scoring failures, deterministic identity, row joins, memory and projected full-budget cost. The remote runtime is not an installation of the current local `uv.lock`.
3. Before training, freeze the native chat/solution-target format, adapter application positions, optimizer and finite fit/selection schedule. Use only fit/selection allocations for those roles and honor the eight-H100-hour baseline ceiling. Reuse the exact upstream adapter through the existing hook; do not add another trainer abstraction without a demonstrated need.
4. Select a genuinely useful fixed action before collecting intervention responses. Then use eight resident workers with shared prefills, per-row RNG and immutable 16-question chunks, while CPU verification consumes completed chunks. Additional GPUs serve independent question-action-seed work; they are not occupied merely to inflate utilization.
5. Run the frozen 128-question response pilot and its held-out-seed opportunity test, followed only on admission by the remaining response packet and full-rollout development. Capability, matched mechanisms, simultaneous strong-comparator tests, independent models and the three-family/three-construct paper gate remain mandatory.

The metric/exposure increment is resolved within its declared scope. The useful fixed baseline, response utility and closed-loop benefit remain central unanswered scientific questions. The next gain in iteration speed comes from resident batched workers and reused passed gates, not another protocol reset. Version 20 and observer-v1 remain closed.

## Prose and gate receipt boundary

Paragraphs above map to the immutable amendment, allocation manifest, unchanged upstream source/tests and transferred runtime receipts. Counts were recomputed from manifest membership, timing from receipt denominators, and power examples from explicitly assumed discordance and multiplicity. No plot is warranted for exact allocation counts and pass/fail conformance. The claim, numerical-result, audit-prose and local self-check gates pass for these narrow audit statements; the experimental-setup and fair-strong research gates remain blocked for scientific execution/paper promotion. No submission prose or new venue-compliance claim is produced.
