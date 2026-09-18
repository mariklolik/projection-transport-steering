# SemanticOutcomeBack benchmark freeze, version 1

Freeze date: 2026-09-07. Status: frozen before implementation, construction generation, judge calls, or model output.

## Research questions

1. Do simultaneous semantic-witness constraints improve held-out worst-view control beyond pooling the same sequence gradients?
2. Does the diagonal sequence-score metric improve specificity and distributional cost beyond an identity metric at matched action cost?
3. Do the two components jointly improve open-generation AxBench HMean beyond strong reproducible baselines?
4. Are any gains stable across concept difficulty, continuation length, baseline preference, witness disagreement, action norm, and prompt split?

## Sources and immutable split boundary

- AxBench code: commit `41c8332543e5a631f9a8c0a9df38799893ace758`.
- FLAS code: commit `720ef8a67697d9b94130b374b5b3a1522a782566`.
- Steer Like the LLM code: commit `3d916c618d146c5d657f055e432a432b0fa493c6`.
- Gemma-2-2B layer-20 Concept500 train data SHA-256: `6e4a00f39e9a11c1f6c74ed0b3241d1122621e6b55ce063eb3ad071b075b977b`.
- Gemma-2-2B layer-20 Concept500 latent data SHA-256: `382887cdd57b7824e327ea19c9a7abcc190ef75f74ab5fc7cb61f9c826985ae9`.
- Concept500 metadata SHA-256: `435aad14e202402b0989d62b279f63dfc63aa2aff0e488cf62652c0700a1efdc`.
- AlpacaEval prompt file SHA-256: `d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51`.
- 2B held-in concept list SHA-256: `f3564c7fbaf4119f248467f24b174a19636bbbeeb0a34f0ea07ac4e0a67861b3`.
- 2B held-out concept list SHA-256: `c7825ec825fb5d9e2b4b63f1b92d3937c00d1f700b99b1cc11d92c3966ee8139`.

The exposed sentinel source packet is `artifacts/source/semantic_outcome_v1_sentinel.json`, file SHA-256 `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`, content SHA-256 `e7f78438c7d1ee7dfb7f39d52722853e93f4614eb83960ae312d1efbfbc47464`. It contains 12 concepts with eight construction and eight latent-evaluation positives each, 16 disjoint neutral negatives, and 208 unique source IDs from the hash-pinned Concept500 release. The model snapshot revision, tokenizer files, and dependency lock must be hashed before the first model forward. Before development, the exact Concept16k packet and FLAS checkpoint must also be hashed, and the overlapping exposed rows must pass a source-equivalence audit. Missing or mismatched material blocks the corresponding stage.

The released 100-concept 2B held-in list is partitioned in its published order:

- sentinel, first 12 IDs: `1, 5, 16, 24, 29, 33, 34, 48, 58, 59, 60, 73`;
- development, next 24 IDs: `74, 76, 82, 84, 85, 87, 91, 114, 118, 123, 131, 132, 134, 140, 141, 142, 143, 144, 157, 169, 174, 182, 194, 204`;
- internal pilot, remaining 64 held-in IDs;
- confirmatory frontier, all 100 2B held-out IDs.

Only sentinel IDs may be read before the sentinel decision. Development opens only after SO.1-SO.4 pass. The 64-concept pilot and 100-concept held-out packet are not copied into an exposed run directory. The 9B lists remain unopened for execution until the 2B pilot passes.

## Construction and evaluation units

The independent inferential unit is a concept. Prompts and witnesses are nested within concept.

For each exposed concept, positive training rows are ordered by SHA-256 of `20260907`, concept ID, normalized input, and normalized output. The first eight rows form construction witnesses. For each input, the pinned unsteered model generates one matched negative continuation with greedy decoding, maximum 256 new tokens, and the exact chat template used at deployment. Both continuations are scored through at most their first 256 continuation tokens, with truncation and original token count recorded. Empty, failed, or tokenization-incompatible generations fail the concept; they are not replaced.

A disjoint 16-row global neutral pool is selected from released negative training rows by the same hash rule and is fixed across concepts. Latent-evaluation positive rows are independently hash-ordered; the first eight per concept form the held-out teacher-forced witness panel. Their matched negative continuations are generated before action construction for that concept and then frozen.

Open generation uses the exact AxBench AlpacaEval sampling rule: ten prompts per concept with pandas `random_state=int(concept_id)`. Input IDs `0-4` are tuning prompts and `5-9` are test prompts. Sentinel and development may access only IDs `0-4`; their test prompts are retained but not scored. Confirmatory evaluation selects a factor on IDs `0-4` and reports only IDs `5-9`.

## Models and sites

The first model is `google/gemma-2-2b-it` at decoder block 20, matching the inspected 2B frontier. Because the cached directory has no embedded Hub revision, its two weight shards, index, configuration, generation configuration, tokenizer, tokenizer model, tokenizer configuration, and special-token map are identified by the nine exact hashes in `semantic-outcome-source-receipt-v1.md`. Gemma-2-9B at block 20 is the second frontier model only after the 2B pilot gate. A later Qwen-family experiment tests architecture transfer and cannot rescue a failed Gemma route.

The shared vector is applied after the frozen decoder block at every causal token position during teacher forcing and at every generated token during decoding. Zero-vector outputs must reproduce the unhooked model exactly under the recorded tolerance.

The metric ridge is fixed as `max(1e-12, 0.01)` times the mean coordinate value of the neutral score-gradient second moment. View covectors are divided by their Euclidean norm before either robust objective is solved; the metric is not used to normalize constraints.

## Methods and factor matching

Required action methods are `pooled_euclidean`, `pooled_sequence_metric`, `robust_euclidean`, `semantic_outcome`, `last_token_semantic_outcome`, and `DiffMean`. `no_op` and AxBench `prompt` are required behavioral baselines. FLAS is required before any development performance claim. CES is required before any broad novelty or SOTA claim; inability to reproduce it is recorded rather than replaced by a weaker method.

The candidate factor grid is `{0.25, 0.5, 1.0, 1.5}` times its unit-margin action. Action baselines are rescaled to the same declared metric cost at each candidate factor. Each method's factor is selected only on the five tuning prompts. FLAS uses the released checkpoint and its paper-selected flow time `2` unless its exact released evaluation contract requires otherwise. All model passes, backward products, wall time, peak memory, stored parameters, and generated tokens are counted.

## Endpoints

The confirmatory primary endpoint is the exact AxBench normalized HMean over concept relevance, instruction relevance, and fluency on five test prompts per concept. The exact released `gpt-4o-mini-2024-07-18` judge is required for a frontier claim. Numerical zero on parse failure is retained for comparability, while raw responses, parse status, retry count, and method-wise failure rate are mandatory. Any missing row or judge failure rate above 1%, or a between-method difference above 0.5 percentage points, blocks promotion.

Mechanism and safety endpoints are:

- held-out positive-minus-negative length-normalized log-likelihood change, including the worst view per concept;
- neutral sequence log-likelihood and teacher-forced tokenwise KL against no-op;
- concept relevance, instruction relevance, and fluency separately;
- repetition, empty output, truncation, non-language output, and severe individual failures;
- action norm, active constraints, witness margins, gradient cosine and disagreement, metric condition ratio, KKT residuals, passes, wall time, peak memory, and artifact size.

No teacher-forced endpoint substitutes for open-generation behavior.

## Uncertainty and strata

All intervals resample concepts, not prompts. Development and later analyses use 10,000 concept-cluster bootstrap resamples with seed `20260907`. Primary candidate-versus-baseline comparisons use one-sided 95% intervals and Holm correction over the frozen required action comparisons. Two-sided 95% intervals accompany component metrics. Exact concept counts, missingness, and all individual failures are reported.

Frozen strata are concept-genre where present, baseline concept score, baseline instruction score, construction witness disagreement, positive and negative continuation length, action cost, and tuning-factor boundary selection. A sign reversal in any stratum with at least eight concepts is disclosed and blocks broad wording unless prospectively explained by a measured interaction.

## Stage gates and stopping

One-GPU smoke runs one sentinel concept and only tests replay, gradient finiteness, metric construction, KKT, generation, serialization, and resource use. It cannot select a method or factor.

The 12-concept sentinel passes only if SO.1-SO.4 pass, at least 11 concepts have complete hard actions, `semantic_outcome` wins the held-out worst-view score in at least 9 concepts versus both primary action ablations, mean paired changes are positive, neutral median likelihood loss is no worse than `0.02` nats per token relative to the strongest ablation, all required generations exist, and no severe collapse occurs. These are premise gates, not significance claims.

The 24-concept development passes only if the candidate's concept-clustered HMean difference over the strongest fair reproduced baseline has a Holm-adjusted one-sided 95% lower endpoint above zero; instruction and fluency lower endpoints exceed `-0.05`; neutral likelihood exceeds `-0.02` nats per token; at least 90% of concepts are complete; there is no unexplained protected-stratum sign reversal; and the candidate is Pareto-undominated at matched deployment passes and declared metric cost.

Failure closes this route without changing the witness count, concept allocation, metric ridge after it is source-frozen, factor grid, layer, judge, margins, or comparator identities. A missing official judge blocks performance promotion but does not invalidate a completed mechanism sentinel.

## Registered SOTA gate

For Gemma-2-2B, the candidate must exceed both the strongest complete common-protocol comparator and the published FLAS marker `1.015`, with a multiplicity-adjusted one-sided concept-clustered interval above the strongest reproduced comparator. For Gemma-2-9B, it must analogously exceed the strongest common-protocol comparator and the published marker `1.120`. Both model gates, complete CES/FLAS receipts, feasibility, component, safety, compute, and no-cell-removal gates are required for registered AxBench SOTA wording.
