# Complete restricted baseline-v3: adverse selection analysis

2026-09-08. Research audit, not submission prose. All eight original workers are terminal, exit 0, without OOM or restart. All 2,048 candidate outputs and their final producer/base/checkpoint/cost joins pass the native CPU audit. The scientific entry criterion fails.

## Decision and evidence boundary

The best full-generation candidate is c05 (rank 4, site 23), with **133/256 primary successes versus 198/256 for zero**. Its paired difference is −25.390625 percentage points (pp), exploratory 95% BCa interval [−32.03125, −19.921875]. The prompt-only winner is c01 (rank 4, site 23), with 153/256. All eight candidates lose to zero.

The original criterion requires the best full candidate to strictly exceed 198. It does not. Close this finite action family before response collection. No response/controller rollouts, vector library, strength sweep, changed product, cap, grader, training schedule, or selection rule follow from these results. Version 20 and earlier scientific closures stay closed. The broad SOTA research goal remains active; this negative baseline does not constitute the required paper.

Authoritative aggregate: [native result](../artifacts/selection/irc_baseline_v3/family-analysis-v1/result.json), SHA-256 `03433c5dc63d9f8ec16308e34a58754eccfb8cc36078d0b1c54a08c584aa8467`. Its 608-entry input map binds raw/component/reuse evidence; the full packet additionally binds the original config, launch, pinned source/helpers and CPU execution provenance. [Aggregate execution](irc-baseline-selection-aggregate-execution-v3.json) retains exact source, argv, terminal and controller output.

## Frozen design and statistical unit

The complete family is rank 4/8 × post-block site 17/23 × prompt-only/full application on the pinned Qwen3-8B revision. Each fit uses the same 1,500 admitted public solutions, twelve epochs, one seed, 564 updates, effective batch 32 and final raw checkpoint at unit strength. Both fitting-position policies were separately trained. The [original execution protocol](irc-baseline-execution-protocol-v3.md) and [config](irc-baseline-config-v3.json) remain unchanged.

Each condition answers the same 256 selection questions with one rollout per question, the original per-question seed, temperature 0.6, top-p 0.95, top-k 20 and 8,192-token cap. Primary scoring retains the author grader; cap failures remain false even with a parsed answer. The predeclared Math-Verify support is fixed at 255 questions and is sensitivity analysis, never replacement gold.

There are 256 independent paired question units, not 2,048 independent questions. The nine conditions contain 2,304 outputs including the shared zero reference. The eight fits are not eight independent seeds. One aggregation execution uses the unchanged paired-question BCa helper to produce 253 interval records, configured for 10,000 resamples, seed 20260907 and 95% confidence. One degenerate cell returns its unchanged zero-width interval without resampling. Intervals are exploratory, neither simultaneous nor corrected for winner selection, and do not estimate training-seed uncertainty. No confirmatory test or cross-model result is reported.

## Complete candidate denominator

Differences and intervals are percentage points versus zero. Every original candidate is shown. The sensitivity column uses the same fixed 255-question support in every row.

| Condition | Primary accuracy | Difference, pp | Exploratory 95% BCa, pp | Cap failures | Mean output tokens | Sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| zero | 198/256 (77.34%) | — | — | 43 | 4467.8 | 203/255 |
| c00 prompt, r4/s17 | 125/256 (48.83%) | -28.52 | [-35.16, -23.05] | 8 | 499.1 | 129/255 |
| c01 prompt, r4/s23 | 153/256 (59.77%) | -17.58 | [-24.61, -12.11] | 17 | 904.7 | 155/255 |
| c02 prompt, r8/s17 | 118/256 (46.09%) | -31.25 | [-38.28, -25.39] | 6 | 450.7 | 123/255 |
| c03 prompt, r8/s23 | 152/256 (59.38%) | -17.97 | [-24.22, -12.50] | 9 | 763.1 | 156/255 |
| c04 full, r4/s17 | 110/256 (42.97%) | -34.38 | [-41.41, -28.13] | 8 | 503.2 | 113/255 |
| c05 full, r4/s23 | 133/256 (51.95%) | -25.39 | [-32.03, -19.92] | 8 | 567.9 | 136/255 |
| c06 full, r8/s17 | 112/256 (43.75%) | -33.59 | [-40.63, -27.34] | 2 | 309.6 | 113/255 |
| c07 full, r8/s23 | 127/256 (49.61%) | -27.73 | [-34.77, -21.48] | 11 | 680.7 | 131/255 |

The unsupported question remains in the primary denominator and is primary-false in every condition. All eight effects remain negative on supported-primary and supported-sensitivity scoring: no observed direction reversal. Disagreements are preserved, not adjudicated into new labels. A lower cap count or shorter response is not an accuracy improvement.

## Paired harm and rescue

Rescue means zero-false/candidate-true; harm means zero-true/candidate-false. These are primary-score transitions, not a manual proof of reasoning correctness.

| Candidate | Rescues / 58 | Harms / 198 | Both correct | Both false | Harms ending EOS with parsed answer |
|---|---:|---:|---:|---:|---:|
| c00 | 6 | 79 | 119 | 52 | 76 |
| c01 | 15 | 60 | 138 | 43 | 54 |
| c02 | 9 | 89 | 109 | 49 | 85 |
| c03 | 11 | 57 | 141 | 47 | 54 |
| c04 | 8 | 96 | 102 | 50 | 89 |
| c05 | 8 | 73 | 125 | 50 | 69 |
| c06 | 7 | 93 | 105 | 51 | 93 |
| c07 | 12 | 83 | 115 | 46 | 77 |

For c05, eight rescues are outweighed by 73 harms: rescue rate 13.79%, harm rate 36.87%, retention 63.13%. Of its 73 harms, 69 end normally with a parsed primary-false answer and four hit the cap; none is an EOS extraction failure. Six of eight rescues come from reference cap cases and two from reference EOS primary-false cases. This contradicts cap/extraction failure as a sufficient explanation of the observed quality loss, but does not establish which mathematical reasoning operation failed.

The prompt winner repairs 15 reference failures and harms 60 reference successes. Comparing the selected full against selected prompt gives 32 rescues and 52 harms, a net −20/256 = −7.8125 pp (exploratory 95% BCa [−15.234375, −1.171875]).

All four rank/site-matched full candidates are worse than their separately trained prompt counterparts, by 15, 20, 6 and 25 correct answers. This is descriptive evidence against the tested full-policy recipe. Because training masks and learned weights differ, it is **not** an isolated causal estimate of adding a decoding-time intervention.

## Termination and extraction are not the main endpoint

The following partition is exhaustive and mutually exclusive: a capped output is assigned to cap even when an answer can be parsed.

| Condition | Primary correct | Cap exhausted | EOS, extraction failed | EOS, parsed, primary false |
|---|---:|---:|---:|---:|
| zero | 198 | 43 | 0 | 15 |
| c00 | 125 | 8 | 0 | 123 |
| c01 | 153 | 17 | 0 | 86 |
| c02 | 118 | 6 | 0 | 132 |
| c03 | 152 | 9 | 0 | 95 |
| c04 | 110 | 8 | 1 | 137 |
| c05 | 133 | 8 | 0 | 115 |
| c06 | 112 | 2 | 0 | 142 |
| c07 | 127 | 11 | 0 | 118 |

c05 reduces caps from 43 to 8, yet loses 65 correct answers overall. Its mean output length falls from 4,467.76 to 567.89 tokens and its median from 3,864 to 194. The paired mean reduction is 3,899.87 tokens (exploratory 95% BCa reduction [3,597.72, 4,197.18]). c06 terminates 254/256 times with only two caps, but answers just 112 correctly. Reliable termination alone is therefore not sufficient to establish a useful action in this family.

All eight online epoch losses decreased; each full-policy final online loss is lower than its rank/site-matched prompt-policy loss, while each full policy has lower observed selection accuracy. The [fit analysis](irc-baseline-fit-analysis-v3.md) records that these are changing-parameter online losses, not a fixed-checkpoint validation curve. Do not choose another epoch, change the objective retrospectively, or infer convergence from them.

## Predeclared source strata for c05

These strata describe the selected full candidate. Small cell sizes, post-selection intervals and one model/seed prevent broad generalization. All seven source-subject point estimates and all five difficulty point estimates are negative.

| Source subject | Questions | Difference, pp | Exploratory 95% BCa, pp |
|---|---:|---:|---:|
| Algebra | 58 | -15.52 | [-29.31, -6.90] |
| Counting & Probability | 21 | -33.33 | [-61.90, -19.05] |
| Geometry | 25 | -32.00 | [-56.00, -16.00] |
| Intermediate Algebra | 57 | -29.82 | [-43.86, -19.30] |
| Number Theory | 28 | -25.00 | [-50.00, -7.14] |
| Prealgebra | 46 | -23.91 | [-41.30, -10.87] |
| Precalculus | 21 | -28.57 | [-57.14, -14.29] |

| Source difficulty | Questions | Difference, pp | Exploratory 95% BCa, pp |
|---|---:|---:|---:|
| Level 1 | 16 | -43.75 | [-75.00, -25.00] |
| Level 2 | 53 | -16.98 | [-32.08, -7.55] |
| Level 3 | 57 | -24.56 | [-38.60, -15.79] |
| Level 4 | 55 | -29.09 | [-45.45, -16.36] |
| Level 5 | 75 | -25.33 | [-37.33, -16.00] |

Prompt-length effects are −26.21 pp for <128 tokens (n=206), −12.50 pp for 128–255 (n=40; interval [−32.50, +2.50]) and −60.00 pp for ≥256 (n=10). The tiny longest-prompt cell is not a population law or permission to exclude hard inputs.

On the 213 reference-EOS questions the difference is −33.33 pp; on 43 reference-cap questions it is +13.95 pp. Reference outcomes are available here because this is an offline paired audit: they cannot serve as an oracle deployment gate. Candidate-output-length strata in the native result are explicitly post-treatment and do not establish a causal mechanism.

## What the failure does and does not identify

| Explanation | Evidence from this packet | Remaining uncertainty / decision |
|---|---|---|
| Incomplete execution or missing members explains this result | All eight complete; original producer/base/source/checkpoint joins pass | Bounded engineering checks are not a proof of arbitrary engine correctness, but a survivor or missing-panel explanation does not apply |
| Cap or extraction failures explain the whole accuracy loss | c05 has 69 normally terminated, parsed harms; sensitivity keeps all effects negative | Primary/sensitivity disagreement is not adjudicated reasoning truth; do not relabel or change cap |
| Short-solution teacher forcing misaligns the reasoning policy | Large shortening, low online loss and worse accuracy occur together | Plausible mechanism, not identified cause; objective, model target distribution and capacity are not factorially isolated |
| Full application is intrinsically inferior | Four matched full recipes lose | Training policies and weights differ; no decoding-only causal conclusion, architecture-wide refutation or general ReFT refutation |
| Response control could rescue a harmful static policy | Not measured; this family fails its prospective useful-action entry gate | Do not launch response collection to repair a failed prerequisite post hoc |

The [pre-outcome design](irc-fixed-adapter-design-v1.md) already warned that public source solutions are much shorter than allowed generated reasoning. That warning is evidence of a prior risk, not advance evidence of its causal mechanism. The restricted one-layer family is not the strongest all-layer ReFT or parameter-matched LoRA comparator; defeating zero would only have opened an initialization gate. It did not even reach that gate.

## Execution, failures and actual resource accounting

The original eight GPU workers were not relaunched in this analysis turn. Selection files were captured only after each respective worker was terminal, in readiness groups c06, c0245 and c137. These are collection groups, not scientific selection rungs. Every new candidate output was audited exactly once on the pinned native CPU path. Hash-matched zero grading and fit/checkpoint audits were reused; no unchanged model, grader, bootstrap or generic test stage was repeated.

| Capture group | Candidate rows | Archive payloads | Archive SHA-256 |
|---|---:|---:|---|
| c06 | 256 | 70 | `ef61abd30569c7c19e254f1a78cc0f92422ea45d36e7185a6ceb92478270ad05` |
| c0245 | 1,024 | 274 | `b6a4e3583217b795839115857a3f635306db37b5645e03e7e8b7769c85091d12` |
| c137 | 768 | 206 | `8941d10db343438fbec1241d02fd69a7575a09d1100d69a7ca3808eebca60980` |

The first c06 capture failed before archive creation or CPU grading because equivalent Docker mounts arrived in a different list order. A source-scoped canonical sort by destination fixed only capture comparison; the original failure and diagnostic are preserved in [c06 execution](irc-baseline-selection-execution-c06-v3.json). No experimental output changed.

Before any aggregate execution, independent review found that a late input-hash failure could leave a stale PASS status. An AST-extracted handler fault injection failed RED, then passed GREEN after one fail-closed status assignment. The exact trace and source delta remain in the aggregate execution record. The aggregate itself ran once, exit 0. These capture/reporting defects and reviewer-local checker errors are not model failures or hidden reruns.

All original worker lifetimes are within the inclusive 18,000-second ceiling. Times below are seconds; fit and selection components are **inside**, not additional to, container charges.

| Worker | Actual fit | Selection generation | Inclusive container lifetime |
|---|---:|---:|---:|
| c00 | 2569.727 | 2552.694 | 6399.735974 |
| c01 | 2133.962 | 5070.384 | 8539.357582 |
| c02 | 2603.121 | 2789.532 | 6648.786933 |
| c03 | 2154.459 | 4440.981 | 7894.619216 |
| c04 | 2578.122 | 3000.872 | 6789.189909 |
| c05 | 2112.507 | 3481.246 | 6836.374413 |
| c06 | 2595.748 | 1465.976 | 5324.078215 |
| c07 | 2200.869 | 3792.117 | 7180.866527 |

v3 costs **15.4480579914 H100-hours**. Adding the preserved v2 failed/cancelled cost of 7.6053742547 gives **23.0534322461 H100-hours** against the 48-hour baseline ceiling, leaving 24.9465677539. Historical engineering remains separately charged, not free. Unspent capacity does not authorize a response stage, an automatic retry or silent budget transfer.

Disjoint nested totals are 18,948.514608 fit seconds, 9,846.580574 zero-generation seconds and 26,593.801261 selection-generation seconds; grading/checking and startup/teardown remain separately identified in receipts. Update timing is nested inside fit. Three CPU audits take 51.550089 seconds in total; aggregate analysis takes 15.032862 seconds. CPU-controller duration is separate. Native container timestamps are converted at microsecond resolution.

At the [saved 07:57:13 UTC native observation](irc-baseline-selection-terminal-observation-v3.json), all eight original containers were terminal; all eight H100s reported 1 MiB used and 0% instantaneous utilization. This timestamped snapshot and shorter outputs do not prove average GPU utilization, current ownership after that time, serving latency or speedup.

## Research delta and next dependency

This increment changes the evidence from “all fits completed, behavioral effect unknown” to “complete restricted action family fails its unchanged entry criterion.” It removes incomplete execution as the explanation for missing efficacy evidence and identifies the dominant observed harm partition. It does not support the intended positive method.

For the full empirical SOTA claim, the unchanged strength vector is [3,2,0,3,0,0,0,4,3] = 15/36; classification remains EVIDENCE_THIN with SCIENTIFIC_WEAKNESS. S1/S4 are anchored in the frozen benchmark; S2 in the verified literature synthesis and still-missing strongest comparator admissions; S3/S5/S7 fail for the proposed controller contribution, which has no valid positive experiment or strong comparison; S6 lacks the required three-family/three-construct evidence; S8/S9 are supported by preserved adverse outputs and source-bound manifests. This does not score the correctness of this narrow negative audit as zero. Presentation [3,3,4] is separate and cannot compensate.

The next primary research increment is a **pre-outcome learning-objective and comparator design**, not response control on these adapters. Its reviewer question is whether a useful, retention-preserving finite action can be learned in the base model's own reasoning distribution under a reward-aligned objective, and whether any advantage survives a strong parameter/information/budget-matched adaptation comparator. This is an untested hypothesis, not a new novelty or SOTA claim.

First reuse the existing primary-source synthesis and inspect the precise target/objective and comparator implementations needed for that question. Distinguish public-solution imitation, on-policy outcome supervision and anchored adaptation; verify the nearest applicable methods from primary sources before expensive work. Freeze one minimal discriminating experiment, exact target provenance, available training-only information, comparator budget, seeds, effect/retention/uncertainty criteria and full inclusive costs **before any new outcomes**. Do not silently replace target sources, create manual labels, select a post hoc vector library, or inherit an untested setting as “strongest.”

The failure outcome must be explicit: if the new action is not useful against zero under its prospective quality/retention criteria, close it before controller learning; if it does not beat the closest matched method, make no novelty or superiority claim. The currently exposed selection questions remain exploratory; development and sealed-test boundaries are not reassigned or opened by this audit. A later positive initialization still needs the independent controller comparison, protected utility, uncertainty, mechanism test, strongest comparator frontier and the full three-architecture/three-construct scope.

[Independent AI review](irc-baseline-selection-independent-review-v3.json) and the final closure record bind the audit and assistance provenance. Human author approval, disclosure adjudication and current venue/year/track compliance are unverified. Paper writers, skeleton, global review and submission compliance are not reached.
