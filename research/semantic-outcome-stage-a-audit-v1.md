# Semantic-outcome Stage A audit, version 1

Decision date: 2026-09-07. Status: valid exploratory premise result; SemanticOutcomeBack version 1 closed before open generation.

## Experiment contract and provenance

The immutable source is `artifacts/source/semantic_outcome_v1_sentinel.json`, SHA-256 `59974429e77e1853beaa8f8a6f84e07ef0fb142feac7a04c01554fdf64e9ad68`. The shared neutral metric comes from the admitted one-concept smoke `artifacts/development/semantic_outcome_v1_smoke_c1_attempt3/actions.pt`, SHA-256 `fb6236abebc91010e57355e7f9ea93079863d8f21108f3eba374f15eca448c20`. The six-shard Stage A packet has listing SHA-256 `d644c6d3132db749a008afaa7db17c89a733e8689b31e3bf055c78341649fa77`. The prospective analyzer produced `artifacts/development/semantic_outcome_v1_stage_a_analysis_attempt1/analysis.json`, SHA-256 `cb9db376f34bd5c853aea546984a6c4e8d7f5a43546fc6063fe6dc88b1490ebb`.

The independent unit is a concept. All 12 planned exposed sentinel concepts completed all 24 method-factor cells; every held-out panel contains eight positive-negative continuation pairs and every neutral panel contains 16 rows. Prompts and continuations are nested diagnostics, not independent replicates. No concept, witness, method, factor, or failure was excluded. This is an exploratory premise sentinel: it cannot establish behavior, SOTA, model-family generality, or confirmatory uncertainty.

## Integrity and compute

All 12 mean-score, Euclidean-robust, and last-token hard systems passed feasibility and KKT checks. All six action methods were matched to the candidate's declared diagonal-metric cost within the frozen numerical tolerance. Zero-action replay error was exactly zero on every shard. The six workers used one H100 each, one model load for two concepts, two CPU threads, 192 backward products and 720 explicitly counted teacher-forced forward calls. The maximum shard wall time was 28.46 seconds and maximum allocated GPU memory was 30.41 GB. Matched-negative autoregressive decode steps are additional model passes and are not hidden inside the 720-forward count; no serving-efficiency claim is made.

## Primary Stage A result

The table reports concept-level worst-view change relative to no-op. Candidate contrasts are paired across the same 12 concepts. BCa intervals are descriptive because the sentinel is exploratory and all four factors were inspected.

| Factor | Candidate mean worst-view change | Wins vs pooled sequence metric | Candidate minus pooled mean worst-view change, BCa 95% | Wins vs robust Euclidean | Candidate minus robust-Euclidean mean worst-view change, BCa 95% | Premise gate |
|---:|---:|---:|---:|---:|---:|---|
| 0.25 | -0.00180 | 3/12 | -0.00052 [-0.00166, 0.00135] | 8/12 | +0.00133 [0.00037, 0.00233] | fail |
| 0.50 | +0.00121 | 5/12 | -0.00051 [-0.00122, 0.00064] | 10/12 | +0.00326 [0.00180, 0.00462] | fail |
| 1.00 | +0.00696 | 5/12 | -0.00033 [-0.00170, 0.00101] | 12/12 | +0.00522 [0.00351, 0.00721] | fail |
| 1.50 | +0.01332 | 7/12 | -0.00082 [-0.00280, 0.00079] | 12/12 | +0.00793 [0.00591, 0.01058] | fail |

No factor reaches the frozen 9/12 win requirement against both primary ablations. The candidate-versus-pooled mean worst-view contrast is negative at every factor. The failure is not caused by infeasibility, missingness, neutral damage, or Euclidean geometry; it isolates the simultaneous robust-witness component.

## Component and adverse analysis

At factor 1.5, pooled sequence-metric steering has mean held-out change `0.02631` and mean worst-view change `0.01414`, versus `0.02545` and `0.01332` for the candidate. Robust Euclidean reaches only `0.01377` and `0.00540`; DiffMean reaches `0.00728` and `0.00080`; last-token semantic steering reaches `0.00108` and `-0.00578`. This supports using multi-token sequence scores and the neutral sequence-score metric as useful components, but those components overlap directly with prior sequence-energy and Fisher-style work and do not support the registered contribution.

The candidate and pooled sequence-metric actions are nearly collinear: mean cosine `0.9875`, range `[0.9774,0.9961]`, with nearly equal Euclidean norms for every concept. All eight constraints are active in all 12 candidate systems, but dual concentration is modest, `0.152` to `0.220`. Construction witnesses are positively aligned rather than contradictory; the maximum pairwise witness cosine ranges from `0.376` to `0.561`. The robust solve therefore adds small witness-balancing components that do not improve held-out worst-view control over pooling.

Neutral behavior does not explain the rejection. Across factors, candidate median neutral score change ranges from `-0.00005` to `+0.00085` nats per token and median forward KL ranges from `0.00031` to `0.00068`. Candidate-minus-primary-ablation median neutral differences remain far inside the `-0.02` margin. These are teacher-forced local diagnostics, not general capability or serving evidence.

## Leakage, selection, and multiplicity audit

The construction and latent-evaluation rows are disjoint within every concept, and the global neutral pool is disjoint from both. The same 12 concepts and neutral rows are reused across methods and factors, so all comparisons are paired and concept-clustered. The sentinel itself influenced the route decision and cannot become untouched confirmation for a successor. No factor is promoted: all four remain reported and none passes. The descriptive BCa intervals do not convert the failed registered win-count gate into a significance claim.

No Stage B generation, judge call, 24-concept development row, 64-concept internal-pilot row, 100-concept held-out row, or 9B row was opened. A plot is not generated because the exact paired table and complete concept-level vectors in `analysis.json` answer the stopping decision without implying a behavioral frontier.

## Claim decision

Claim C54 is contradicted under its frozen exposed-sentinel gate. SemanticOutcomeBack version 1 is closed. The route cannot be rescued by changing witness count, constraint margin, ridge, layer, factor grid, endpoint, or comparator. The retained scientific result is narrower: under this source and model, the diagonal sequence-score metric improves the robust Euclidean action, while hard per-witness robustness adds no held-out worst-view benefit over the matched-cost pooled sequence-metric action.

Any successor must be declared as a new post-sentinel hypothesis, treat all 12 concepts as development evidence, and use fresh concepts for its next decision. Paper prose and SOTA claims remain closed.
