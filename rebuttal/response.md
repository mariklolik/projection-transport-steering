# Response to the program-chair feedback

Each item below names the concern, what we ran, and where it now lives in the
manuscript. Numbers are from the regenerated runs unless marked.

## W1. Does the intervention disrupt unrelated open-ended generation?

New experiment, three tasks the directions never saw (Appendix: capability
preservation; Sec. 5 paragraph "The edit costs nothing outside the behavior it
was fitted for").

| operator | HumanEval pass@1 | WikiText-103 ppl | distinct-3 (200 open-ended) | degenerate |
|---|---:|---:|---:|---:|
| unsteered | .396 | 23.4 | .953 | 0% |
| additive −0.75 (published dose) | **.128** | **47.3** | **.480** | **57.5%** |
| additive −0.25 (equal-budget dose) | .408 | 23.6 | .922 | 0% |
| MiMiC | .396 | 24.2 | .949 | 0% |
| MiMiC, tuned layer 18 | .384 | 24.6 | .948 | 0% |
| Linear-AcT | .390 | 23.4 | .950 | 0% |
| clamp q50 (ours) | .390 | 23.1 | .946 | 0% |
| quantile-OT (ours) | .396 | 23.3 | .951 | 0% |
| sequential gate + ablation (ours) | .396 | 23.4 | .946 | 0% |
| sequential gate + full-rank map (ours) | .396 | 23.4 | .954 | 0% |

The additive dose that removes every overconfident-wrong answer costs 68% of
HumanEval pass@1, doubles held-out perplexity, and makes 57.5% of open-ended
generations degenerate. Our operators are at the unsteered value on all three,
even with the action left on for every token. We add the equal-budget dose
because it is the fair comparison and it is not destructive: at −0.25||h|| the
additive edit leaves HumanEval and perplexity alone, which says the parity
protocol selects a dose too small to break the model rather than that the
additive shape is safe — it still leaves 40–70% of tokens off-manifold (W2).
Our two gated rows are the unsteered numbers to the digit on HumanEval and
WikiText because the gate fires on neither (fire rate 0.00), and it does fire
on 59–68% of the open-ended prompts without changing the measurements. A judge (Qwen2.5-7B-Instruct, both presentation orders, a win counted only on
agreement across orders) prefers the unsteered answer to the additive one on
185 of 200 prompts, for an adjusted win rate of 0.043. Every other operator
lands between 0.458 and 0.512 against a tie value of 0.5, the equal-budget
additive dose included at 0.495.

Protocol: HumanEval greedy, n=164, executed unit tests; WikiText-103 test, 300
passages over 400 characters; 200 AlpacaEval instructions, 384 new tokens,
fluency measured as the perplexity the unsteered model assigns to the
continuation.

## W2. Do the maps stay on-manifold under zero-shot transfer?

New experiment on MMLU, ARC and GSM8K (Appendix: on-manifold diagnostics).
We report that the obvious statistic fails and why: a full-space Mahalanobis
distance is diluted by the d−k coordinates a projection-local edit never
touches and moves under 1% for every operator, including the one that destroys
generation. The same dilution is reported independently by In-Distribution
Steering (arXiv:2510.13285). We therefore condition the edited coordinates on
the untouched complement, which is chi-squared_k under the null.

| operator | out% MMLU / ARC / GSM8K | NN distance ratio | KL next token (MMLU) |
|---|---|---|---:|
| unsteered | 0.9 / 1.2 / 2.9 | 1.00 | — |
| random, norm-matched | 1.6 / 1.7 / 0.9 | 1.90 | **0.445** |
| additive −0.75 | **100 / 100 / 100** | 1.88 / 2.25 / 2.04 | 0.032 |
| additive −0.25 (equal-budget) | 69.9 / 40.3 / — | 1.22 / 1.33 | 0.005 |
| MiMiC | 1.3 / 1.8 / 4.5 | 1.05 | 0.001 |
| clamp q50 (ours) | 1.3 / 1.1 / 2.9 | 1.03 | 0.006 |
| quantile-OT (ours) | 1.0 / 1.3 / 4.0 | 1.00 | 0.000 |

Expected out% under the null is 1%. Additive steering puts every token outside
the 99% conditional ellipsoid on all three domains, and further off-manifold
off-domain by nearest-neighbour distance (2.25x on ARC against 1.88x on MMLU),
while the projection-local operators sit at the null everywhere. The KL check
(threshold 0.1, the criterion used for direction selection in the refusal
literature) ranks differently and is reported alongside: a random direction of
the same norm is the most behaviorally disruptive operator we measured.

## W3. Were the baselines given an equal tuning budget? What is the real latency?

**Equal budget.** Every method now receives at least as many
configurations as our own eight on the same held-out MMLU tuning split
(n=1200, disjoint from the extraction split and from all evaluation subsets)
under one pre-declared rule: highest selectivity subject to an accuracy change
of at least −0.01. The split size is itself a result. We first ran the search
on 400 questions; on 1200 six of the nine winners change and the measured
selectivity of a fixed configuration moves by up to 0.131, so a 400-question
split does not identify a configuration, it samples one. We also fix the
batch size across every search, because batched greedy decoding is not
invariant to it: the same MiMiC configuration measures +0.288 at batch 64 and
+0.235 at batch 32. Both facts are reported rather than quietly absorbed,
and every number below is from the n=1200 winners at one batch size. CAST
searches threshold x dose, MiMiC searches layer x shrinkage, Linear-AcT
searches layer x lambda, and our operators search the gate depth, the gate
quantile, the local action, the subspace dimension and the prefix budget.
The winners are reported and then run unchanged on the evaluation seeds
(Appendix: equal-budget tuning; Table 1 is now the tuned comparison, and the
default-configuration grid moves to the appendix).

Tuning materially changes the baselines: the additive dose moves from
−0.75||h|| to −0.25||h|| and its tuning-split selectivity rises from ~0 to
+0.196, CAST reaches +0.182, and MiMiC reaches +0.281 at layer 18, above our
own tuned gated ablation at +0.232.

We did not leave that where it was. Three follow-ups, all inside the existing
theory.

1. *Is the gate what MiMiC lacks?* No. Composing our gate with MiMiC's action,
   which Theorem 3 licenses, reaches +0.155 against +0.263 ungated on the
   reliable split. Note that this operator is not a competitor: a full-space
   Gaussian map is the k = d member of the family Theorem 1 defines, so gating
   it is our own method at full rank, and it is not where the family's
   strength lies.
2. *Is it the subspace dimension?* No. On the n=1200 split selectivity peaks
   at k = 2 (+0.186) and is flat above it (+0.148 at k = 8 and again at
   k = 32), so the two interpretable axes already carry this behavior. The
   monotone rise we saw at n=400 was selection noise.
3. *Is it the all-or-nothing action?* Yes. Letting the strength of the action
   follow the gate score instead of switching on it — the interpolation
   g_lambda the paper already defines — reaches **+0.267** under the same
   eight-configuration budget, the best configuration any method gets
   (additive +0.271, MiMiC +0.263, our hard gate +0.245). The selected
   configuration is not the conservative one we expected: it fires at the 10th
   percentile and scales each edit by its margin, buying 0.607 removal at
   0.660 retention where the hard switch bought 0.527 at 0.718. Grading the
   dose is what makes a permissive threshold affordable.

The decisive comparison is what survives carrying those winners unchanged to
five resampled question sets:

| operator (M4, 5 seeds) | Sel | d_acc | cr_keep | ECE |
|---|---:|---:|---:|---:|
| additive CAA (tuned) | +0.256 ± 0.133 | −0.004 | 0.621 | 0.080 |
| **score-proportional dose (ours)** | +0.231 ± 0.120 | −0.005 | 0.650 | 0.089 |
| CAST-style (tuned) | +0.227 ± 0.082 | −0.007 | 0.795 | 0.072 |
| **gated local action (ours)** | +0.206 ± 0.059 | −0.015 | 0.713 | 0.089 |
| Linear-AcT (tuned) | +0.146 ± 0.141 | +0.001 | 0.868 | 0.101 |
| gated full-rank (ours) | +0.146 ± 0.042 | +0.003 | 0.854 | 0.108 |
| MiMiC (tuned) | +0.132 ± 0.103 | −0.015 | 0.763 | 0.113 |

Protocol note: the split is n=1200 rather than n=400 and every search runs at
one batch size, both for reasons that are results in their own right (see
above). The 9 configurations MiMiC receives exceed our 8.

We also checked the one place our own theory said we had been sloppy.
Theorem 3 gives the cost-minimal gated intervention as the transport map of
the law *conditioned* on the gate event, and every gated transport we had run
used the marginal law. We fitted the conditional moments on the extraction
split at four gate quantiles and compared: the conditional map is not better
(+0.155 vs +0.183 at the 20th percentile, +0.153 vs +0.125 at the 30th, no
systematic ordering). That is Theorem 3 delivering exactly what it claims —
minimum displacement subject to the target law — and nothing more, because
selectivity here is bound by the detector, not by the map. Five actions inside
the same gate (larger subspace, full-rank map, graded dose, token-local clamp,
conditional map) each bought removal at the same rate in retention, which is
what Corollary 1 predicts for a subpopulation pair separated by d = 0.049.
The design ablation appendix reports all five.

MiMiC keeps half of what the layer search bought it. Our gated local action
moves least of anything in the table, ±0.059 against ±0.133 for the tuned
additive dose, and the score-proportional dose is second on selectivity while
holding accuracy. We state plainly that equal tuning makes the additive and
CAST baselines genuinely competitive on MMLU selectivity; what separates the
operators is what they cost elsewhere. On GSM8K-MCQ, where the fitted tail is
absent, every conditional operator of ours is the exact identity (d_acc 0.000,
retention 1.000) while the same tuned additive dose removes all eleven
confident-right answers and the tuned CAST gate costs 6.3 accuracy points. On
ARC-Challenge our score-proportional dose reaches +0.342 at retention 0.815
against +0.184 for CAST and +0.151 for MiMiC.

**Latency.** Measured end-to-end on the whole 300-question workload on one
idle H100, with the post-hoc gate charged for its scoring pass and for
regenerating what it flags. A repeat of the unsteered pass at the end of the
run differs from the first by 0.5%, so the measurement is trustworthy.

| method | passes | seconds | vs unsteered |
|---|---:|---:|---:|
| unsteered | 1 | 43.5 | 1.00x |
| additive | 1 | 45.2 | 1.04x |
| ablation | 1 | 44.6 | 1.02x |
| clamp q50 | 1 | 41.4 | 0.95x |
| quantile-OT | 1 | 44.8 | 1.03x |
| MiMiC | 1 | 42.9 | 0.99x |
| Linear-AcT | 1 | 40.9 | 0.94x |
| gated ablation, post-hoc gate | 2 | 85.8 (223/300 regenerated) | **1.97x** |
| gated ablation, sequential gate | 1 | 45.5 | **1.05x** |

Two things follow, and we state both. The wall clock does not separate the
single-pass operators from one another at this scale: every one of them,
MiMiC's full-space map included, lands within 6% of the unsteered time,
because the extra arithmetic hides behind a memory-bound decode. What it does
separate is the number of passes, which is exactly what the sequential gate of
Theorem 4 removes.

## W4. Mathematical and notational corrections

- Theorem 1 and Proposition 2: the map is now `T*(h) = h + V^T(S(Vh) − Vh)`.
- Proposition 1 is reproved without the y-space isomorphism, which does not
  preserve the duality between the measurement matrix and the displacement
  subspace. The problem is posed directly in the k-dimensional latent space
  under the cost `M_V = V M V^T`, and the optimal map is
  `M_V^{-1/2} . grad phi . M_V^{1/2}`.
- Corollary 1's selectivity clause now states its premise: equal projection
  laws give equal displacement laws, so no choice of g can discriminate; the
  marginal change rates coincide under the stated conditional independence,
  and without it the residual gap is a property of the downstream geometry and
  not reachable by tuning g.
- Theorem 2(a): the log-likelihood ratio is affine in the trace feature vector
  z, and thresholding the induced scalar is the LR test.
- Brenier: absolute continuity with respect to Lebesgue measure, not
  atomlessness.
- Cohen's d is named at first use; the pushforward is written (V . T)_#.

## W5. Presentation and verifiability

- Linear-AcT now appears in the main head-to-head with seed variance.
- Gated ablation and the sequential gate appear in the transfer table and in
  the curated MMLU table, both of which are now generated from the analysis
  files rather than typed.
- Table 2 (9B) gains the missing ECE column: the 0.144 that appeared under
  cr_keep was indeed the ECE.
- Appendix "Definitions used in the text" gives the hedge-marker lexicon, the
  derivation of the calibrated zone, the construction of the norm-matched
  spherical action and its controls, and a full mapping from run identifiers
  to the names used in the text.
- Absolute baseline accuracy is reported in the baseline row of every
  multi-seed table.
- The Pareto figure is generated from the analysis files, so its coordinates
  cannot drift from the tables.
- Figure axis labels, the mmlu-5125 trace label, the `\boxed` escaping
  artifact and the self-referential appendix pointer are fixed.

## W6. Relation to concurrent work on overconfidence and collateral damage

Related work now separates a paragraph on deciding during generation, which
covers CAST, DSAS (arXiv:2512.03661), GAPS (arXiv:2609.01878), steerability
prediction (arXiv:2606.11599) and YOPO (arXiv:2608.14465). We do not claim
novelty for deciding inside the loop; the claim is a gate whose online decision
is tied to the post-hoc one at a stated rate. Two findings from that literature
shape the design and are cited as such: reading a direction off already-steered
activations degrades it, which our reader-above-actor order avoids, and a
feature that detects a behavior in finished text predicts it poorly from a
prefix (arXiv:2606.11172), which is why we refit the gate head on the prefix
budget and report the budget-AUROC curve. For PCHI we note that its only
additive baseline is ungated, so gating is confounded with the multiplicative
head parameterisation, and our grid fills that cell. The missing arXiv id for
"Wired for Overconfidence" (2604.01457) is added.
