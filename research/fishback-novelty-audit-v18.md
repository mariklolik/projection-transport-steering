# FishBack novelty audit, version 18

Status: primary-paper inspection complete; version-16 novelty boundary is superseded, while its mathematical and development receipts remain unchanged.

## Source receipt

- Paper: Sihan Wang, Jiayi Zhao, Qingyan Cao, Hongbo Yao, and Lin Shu, *FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers*, arXiv:2605.17231v2, 18 August 2026.
- Primary text: `https://arxiv.org/html/2605.17231v2` and the corresponding TeX source archive.
- TeX source archive: `sha256:1c0070747402893ae9a62bebb3916df3ce08e92c63d2addb5f33d18998e71918`.
- The inspected source contains no active code, data, or project link. Exact-title, arXiv-ID, GitHub, and OpenReview searches found no author repository or review forum. This is a bounded search result, not evidence that no private or later artifact exists.

## Method extraction

At intervention layer $\ell$, FishBack linearizes the remaining network as $\Delta\lambda=J\delta h$ and uses the softmax Fisher matrix $H$ to obtain the pullback metric

\[
G=J^\top HJ.
\]

It then solves

\[
\min_{\delta h}\frac12\delta h^\top G\delta h
\quad\text{subject to}\quad
q^\top\delta h=\rho,
\qquad q=J^\top\beta_W,
\]

with minimum-norm solution

\[
\delta h^\star=\frac{\rho}{q^\top G^\dagger q}G^\dagger q.
\]

The paper also proves a proxy-metric cost ratio for any $M\succ0$ and any concept covector $q$. Its equality condition is exactly alignment of $M^{-1}q$ with $G^{-1}q$. The practical 8B action regularizes the solve, retains the orthogonal Fisher component, fixes a minimum concept efficiency, and iterates for 30 steps. A rank-50 randomized approximation uses 140 metric-vector products, top-1000 Fisher truncation, and a Jacobian frozen at the base point.

## Experimental extraction

The main panel uses GPT-2 Small, Llama-3-8B, and Qwen3-8B; three verb-morphology concepts; early, middle, and late layers; and four matched concept-probability targets. Every off-target comparison is localized by bisection at the same reached target. Comparators are CAA, ActAdd, ITI, ReFT, Representation Surgery, a Euclidean ablation, and a best-per-case oracle.

The pooled early-and-middle-layer median baseline-to-FishBack off-target-KL ratios are 1.42--6.52 on GPT-2 Small, 2.20--3.50 on Llama-3-8B, and 1.77--3.61 on Qwen3-8B. The Euclidean-ablation ratios are 1.30, 1.79, and 2.45. The paper reports one-sided paired Wilcoxon tests on log ratios with Benjamini--Hochberg correction, per-cell denominators, reachability, IQRs, and a deepest-layer failure regime where the ablation gap closes. Reported compute is 31.4, 10.8, and 30.8 GPU-hours for the three model panels.

## Strengths and limitations

The strongest design choices are exact target matching, a geometry-only ablation with the same covector and calibration, an oracle comparator, explicit reachability, multiplicity correction, a matrix-free 8B implementation, and a predicted depth-dependent failure regime.

The evidence is narrow in behavioral scope: all targets are single-step verb inflections selected from C4 contexts where the concept already dominates the next-token distribution. Matched comparisons condition on both methods reaching a target, so baseline denominators shrink and differ. The pooled headline ratios have IQRs and corrected tests rather than cluster-level effect intervals. The 8B method costs hundreds of partial forward equivalents per case, freezes the Jacobian, and has no inspected public code or raw-table artifact. No primary peer review or rebuttal was found.

## Novelty decision

FishBack occupies the broad claim that a Fisher or output-functional metric yields a minimum-distortion activation-steering action. It also occupies the natural-gradient form, iterative re-evaluation, proxy-metric excess-cost analysis, low-rank matrix-free approximation, and 8B empirical validation. Therefore:

1. version 16's pointwise optimizer remains mathematically correct but is not a standalone novel steering principle;
2. the observed cheap generic-basis KL reduction remains project-specific development evidence, because FishBack uses a target-specific covector and a per-case pullback metric;
3. a Fisher trust-region reparameterization or equal-metric-budget dual is not a sufficient successor contribution;
4. any successor must be centered on a materially different controlled object and demonstrate value beyond FishBack, such as amortizing the pullback action across concepts and tokens while preserving FishBack fidelity, deployment cost, broad behavioral efficacy, and safety under a common protocol.

No new GPU experiment is authorized by this audit. The next method must first survive a nearest-neighbor search for amortized, distilled, and reusable pullback-Fisher steering and then freeze a direct FishBack comparison or a justified reproducibility substitute.
