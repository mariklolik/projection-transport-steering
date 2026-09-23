# Competitor methods — verified citations and implementation detail

Prepared for the rebuttal. All four papers were retrieved and read in full from the
arXiv HTML (v1) renderings on 2026-09-21.

---

## 0. Citation audit (read this first)

| Draft label | Draft claim | Verdict | Correct identifier |
|---|---|---|---|
| PCHI | arXiv:2606.09876 (2026) | **CORRECT** | 2606.09876 |
| Wired for Overconfidence | "COLM 2026", no arXiv id | **INCOMPLETE** — venue is right, id was missing | **arXiv:2604.01457**, COLM 2026 |
| COAST | arXiv:2605.01167 (2026) | **CORRECT** | 2605.01167 |
| A-LQR | arXiv:2604.19018 (2026) | **CORRECT** | 2604.19018 |

Three of the four ids are fine. The only defect is the "Wired for Overconfidence"
entry, which carries a venue but no arXiv id; the arXiv comments field on 2604.01457
literally reads `COLM 2026`, so the venue claim checks out.

Two further corrections to watch for in the bibliography:

* A-LQR's third author is **Xinyue Annie Yang** (arXiv metadata), which some
  secondary sources abbreviate to "X. A. Yang".
* The A-LQR title capitalises as *"Local Linearity of LLMs Enables Activation
  Steering via Model-Based Linear Optimal Control"* (title case), not the
  lowercase form in our draft.

### Bibliography entries as they should appear

```bibtex
@article{li2026pchi,
  title   = {Calibrating Overconfidence Without Sacrificing Confidence:
             Probe-Conditioned Head Intervention for LLMs},
  author  = {Li, Ke and Zhang, Chongzhe and Zeng, Zifan and Liu, Feng and
             Zhang, Qunli and Hu, Zheng},
  journal = {arXiv preprint arXiv:2606.09876},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.09876},
  note    = {11 pages, 4 figures; cs.LG; submitted 2 June 2026}
}

@inproceedings{zhao2026wired,
  title     = {Wired for Overconfidence: A Mechanistic Perspective on Inflated
               Verbalized Confidence in LLMs},
  author    = {Zhao, Tianyi and He, Yinhan and Zheng, Wendy and Zhang, Yujie and
               Chen, Chen},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  eprint    = {2604.01457},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2604.01457}
}

@article{nguyen2026coast,
  title   = {Minimizing Collateral Damage in Activation Steering},
  author  = {Nguyen, Tam and Nguyen, Tu Anh and Alemohammad, Sina and
             Baraniuk, Richard G.},
  journal = {arXiv preprint arXiv:2605.01167},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.01167}
}

@article{skifstad2026alqr,
  title   = {Local Linearity of LLMs Enables Activation Steering via
             Model-Based Linear Optimal Control},
  author  = {Skifstad, Julian and Yang, Xinyue Annie and Chou, Glen},
  journal = {arXiv preprint arXiv:2604.19018},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.19018},
  note    = {Code: https://github.com/trustworthyrobotics/lqr-activation-steering}
}
```

Dates: PCHI submitted 2026-06-02; COAST 2026-05-01; A-LQR 2026-04-21;
Wired for Overconfidence 2026-04-01 (v1), revised 2026-07-27.

---

## 1. PCHI — Probe-Conditioned Head Intervention

**Exact title.** Calibrating Overconfidence Without Sacrificing Confidence:
Probe-Conditioned Head Intervention for LLMs
**Authors.** Ke Li, Chongzhe Zhang, Zifan Zeng, Feng Liu, Qunli Zhang, Zheng Hu
**Venue.** arXiv preprint, cs.LG, 2 June 2026. 11 pages, 4 figures. No venue given.
**URL.** https://arxiv.org/abs/2606.09876 · HTML: https://arxiv.org/html/2606.09876v1
**Code.** **No code release.** The paper contains no GitHub link, no anonymous
repository and no data/artifact statement. We will have to reimplement from the text.

This is the closest competitor to Projection-Transport Steering: it is the only one
of the four that is explicitly gated (an intervention that fires only on flagged
examples) and explicitly optimises the confident-right / confident-wrong trade-off.

### 1.1 Task setup

The model is forced into a fixed JSON schema with three fields:

```
{"reasoning": ..., "answer": ..., "is_confident": "yes" | "no"}
```

Each response is parsed into an answer `a_i`, a correctness bit `c_i`, and a
verbalized confidence `r_i ∈ {yes, no}`, giving four groups: **CY** (correct-yes),
**CN**, **WY** (wrong-yes), **WN**. PCHI targets WY and protects CY.

The **confidence readout** is measured as a yes/no logit gap at the
confidence-value prediction position (Eq. 1):

```
Δ_i = LSE_{v ∈ V_yes} o_{i,v}  −  LSE_{v ∈ V_no} o_{i,v}
```

with `o_i` the next-token logit vector. `Δ_i > 0` ⇒ yes.

The **confidence template** is the token span between the answer value and the
`is_confident` value — six tokens, indexed 1..6, with token 6 the readout token
whose forward pass emits the yes/no logits. Token 3 corresponds to the `_conf`
sub-word and is a known weak position.

### 1.2 What is fit (three frozen/learned pieces)

**(a) The probe — frozen, trained offline.**
For each template coordinate `t ∈ {1..6}` and each layer `ℓ`, a *separate*
position- and layer-specific linear probe on the post-layer hidden state
`h^{(ℓ)}_{i,t} ∈ R^d`:

```
p^{(ℓ)}_{i,t} = σ( w_{t,ℓ}ᵀ h^{(ℓ)}_{i,t} + b_{t,ℓ} )        (Eq. 2)
```

* Probe class: **calibrated diagonal-LDA** (i.e. LDA with a diagonal covariance
  estimate, then probability calibration). Not logistic regression.
* Training data: **positives = WY responses, negatives = CY responses.** CN and WN
  are not used at all.
* Data source: 5,000 problems sampled from the OpenMathInstruct *training* split
  (after the invalid-example filtering of Nam et al. 2025); one greedy
  (temperature 0) generation per problem; hidden states harvested at the six
  template coordinates.

**(b) Probe-layer selection.**
Criterion: training-set AUROC heatmaps of WY-vs-CY over (template position × layer).
Pick `ℓ_p` where (i) separability is already high, and (ii) enough layers remain
downstream in the same forward pass. Validation split is never used for this.
Result: **Qwen3-4B → ℓ_p = 18; Gemma3-4B → ℓ_p = 16.**

**(c) Head selection — there is none in the usual sense.**
This is the important detail and it is easy to get wrong. PCHI does **not** rank
heads and take a top-k. `H` is simply **all attention heads in a contiguous band of
layers strictly after `ℓ_p`**:

* Qwen3-4B: intervention layers **19–22** → 4 layers × 32 heads = **128 coefficients**.
* Gemma3-4B: intervention layers **17–20** → 4 layers × 8 heads = **32 coefficients**.

Sparsity is achieved *post hoc* by an L1 identity regulariser that keeps most `g_h`
near 1; the paper explicitly disclaims this as a head-selection or head-importance
claim.

### 1.3 The intervention

Let `z_{i,t,h}` be the **pre-output-projection** attention output of head `h = (ℓ, a)`
at template coordinate `t` (i.e. the per-head slice *before* `W_O`, shape `d_head`).
PCHI **multiplicatively rescales** it — there is no additive term anywhere:

```
z'_{i,t,h} = ( 1 + (g_h − 1) · s_{i,t} ) · z_{i,t,h}          (Eq. 3)
```

* `g_h ∈ R` is a learned scalar per head, initialised to 1 (identity).
* `s_{i,t} ∈ [0,1]` is the probe-conditioned gate strength.
* `s = 0` ⇒ exact identity; `s = 1` ⇒ full scaling by `g_h`; in between it
  interpolates. `g_h` may be < 1 (suppress) or > 1 (amplify).

**Gating rule.** Soft during training, hard at inference (Eq. 4):

```
training:   s_{i,t} = p^{(ℓ_p)}_{i,t}                      (differentiable)
inference:  s_{i,t} = p^{(ℓ_p)}_{i,t}   if p ≥ τ
                    = 0                 otherwise ,   τ = 0.5
```

So the *decision* to intervene is binary but the *magnitude* above threshold stays
proportional to the probe probability. The inference rule needs no correctness label.

**Optional post-probe attention mask.** The pre-template context is partitioned into
prompt `P`, reasoning `R`, answer `A`. A visible-context choice
`C ∈ {P, R, A, R∪A, full}` restricts what template queries may attend to — but **only
in layers ℓ > ℓ_p**, so the probe input distribution is untouched.

### 1.4 Training objective (Eq. 5)

Base model and probes frozen; only the `g_h` are trained. Training examples are WY
and CY only. `Δ_i` = original gap, `Δ'_i` = post-intervention gap.

```
L    = L_WY + L_CY + λ · Σ_{h∈H} |g_h − 1|
L_WY = E_{i∈WY} [ max(0, Δ'_i − m) ]
L_CY = E_{i∈CY} [ max(0, ρ·Δ_i − Δ'_i) ]
```

* `L_WY` pushes wrong-confident gaps below a target margin `m`.
* `L_CY` is the *collateral-protection* term: the post-intervention gap must stay at
  at least a fraction `ρ` of its original value. This is PCHI's analogue of our
  "don't damage confident-right" constraint.
* `λ·Σ|g_h − 1|` is the identity regulariser.

Training runs on **fixed replays of the confidence-template span**, so `Δ_i` and
`Δ'_i` are compared at the identical confidence-value prediction point (teacher-forced
template tokens; the generated answer is never resampled).

### 1.5 Complete hyperparameter table

| Hyperparameter | Symbol | Qwen3-4B | Gemma3-4B |
|---|---|---|---|
| Probe layer | `ℓ_p` | 18 | 16 |
| Intervention layers | — | 19–22 | 17–20 |
| Number of learned coefficients | `|H|` | 128 | 32 |
| Probe family | — | calibrated diagonal-LDA | same |
| Probe labels | — | WY = 1, CY = 0 | same |
| Confidence-template length | `|T|` | 6 tokens | 6 tokens |
| Inference gate threshold | `τ` | 0.5 | 0.5 |
| Batch size | — | 8 | 8 |
| Learning rate | — | 0.04 | 0.04 |
| Steps | — | 200 | 200 |
| L1 weight | `λ` | 0.05 | 0.05 |
| Hinge ratio | `ρ` | 0.7 | 0.7 |
| Random seed | — | 42 | 42 |
| `g_h` init | — | 1.0 | 1.0 |
| Decoding | — | greedy, T = 0 | same |
| ECE bins | `M` | 10, equal-width | 10 |

**Not stated in the paper** (we must choose and declare): the WY hinge margin `m`;
the optimizer (Adam is the obvious default at lr 0.04 for 128 scalars); whether `g_h`
is clamped; the exact token strings in `V_yes` / `V_no`; and the calibration method
used on the diagonal-LDA scores.

### 1.6 Evaluation protocol

* **Models.** Qwen3-4B-Instruct, Gemma3-4b-it. Nothing larger.
* **Dataset.** OpenMathInstruct (Toshniwal et al. 2024) only. 5,000 train problems
  and 5,000 validation problems after filtering (preprocessing from Nam et al. 2025).
  Single-dataset, single-domain (math QA).
* **Baseline group composition (validation).**
  Qwen3-4B: CY 3808 / WY 1067 / CN 28 / WN 97; accuracy 76.7%; yes-rate 97.5%.
  Gemma3-4B: CY 3527 / WY 1298 / CN 19 / WN 156; accuracy 70.9%; yes-rate 96.5%.
* **Metrics.** ECE (10 equal-width bins over the yes/no-restricted probability
  `q_i = softmax over {LSE_yes, LSE_no}`); AUROC with correctness as the positive
  label and `q_i` as the score; **WY Corr.** = fraction of WY whose `Δ' ≤ 0`;
  **CY Dmg.** = fraction of CY whose `Δ' ≤ 0`.

Note the yes-rate: 96–97% of responses say yes, so CY+WY is essentially the whole
evaluation set and the AUROC reported is close to a within-targeted-subset AUROC.

### 1.7 Headline numbers

**Table 2 — readout-token (token 6) intervention.** All percentages.

| Model | Method | ECE ↓ | AUROC ↑ | WY Corr. ↑ | CY Dmg. ↓ |
|---|---|---|---|---|---|
| Qwen3-4B | No intervention | 21.9 | 66.5 | — | — |
| Qwen3-4B | **Activation steering (additive/CAA-style)** | **21.9** | **66.7** | **0.1** | **0.0** |
| Qwen3-4B | PCHI | 11.7 | 90.3 | 82.2 | 10.8 |
| Gemma3-4B | No intervention | 26.3 | 76.8 | — | — |
| Gemma3-4B | **Activation steering (additive/CAA-style)** | **26.2** | **76.8** | **0.5** | **0.0** |
| Gemma3-4B | PCHI | 16.9 | 81.7 | 52.5 | 11.6 |

**Table 3 — upstream confidence-template tokens, Qwen3-4B.**

| Position | ECE ↓ | AUROC ↑ | WY Corr. ↑ | CY Dmg. ↓ |
|---|---|---|---|---|
| No intervention | 21.9 | 66.5 | — | — |
| Token 1 | 21.9 | 66.7 | 0.1 | 0.0 |
| Token 2 | 21.9 | 69.6 | 0.2 | 0.0 |
| Token 3 | 22.7 | 38.2 | 0.0 | 0.0 |
| Token 4 | 15.1 | 88.1 | 35.3 | 3.8 |
| Token 5 | 12.5 | 86.9 | 80.1 | 10.3 |
| **Joint 1–5** | **9.2** | **91.1** | **63.3** | **5.1** |

Gemma3-4B upstream is much weaker: tokens 1–4 correct <1% of WY, token 5 corrects
4.5%, and joint 1–5 gets ECE 26.3 → 23.6 and AUROC 76.8 → 82.1 with only 9.5% WY
corrected. With a Reasoning+Answer mask, Gemma joint 1–5 reaches ECE 18.7 /
AUROC 80.1 / WY Corr. 30.4 / CY Dmg. 4.7.

**Table 6 — mask-only ablation, Qwen3-4B.**

| Setting | ECE ↓ | AUROC ↑ | WY Corr. ↑ | CY Dmg. ↓ |
|---|---|---|---|---|
| No intervention | 21.9 | 66.5 | — | — |
| Mask only, Prompt | 22.6 | 62.0 | 1.0 | 0.1 |
| PCHI, Token 1 + Prompt | 14.6 | 86.2 | 70.2 | 11.1 |
| PCHI, Token 2 + Prompt | 13.2 | 88.3 | 80.6 | 11.0 |
| PCHI, Token 3 + Prompt | 22.6 | 59.5 | 0.9 | 0.1 |

### 1.8 What PCHI reports for additive / CAA baselines — and why this is our opening

The paper's only additive baseline is described as follows:

> "The activation-steering baseline uses the same probe layer as PCHI, but replaces
> learned head coefficients with a single hidden-state shift. For the selected
> confidence-template token, we compute a normalized mean-difference direction on
> the training set from wrong-yes activations to wrong-no activations at that layer,
> and add a scaled version of this direction to the current-token hidden state during
> inference. We sweep the scaling coefficient over [1,8] and report the best result."

That is a CAA / ActAdd / diff-in-means construction, but note three things we should
say out loud in the rebuttal:

1. It is **ungated** — no probe conditioning, applied unconditionally, so it is not a
   like-for-like control for PCHI's gate. PCHI therefore has **no gated additive
   baseline at all**; the gating and the multiplicative head parameterisation are
   confounded in their comparison.
2. The direction is **WY → WN**, not WY → CY or wrong → right. It is a
   "confidence-lowering" direction, not a correctness direction.
3. The reported result is essentially *null* (WY Corr. 0.1% / 0.5%, ECE unchanged),
   which is a suspiciously weak additive baseline given that a sweep to coefficient 8
   on a normalised direction should move the logits substantially. Their own framing
   — "the improvement is not explained by a generic mean-difference shift at the same
   layer" — is doing a lot of work. A stronger additive baseline (multiple layers,
   larger coefficients, or projection-based) is a fair criticism to raise, and our
   gated-additive ablation is exactly the missing cell in their design.

**No gated baseline other than PCHI itself is reported.** The mask-only ablation
(Table 6) is the closest thing to an intervention-free control.

### 1.9 Stated limitations (useful for the rebuttal)

* Single task family (math QA), single structured JSON format, binary yes/no
  confidence. No free-form or multi-level confidence, no open-ended tasks.
* Probe layers, intervention layers, template positions and masks are all selected
  per model from training-set diagnostics; the authors say stability across model
  families is untested.
* Two models, both 4B. No scale study.
* CY damage is never below ~5% in any configuration that corrects a meaningful
  fraction of WY (best trade-off: 63.3% WY corrected at 5.1% CY damage).

---

## 2. Wired for Overconfidence (COLM 2026)

**Exact title.** Wired for Overconfidence: A Mechanistic Perspective on Inflated
Verbalized Confidence in LLMs
**Authors.** Tianyi Zhao, Yinhan He, Wendy Zheng, Yujie Zhang, Chen Chen (University of Virginia)
**Venue.** COLM 2026 (arXiv comments field reads exactly `COLM 2026`). cs.CL.
**arXiv.** 2604.01457 — v1 2026-04-01, revised 2026-07-27.
**URL.** https://arxiv.org/abs/2604.01457
**Code.** **No release found.** No GitHub/anonymous link anywhere in the paper.

This is an analysis paper with an intervention appendix, not a steering-method paper.
It is the mechanistic justification for "confidence lives in mid-to-late components"
that PCHI and we both lean on.

### 2.1 Setup

* **Two-step elicitation.** Step 1: model answers the question. Step 2: model reports
  an integer confidence 0–99 for its own answer. Confidence verbalization is thereby
  isolated as a separate forward pass.
* **Truth-injection counterfactual.** For each incorrect record, the corrupted prompt
  replaces the Step-1 assistant turn with the ground-truth answer, all other tokens
  identical. Gives a clean/corrupt pair isolating the causal effect of correctness.
  Ground truth (not a random token) is used deliberately to stay on-manifold.

### 2.2 The differentiable confidence signal: TSLD

Verbalized confidence has no single golden token, so they define **Target-Set Logit
Difference** at the final prompt position `pos_end`:

```
L_TSLD(x) = (1/|H|) Σ_{c∈H} logits[-1, c_1]  −  (1/|L|) Σ_{c∈L} logits[-1, c_1]
H = {70,75,80,85,90,99}       L = {0,10,15,20,25,30}
```

`c_1` is the first token of the integer `c`. `Δ_TSLD = L_TSLD(x_corrupt) − L_TSLD(x_clean)`.
Validation: `Δ_TSLD` correlates with the change in raw verbalized confidence at
r = 0.815 (Qwen2.5-3B) and r = 0.734 (Llama-3.2-3B).

Note the design rationale, which is directly relevant to our "don't collapse
everything" argument: they track a *difference* rather than `Mean(H)` alone in order
to avoid an "inhibition trap", where an intervention lowers high-confidence logits
simply by breaking generation. This is the same failure mode our CY-damage /
confident-right metric is designed to catch.

### 2.3 Circuit discovery

**EAP-IG** (edge attribution patching with integrated gradients), m = 5 integration
steps:

```
EAP-IG(e_{a→b}) = (1/m) Σ_{k=0}^{m-1} ∂L_TSLD/∂input_b |_{x_k} · ( out_a(x_clean) − out_a(x_corrupt) )
```

Edge scores are summed onto incident nodes and averaged over the "Bucket 1"
(overconfident) discovery records. The result is the **Confidence Mover Circuit (CMC)**.

Findings: the CMC concentrates in **middle-to-late layers** at the **final token
position**. Qwen2.5-3B is strongly MLP-dominant (MLPs in layers 24–35; a few heads
like L24H5, L27H1). Llama-3.2-3B looks distributed but after single-component
ablation the causally necessary set is M27, M25, M23, M14, L14H3, L13H18 — early MLPs
M0–M4 give ~zero individual TSLD reduction and are relay nodes. ≥80% cross-dataset
overlap of the top-10 in both models; at 7B, 9 of the top 10 are shared across
PopQA/NQOpen.

### 2.4 The interventions (both ungated)

Applied to the **top-10 CMC components** at `pos_end`. A global reference is estimated
once from a small Bucket-1 set and applied **uniformly at inference — no per-sample
counterfactual, and no gate.**

**(a) Mean ablation** — overwrite:
```
h_out(x, pos_end) ← μ_ref = (1/N) Σ_i h_out(x_corrupt^(i), pos_end)
```

**(b) Activation steering** — a plain additive CAA/diff-in-means subtraction:
```
v_conf = (1/N) Σ_i [ h_out(x_clean^(i), pos_end) − h_out(x_corrupt^(i), pos_end) ]
h_out(x, pos_end) ← h_out(x, pos_end) − α · v_conf ,      α ∈ [0,1]
```

### 2.5 Evaluation and headline numbers

* **Models.** Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct (plus Qwen2.5-7B-Instruct in
  Appendix J for scale).
* **Datasets.** PopQA, MMLU, NQOpen.
* **Metrics.** ECE, Brier score, reliability curves. Evaluation is on the **full**
  dataset even though the references come from the small discovery subset.

Results: ECE reduced by ~78–97% on PopQA and ~81–83% on NQOpen; MMLU is weaker
(~33% Qwen, ~57% Llama). Steering shows a clean **dose–response**: improves from
α = 0.3, optimum around **α = 0.5–0.6**, then degrades — stable in 4 of 6
configurations. Mean ablation is brittle (96.9% ECE reduction on Llama/PopQA but only
3.1% on Llama/MMLU). Qwen2.5-7B: ECE 0.647 → 0.068 (PopQA) and 0.626 → 0.109
(NQOpen) at α = 0.6.

Table 3 (PopQA) steering sweep shows the non-monotone shape explicitly:
α = 0.4 → ECE 0.196; α = 0.5 → 0.063; α = 0.6 → 0.020; α = 0.7 → 0.027; α = 0.8 → 0.031.

**Post-hoc baselines (Appendix I).** Temperature, Platt, isotonic, histogram binning,
each fit on a labeled 50% split and evaluated on the disjoint half. On
Llama-3.2-3B/PopQA: raw ECE 0.568, temperature 0.302, Platt 0.008, isotonic 0.075,
histogram 0.147, mean ablation 0.018, best steering 0.020. On Qwen2.5-3B/NQOpen: raw
0.555, temperature 0.304, Platt 0.016, isotonic 0.054, histogram 0.033, mean ablation
0.374, best steering 0.103. They explicitly disclaim any SOTA-calibration claim.

### 2.6 Relevance to us

* **No gated baseline at all.** Both interventions are unconditional and applied to
  every example. The paper's own discussion concedes that the optimum sits at
  *moderate* strength because stronger intervention "begins to disrupt other useful
  confidence-related computation" and because the averaged `v_conf` "only approximates
  the true per-example overconfidence axis". That is precisely the argument for a gate
  and for a per-example transport map, i.e. for our method. We should cite this
  passage.
* They note that the verbal-confidence circuit is **correctness-insensitive** and
  structurally misaligned with the factual-retrieval circuit — which supports the claim
  that a single global direction cannot separate confident-right from confident-wrong.

---

## 3. COAST — Minimizing Collateral Damage in Activation Steering

**Exact title.** Minimizing Collateral Damage in Activation Steering
**Authors.** Tam Nguyen, Tu Anh Nguyen, Sina Alemohammad, Richard G. Baraniuk (Rice)
**Venue.** arXiv preprint, cs.LG / cs.AI, 1 May 2026. ICML-style format; no venue stated.
**URL.** https://arxiv.org/abs/2605.01167
**Code.** **No release found.** The only GitHub link in the paper is the Stanford
Alpaca dataset reference.

### 3.1 The formulation

Work on the unit sphere (norm restored afterwards). `h` = current activation,
`d` = unit target direction, `α ∈ [-1,1]` = desired post-steering cosine alignment.

Collateral damage of a move `Δ = x − h`, over a population of non-target feature
directions `f`:

```
E[(fᵀ(x − h))²] = (x − h)ᵀ Σ_f (x − h) ,      Σ_f := E[f fᵀ]
```

Feasible set (norm preservation + alignment budget):

```
M := { x ∈ R^p : ‖x‖ = 1,  dᵀx = α }
```

which is a `(p−2)`-sphere centred at `αd` with radius `r = √(1−α²)`. The steering
problem is

```
min_{x ∈ M} (x − h)ᵀ Σ (x − h)                                  (Eq. 3)
```

### 3.2 Choice of Σ

Three options, in increasing order of what they actually use:

1. `Σ_f = E[f fᵀ]` — uniform feature importance.
2. `Σ_w = Σ_i w_i f_i f_iᵀ` from an SAE dictionary with `w_i = E_{h∼H_ref}[c_i(h)²]`.
   Rejected: needs a good SAE per model per layer, and ignores feature co-activation.
3. **What they use: the empirical second moment of activations**
   `Σ_{h_ref} = E_{h∼H_ref}[ ĥ ĥᵀ ]`, which captures both individual feature moments
   and cross terms `c_i c_j f_i f_jᵀ` without any dictionary.

They draw a distinction worth borrowing for our related work: **uniform importance**
(prescriptive: penalise all non-target directions equally) is *not* the same as
**uniform geometric distribution** (descriptive: features are isotropically spread).
Standard additive steering conflates the two.

### 3.3 The algorithm (Riemannian descent on the budget sphere)

Tangent space and projector:
```
T_x M = { ξ : xᵀξ = 0, dᵀξ = 0 }
Π_x   = I − x xᵀ − (d − αx)(d − αx)ᵀ / (1 − α²)
```
Riemannian gradient: `grad J(x) = Π_x ( 2 Σ (x − h) )`.
Geodesic update via the closed-form exponential map (Lemma 1), with `v = −η·grad J(x)`:
```
exp_x(v) = α d + (x − α d) cos τ + r · (v/‖v‖) sin τ ,   τ = ‖v‖/r,  r = √(1−α²)
```
Because the update is a geodesic *on* `M`, the alignment budget `dᵀx = α` holds
exactly at every iteration. A KKT root-finding solver that gives the exact global
optimum exists (Appendix E) but is discarded as too slow — the approximate solution is
sufficient.

### 3.4 Relation to SLERP / ActAdd

**COAST strictly generalises SLERP.** Under Isotropic Collateral Damage — (i)
`Σ = Π_{d⊥} = I − ddᵀ` (worst-case damage, features orthogonal to `d`), (ii) features
only near-orthogonal, `|dᵀf| ≤ ε`, or (iii) features uniform on the sphere so
`Σ ∝ I` — the optimum has the closed form

```
x* = α d + r · (h − (dᵀh) d) / ‖h − (dᵀh) d‖
```

which is exactly a SLERP from `h` toward `d`, stopped at the alignment budget. SLERP
is described as "the norm-preserving counterpart of ActAdd".

### 3.5 Implementation constants

* **Intervention locations.** `2L` locations — after the layer norm of *both* the
  attention and the MLP sub-layer, in every layer. Applied identically to all methods.
* **Direction.** Per-location diff-in-means on **unit-normalised** activations
  (Belrose 2023), `d_ℓ ∝ μ_harmful − μ_harmless`, renormalised. `D_harmful` = 80% of
  AdvBench (416 harmful instructions), `D_harmless` = 512 random Alpaca instructions.
  A different `d_ℓ` per location (Angular Steering keeps one global direction).
* **Σ estimation.** N = **100,000** token activations from C4 per location,
  unit-normalised, `Σ = (1/N) Σ ĥ ĥᵀ`, then **normalised by its top eigenvalue** so
  `‖Σ‖₂ = 1`. This makes Proposition 1 guarantee descent for any `η ∈ (0, 0.5]`.
* **Optimisation.** `η = 0.3`, **T = 1 iteration** — a single geodesic step is enough.
* **Adaptive alignment budget.** `α_{ℓ,t} = α · |⟨ĥ_{ℓ,t}, d_ℓ⟩|`, so tokens that
  already express the feature get more budget and unrelated tokens are constrained
  tighter. (Note this is COAST's *only* form of conditioning — it is a soft, purely
  geometric gate on the *budget*, not on whether to fire.)
* **Norm handling.** normalise → spherical update → rescale by the original `‖h‖`.

### 3.6 Evaluation and numbers

* **Models.** Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct,
  Gemma-2-9B-it.
* **Task.** Refusal-direction jailbreaking. ASR on HarmBench; capability preservation
  on tinyBenchmarks (100 examples each of tinyARC, tinyHellaSwag, tinyMMLU,
  tinyTruthfulQA, tinyWinogrande, tinyGSM8k); plus perplexity.
* **Baselines.** ActAdd, Angular Steering, SLERP, No Steering.
* **Strength matching.** All methods put on a common angular scale: sweep
  `θ ∈ [0°,180°]`, `α = cos θ`. **ActAdd coefficient swept −10..10 in steps of 1.**

Headline: up to **+30% ASR over Angular Steering** and **+20% accuracy over ActAdd at
matched ASR**. ActAdd shows a steep ASR/accuracy trade-off; Angular Steering cannot
reach high ASR.

COAST vs SLERP (Table 1, averaged over `θ ∈ [0°,180°]`), ASR / PPL / Avg Acc:

| Model | SLERP | COAST |
|---|---|---|
| Llama-3.1-8B-Instruct | 51.67 / 6.60 / 65.95 | 52.13 / 6.53 / 65.70 |
| Llama-3.2-3B-Instruct | 52.68 / 5.67 / 60.10 | 54.76 / 5.56 / 60.43 |
| Qwen2.5-14B-Instruct | 7.64 / 2.81 / 71.41 | 6.93 / 2.80 / 71.39 |
| Gemma-2-9B-it | 19.28 / 2.15 / 71.88 | 19.48 / 2.15 / 72.17 |

So against its nearest neighbour SLERP the margins are small (and COAST loses on
Qwen2.5-14B, which the authors attribute to optimisation dynamics with an aggressive
step size in near-isotropic regions). The big claimed wins are all against ActAdd and
Angular Steering.

Throughput cost: 0.9–4.0% slower than unsteered, <1.5% slower than SLERP.

They also validate collateral damage `(x−h)ᵀΣ(x−h)` as a proxy for downstream
degradation: strong negative Pearson correlation with tinyBenchmarks accuracy on
Qwen2.5-14B.

### 3.7 Relevance to us

* COAST is the natural "optimal-transport-flavoured" competitor: it is a
  **norm-preserving, metric-aware, projection-local** move, and `Π_x` is literally a
  projection onto the constraint tangent space. The differences we should argue are
  (i) COAST is **completely ungated** — every token at every one of `2L` locations is
  moved; (ii) its notion of "what to preserve" is a *global* second moment from C4,
  not a per-example or per-population transport plan; (iii) it has no notion of
  correct-vs-wrong, so it cannot in principle spare confident-right examples.
* Their `α_{ℓ,t} = α·|⟨ĥ,d⟩|` adaptive budget is the closest thing in the literature
  to our gate and should be cited as such, and ideally run as an ablation.
* **ActAdd/CAA baseline result:** ActAdd is the worst trade-off curve in every figure
  (~20% accuracy behind COAST at matched ASR), swept over coefficients −10..10.

---

## 4. A-LQR — Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control

**Exact title.** Local Linearity of LLMs Enables Activation Steering via Model-Based
Linear Optimal Control
**Authors.** Julian Skifstad, Xinyue Annie Yang, Glen Chou
**Venue.** arXiv preprint, cs.LG / cs.AI / eess.SY / math.OC / stat.ML, 21 April 2026.
**URL.** https://arxiv.org/abs/2604.19018
**Code.** **YES** — https://github.com/trustworthyrobotics/lqr-activation-steering
(the only one of the four with a release). Look for the LQR gain solve (Riccati
recursion producing `{K_k}`), the Jacobian extraction `A_k = ∂φ_k/∂z`, and the
per-layer hook implementing `u_k = (β*_k − v_kᵀ z_k) K_k v_k`.

### 4.1 Method

**LFS (Linear Feature Setpoint)** — the setpoint generator. Standard contrastive
diff-in-means per layer `k` over `D_+` (representative) and `D_−` (contrastive):

```
e_k = z_{k,+} − z_{k,−} ,   μ_k = ‖e_k‖₂ ,   v_k = e_k / μ_k
β_k = v_kᵀ z_k                        (feature strength)
α_k = β*_k − v_kᵀ z_k                 (feature-strength error)
z'_k = z_k + α_k v_k                  (unique target state, Thm 4.1)
```

**A-LQR** — the controller. Linearise each transformer block around
`z̄_k := z_{k,+}`, `ū_k := 0`:

```
δz_{k+1} ≈ A_k δz_k + B_k δu_k ,   A_k := ∂φ_k/∂z |_(z_{k,+}, 0) ,   B_k := I
```

Solve the LQR problem via Riccati recursion for gains `K_k ∈ R^{d×d}`; the control law is

```
u*_k = ū_k − K_k δz_k = (β*_k − v_kᵀ z_k) · K_k v_k              (Eq. 19)
```

This is genuinely **closed-loop**: the intervention magnitude is the *online-measured*
feature error, so it self-attenuates when the activation is already on target. That is
an important distinction from a fixed additive vector, and it is the closest thing in
this set of papers to a continuous gate — though it is a *magnitude* gate, not a
fire/don't-fire gate, and it has no notion of correctness.

**Efficiency.** The local-linearity result (Sec. 5: `A_k` is stable across reachable
activations for fixed `k`) lets them compute `{K_k}` **offline once** and reuse them;
the controller stays closed-loop because it still reads `z_k` online. Gain computation
is `O(ℓ d³)` on CPU. No offline training of any kind.

**Hyperparameters.** The LQR cost matrices `{Q_t}`, `{R_t}` are the tunable knobs
(state-deviation vs control-effort penalties), tuned empirically; values in their
Appendix E. `λ` is the user-facing target feature-strength scale. **Intervention is on
the last token only** by default; **A-LQR+** applies the same `K_k` at all token
positions with per-position error signals (used only for jailbreaking, since it is
more invasive and raises PPL).

### 4.2 Evaluation

* **Models (8).** Llama-3.2-1B, Gemma-2-2B, Qwen-2.5-3B, Llama-3-8B, Gemma-2-9B,
  Qwen-2.5-14B, Qwen-2.5-32B, Llama-3.1-70B (last two partial).
* **Baselines.** ITI, **ActAdd**, Mean-AcT, Linear-AcT, PID-AcT, ODESteer, and their
  own **S-PID** (a PID controller tracking the same LFS setpoint — this is the clean
  ablation isolating "optimal control" from "closed-loop feedback").
* **Tasks / metrics.**
  * *Arbitrary concepts*: OneSeC sentences for `D_+`; 500 generations from "Once upon
    a time,"; concept prevalence judged by Llama-3.1-8B-as-judge.
  * *Toxicity*: 1000 RealToxicityPrompts samples, RoBERTa toxicity classifier, Dist-1/2/3
    for diversity, Mistral-7B PPL, 5-shot MMLU as a blind over-tuning check. Zero-shot
    transfer tested on Jigsaw.
  * *Truthfulness*: TruthfulQA generation split, Truthfulness × Informativeness with the
    two finetuned Llama-2-7B judges.
  * *Jailbreaking*: 80/20 AdvBench split (104 eval prompts), HarmBench classifier ASR +
    substring refusal score, greedy decoding, baseline Adaptive Angular Steering (AAS).

### 4.3 Headline numbers

* **Toxicity:** A-LQR gives ~**30×–50×** reduction in toxic outputs vs the base model,
  where baselines typically manage ~8–10×, while holding Dist and MMLU and adding only
  small PPL. Example, Gemma-2-2B (toxicity % / Dist / MMLU / PPL):
  Original 5.14 / 0.67 / 66.04 / 6.74; ITI 0.96 / 0.67 / 50.12 / 10.24;
  **ActAdd 1.10 / 0.64 / 35.78 / 11.42** (note the MMLU collapse — this is the
  collateral-damage point); Mean-AcT 0.50 / 0.68 / 54.62 / 8.70;
  Linear-AcT 0.92 / 0.69 / 54.42 / 8.77; PID-AcT 0.86 / 0.69 / 54.00 / 8.20;
  ODESteer 0.58 / 0.64 / 52.52 / 11.62; S-PID 0.80 / 0.70 / 53.20 / 11.68;
  **A-LQR 0.18 / 0.68 / 53.56 / 12.26**.
* **Truthfulness:** Llama-3-8B, A-LQR raises T×I ~17% over base while keeping >96%
  informativeness; ActAdd gets a competitive ~13% but at a nontrivial informativeness
  cost.
* **Jailbreaking:** plain A-LQR *underperforms* AAS on ASR (they diagnose "benign
  non-refusal": the model neither refuses nor answers). A-LQR+ (all token positions)
  matches or beats the baseline.

### 4.4 Relevance to us

The honest framing is that A-LQR is a *control-theoretic* competitor rather than a
calibration one: it never touches confidence and never conditions on correctness. Its
closed-loop error term is the feature we should compare our gate against
conceptually — A-LQR self-attenuates when the feature is already at target, which is a
weaker form of the same "don't move what's already fine" instinct. S-PID is their own
gated-ish ablation and is the number to quote if a reviewer asks "does the
sophistication buy anything over simple feedback": S-PID is competitive on the primary
metric but spikes PPL, especially at small scale.

---

## 5. Cross-cutting summary for the rebuttal

| | PCHI | Wired (COLM) | COAST | A-LQR |
|---|---|---|---|---|
| arXiv | 2606.09876 | 2604.01457 | 2605.01167 | 2604.19018 |
| Code | no | no | no | **yes** |
| Target behaviour | verbalized confidence | verbalized confidence | refusal/jailbreak | toxicity, truthfulness, refusal, concepts |
| Where it acts | per-head, pre-`W_O`, 4-layer band | top-10 MLPs/heads at `pos_end` | residual, `2L` locations | residual, per layer |
| Op | multiplicative per-head gain | additive (or mean-ablation) | geodesic on budget sphere | closed-loop linear feedback |
| Gated? | **yes** (frozen probe, τ = 0.5) | no | no (only an adaptive *budget*) | magnitude self-attenuates; not a gate |
| Preserves correct-confident? | explicitly, via the `ρ` hinge | not measured | only via global Σ | not measured |
| Reports a CAA/additive baseline? | yes, ungated, essentially null | it *is* the additive method | yes, ActAdd, swept −10..10 | yes, ActAdd |
| Reports a *gated* baseline? | **no** | no | no | S-PID (feedback, not gating) |

**The gap we occupy.** Nobody in this set reports a gated-additive control. PCHI is
the only gated method, and it confounds the gate with a multiplicative per-head
parameterisation; COAST is the only geometry-aware/transport-flavoured method, and it
is unconditional; Wired is the only one with a proper mechanistic justification, and
its own discussion section argues that a single averaged direction under-approximates
the per-example axis. A method that is *both* gated *and* projection-local optimal
transport has no direct precedent here, and the missing cells in their ablation tables
are exactly the ones we fill.

**Datasets/metrics we can reuse for head-to-head comparison.**
PCHI: OpenMathInstruct with the JSON `is_confident` schema; WY Corr. / CY Dmg. / ECE(10 bins) / AUROC.
Wired: PopQA, MMLU, NQOpen with two-step 0–99 confidence; ECE, Brier, reliability curves.
COAST: HarmBench ASR × tinyBenchmarks accuracy trade-off curves.
A-LQR: RealToxicityPrompts, TruthfulQA gen split, AdvBench.

---

## 6. REIMPLEMENTATION SPEC — PCHI in ~80 lines of PyTorch

Target: a HuggingFace decoder-only model with (a) residual-stream hooks per layer and
(b) per-head hooks on the attention output *before* `o_proj`.

### 6.1 Hook point

The intervention is on `z ∈ R^{n_heads × d_head}`, the concatenated per-head attention
output **immediately before** `o_proj`. In a Llama/Qwen-style `LlamaAttention`, that is
the tensor passed to `self.o_proj(...)`. The cleanest implementation is a
`forward_pre_hook` on `layer.self_attn.o_proj`:

```python
def make_hook(layer_idx, gains, state):          # gains: nn.Parameter [n_heads]
    def pre_hook(module, args):
        z = args[0]                              # [B, T, n_heads * d_head]
        s = state["s"]                           # [B, T] gate, 0 outside template coords
        if s is None: return None
        z = z.view(*z.shape[:2], n_heads, d_head)
        scale = 1.0 + (gains.view(1, 1, -1) - 1.0) * s.unsqueeze(-1).unsqueeze(-1)
        return (( z * scale ).flatten(-2),) + args[1:]
    return pre_hook
```

`state["s"]` is written by the probe hook (below) and is nonzero **only** at the
template coordinates in `S`, and only for layers in the intervention band.

### 6.2 Pseudo-code

```
# ---------- STAGE 0: data collection (offline, once) ----------
prompts     = sample(OpenMathInstruct.train, n=5000)   # after Nam et al. 2025 filtering
val_prompts = sample(OpenMathInstruct.valid, n=5000)
for x in prompts:
    y = model.generate(JSON_PROMPT(x), temperature=0)          # greedy
    reasoning, answer, is_conf = parse_json(y)
    c = evaluator(answer, gold(x))                             # correctness bit
    group = {(1,'yes'):'CY', (0,'yes'):'WY', (1,'no'):'CN', (0,'no'):'WN'}[(c, is_conf)]
    T = locate_confidence_template(y)      # 6 token indices; T[5] is the readout token
    H = forward_and_cache_hidden_states(y, positions=T)        # [6, n_layers, d]
    store(group, T, H, answer_span, reasoning_span, prompt_span)

# ---------- STAGE 1: probes (offline, once) ----------
for t in range(6):
  for L in range(n_layers):
      X = stack([H[t, L] for i in WY + CY]);  y = [1]*|WY| + [0]*|CY|
      probe[t][L] = calibrated_diagonal_LDA(X, y)       # diagonal covariance; then calibrate
      auroc[t][L] = auroc_on_train(probe[t][L], X, y)
L_p = choose_layer(auroc)      # high separability, >= 4 layers left downstream
                               # paper: Qwen3-4B -> 18 ; Gemma3-4B -> 16
INTERVENTION_LAYERS = [L_p+1, L_p+2, L_p+3, L_p+4]
H_set = [(L, a) for L in INTERVENTION_LAYERS for a in range(n_heads)]   # ALL heads, no top-k

# ---------- STAGE 2: learn the head gains (offline, once) ----------
g = nn.Parameter(torch.ones(len(INTERVENTION_LAYERS), n_heads))
opt = Adam([g], lr=0.04)                                   # optimizer not stated; Adam assumed
for step in range(200):
    batch = sample(WY + CY, 8)
    # teacher-forced replay of the template span; answer is NEVER resampled
    with probe_hook(L_p, positions=S, mode="soft"), gain_hooks(INTERVENTION_LAYERS, g):
        logits = model(replay_ids(batch)).logits
    D_prime = LSE(logits[readout_pos, V_yes]) - LSE(logits[readout_pos, V_no])
    L_WY = relu(D_prime[is_WY] - m).mean()                 # m: NOT stated in the paper
    L_CY = relu(0.7 * D_cached[is_CY] - D_prime[is_CY]).mean()
    loss = L_WY + L_CY + 0.05 * (g - 1).abs().sum()
    opt.zero_grad(); loss.backward(); opt.step()

# ---------- STAGE 3: inference ----------
# forward the full response; at each template coordinate t in S:
p   = probe[t][L_p](h[t, L_p])
s_t = p if p >= 0.5 else 0.0                # hard gate, soft magnitude
# then for every head in INTERVENTION_LAYERS at coordinate t:
z  <-  (1 + (g[L, a] - 1) * s_t) * z
# optional: for layers L > L_p only, mask template queries to the visible context C
```

Notes that matter for fidelity:

* The gate `s` is **per template coordinate**, and layers `≤ L_p` are never touched —
  otherwise the probe reads off-distribution states and the whole thing is invalid.
* Soft gate (`s = p`) during training, hard gate (`s = p·1[p ≥ τ]`) at inference. Do
  not use the hard gate during training; the objective stops being differentiable in
  the region that matters.
* `Δ` (the cached, pre-intervention gap) must be measured at the *same* prediction
  point as `Δ'`, which is why the template is replayed rather than regenerated.
* The optional attention mask applies only to template queries and only in layers
  `> L_p`.

### 6.3 Hyperparameters, with the paper's values

| Name | Symbol | Paper value | Notes |
|---|---|---|---|
| Probe layer | `ℓ_p` | Qwen3-4B 18, Gemma3-4B 16 | from train-set AUROC heatmap |
| Intervention band | — | `ℓ_p+1 … ℓ_p+4` (Qwen 19–22, Gemma 17–20) | contiguous, all heads |
| Number of gains | `|H|` | 128 (Qwen), 32 (Gemma) | = 4 × n_heads |
| Gain init | `g_h` | 1.0 | identity |
| Gate threshold | `τ` | 0.5 | inference only |
| L1 identity weight | `λ` | 0.05 | |
| CY hinge ratio | `ρ` | 0.7 | protects correct-confident |
| WY target margin | `m` | **not stated** | we must pick; try `m = 0` and `m = −1` |
| Learning rate | — | 0.04 | |
| Batch size | — | 8 | |
| Steps | — | 200 | |
| Seed | — | 42 | |
| Optimizer | — | **not stated** | Adam is the natural default |
| Probe family | — | calibrated diagonal-LDA | not logistic regression |
| Probe labels | — | WY vs CY only | CN/WN discarded |
| Template length | — | 6 tokens | token 6 = readout |
| Intervention coords | `S` | `{6}` or `{1..5}` (joint) | joint 1–5 is their best Qwen result |
| Attention mask | `C` | `{prompt, reasoning, answer, reasoning+answer, full}` | optional, layers > `ℓ_p` |
| Decoding | — | greedy, temperature 0 | |
| Train/eval size | — | 5,000 / 5,000 | OpenMathInstruct |
| ECE bins | `M` | 10, equal width | |

### 6.4 Sanity checks before trusting our port

1. Reproduce the group composition: Qwen3-4B should give roughly CY 3808 / WY 1067 /
   CN 28 / WN 97 at 76.7% accuracy and a 97.5% yes-rate on the 5,000 validation
   problems. If our yes-rate is far from 97%, the JSON prompt or the parser differs.
2. Reproduce the probe AUROC shape: near chance in early layers, rising through the
   middle, saturating; token 3 (`_conf`) visibly weaker than tokens 2 and 4 on Qwen.
3. Reproduce the null additive baseline: their diff-in-means WY→WN direction at `ℓ_p`,
   coefficient swept over [1,8], should correct ≈0% of WY. If ours corrects a lot, we
   have built a *stronger* additive baseline than theirs — which is a result worth
   reporting, not a bug.
4. Readout-token PCHI on Qwen3-4B should land near ECE 11.7 / AUROC 90.3 / WY 82.2 /
   CY 10.8.
