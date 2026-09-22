# Writing-style reference analysis

Five closely related papers, all accepted at ICLR and all published before 2024,
measured against the current manuscript. Statistics are computed on running
prose only: floats, equations, math mode, citation/reference macros and
bibliographies are stripped before counting (`scratchpad/stats.py`,
`scratchpad/stats2.py`).

Two measurement passes are reported. The first was taken before the manuscript's
punctuation edit; the second (`ours, current`) is the live `paper/paper.tex`.

## 0. How the reference set was found

The set below was assembled with arXiv, Semantic Scholar, Papers With Code and
web search while the alphaXiv MCP server was returning HTTP 429 from its
sign-in server, and re-checked against alphaXiv once it came back. Two
alphaXiv `discover_papers` calls were run, one on the steering/representation-
editing topic restricted to `published_before 2023-12-31` with historical
ranking, and one asking directly for the closest ICLR-accepted antecedents of
a minimum-norm edit along a low-dimensional behavioral direction. The first
returns ActAdd (2308.10248), CAA (2312.06681), Mean-Centring (2312.03813),
LEACE (2306.03819) and activation-patching best practices (2309.16042) —
none of them ICLR-accepted before 2024. The second ranks Task Arithmetic
(2212.04089, ICLR 2023) first, followed by the Othello-GPT follow-up
(2309.00941) and ITI (2306.03341, NeurIPS 2023). That is the same picture the
other tools gave: alphaXiv surfaced no pre-2024 ICLR paper missing from the
set below, and it put one member of the set at the top of the ICLR-specific
query. The set was then checked two further ways.

First, venue: each candidate's arXiv source was downloaded and its ICLR style
file inspected (column below), which is direct evidence independent of any
metadata service.

Second, coverage: a search for pre-2024 ICLR-accepted work on activation
steering itself returns nothing. The two canonical steering papers of that
period, ActAdd (arXiv:2308.10248) and Inference-Time Intervention
(arXiv:2306.03341), are not ICLR-accepted — ITI is NeurIPS 2023 and ActAdd
did not appear in the ICLR proceedings. The nearest pre-2024 ICLR neighbours
are consequently the representation-probing and representation-editing papers
listed below, not steering papers, and the rejected-candidate table at the end
of this section records every close alternative with the venue that
disqualified it.

## 1. Reference set (venue verified)

Verification method: the arXiv e-print source of each paper was downloaded and
inspected. Each one compiles against the official ICLR conference style file
shipped inside the submission tarball, which is direct evidence of the venue and
year, and is independent of any metadata service. OpenReview's API is currently
behind a bot challenge (HTTP 403 `ChallengeRequiredError`) and the Camoufox
fetch backend returned HTTP 500 for the duration of this task, so the style-file
evidence plus the published proceedings record is what the venue column rests on.

| # | Paper | Authors | Venue | Style file found in source | arXiv | OpenReview | Why it is the right comparison |
|---|---|---|---|---|---|---|---|
| 1 | *Discovering Latent Knowledge in Language Models Without Supervision* (CCS) | Burns, Ye, Klein, Steinhardt | ICLR 2023 | `iclr2023_conference_arxiv.sty` | [2212.03827](https://arxiv.org/abs/2212.03827) | https://openreview.net/forum?id=ETKGuby0hcs | Finds a single linear direction in activation space that carries a behavioral property (truth), unsupervised. The closest antecedent for reading out a low-dimensional behavioral subspace. |
| 2 | *Mass-Editing Memory in a Transformer* (MEMIT) | Meng, Sharma, Andonian, Belinkov, Bau | ICLR 2023 | `iclr2023_conference.sty` | [2210.07229](https://arxiv.org/abs/2210.07229) | https://openreview.net/forum?id=MkbcAHIYgyS | Closed-form edit of a representation under a minimum-norm / least-squares objective with a preservation constraint. Structurally the same "move the representation as little as possible subject to a target" problem our transport step solves. |
| 3 | *Editing Models with Task Arithmetic* | Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi | ICLR 2023 | `iclr2023_conference.sty` | [2212.04089](https://arxiv.org/abs/2212.04089) | https://openreview.net/forum?id=6t0Kwf8-jrj | Additive direction arithmetic as a behavior-control interface; the canonical "steer by adding a vector" baseline our projection-transport formulation generalizes. |
| 4 | *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small* (IOI) | Wang, Variengien, Conmy, Shlegeris, Steinhardt | ICLR 2023 | `iclr2023_conference.sty` | [2211.00593](https://arxiv.org/abs/2211.00593) | https://openreview.net/forum?id=NpsVSN6o4ul | Mechanistic analysis of transformer internals with ablation/patching as the causal tool, and an explicit criteria framework (faithfulness, completeness, minimality) for validating an intervention claim. |
| 5 | *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task* (Othello-GPT) | Li, Hopkins, Bau, Viégas, Pfister, Wattenberg | ICLR 2023 | `iclr2023_conference.sty` | [2210.13382](https://arxiv.org/abs/2210.13382) | https://openreview.net/forum?id=DeG07_TcZvT | Probe-then-intervene methodology: establishes that a direction found by a probe is causal by editing activations along it. The template for our "read the axis, then act on it" argument. |

**Verified alternate (sixth):** *Fast Model Editing at Scale* (MEND), Mitchell,
Lin, Bosselut, Finn, Manning — **ICLR 2022**, source compiles against
`iclr2022_conference.sty`, [arXiv 2110.11309](https://arxiv.org/abs/2110.11309),
https://openreview.net/forum?id=0DcZxeWfOPt. It is the best pre-2024 ICLR match
for the *gating* half of our contribution (deciding which inputs are in scope for
an edit) and is a reasonable swap for #4 if the rebuttal needs an editing-heavy
rather than interpretability-heavy comparison set. Its prose statistics are not
in the tables below because it was displaced by IOI.

Candidates considered and rejected on venue grounds (all topically close, none
ICLR pre-2024): INLP / *Null It Out* (ACL 2020), *Linear Adversarial Concept
Erasure* (ICML 2022), LEACE (NeurIPS 2023), ROME (NeurIPS 2022), SERAC /
*Memory-Based Model Editing at Scale* (ICML 2022), *Inference-Time Intervention*
(NeurIPS 2023), *Function Vectors* (ICLR 2024 — out of range), PPLM (ICLR 2020 —
out of range).

## 2. Measured prose statistics

| paper | words | sentences | mean len | median | >45 words | semicolons /1k | em-dashes /1k | colons /1k | parens /1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Meng+23 (MEMIT) | 5351 | 259 | 25.4 | 20 | 7.7% | 3.4 | 0.0 | 9.7 | 18.7 |
| Li+23 (Othello) | 5513 | 274 | 21.7 | 20 | 1.5% | 1.6 | 1.6 | 4.7 | 9.1 |
| Wang+23 (IOI) | 9579 | 508 | 20.6 | 19 | 2.4% | 0.6 | 0.2 | 4.9 | 15.2 |
| Burns+23 (CCS) | 8635 | 412 | 23.0 | 21 | 5.6% | 2.7 | 0.2 | 12.7 | 17.1 |
| Ilharco+23 (task arithmetic) | 8378 | 376 | 23.7 | 22 | 2.9% | 1.1 | 1.9 | 2.5 | 7.5 |
| **reference range** | 5.4k–9.6k | 259–508 | **20.6–25.4** | **19–22** | **1.5–7.7%** | **0.6–3.4** | **0.0–1.9** | **2.5–12.7** | **7.5–18.7** |
| ours (before punctuation edit) | 3887 | 217 | 20.7 | 18 | 5.5% | **25.5** | 1.8 | — | 29.8 |
| **ours (current `paper.tex`)** | 3968 | 230 | 19.8 | 18 | 3.9% | 5.3 | 2.3 | **16.4** | **28.0** |

Reference mean of means: sentence length **22.9 words**, median **20**, long-sentence
tail **4.0%**, semicolons **1.9 /1k**, em-dashes **0.8 /1k**, colons **6.9 /1k**,
parentheses **13.5 /1k**.

### Voice, person and hedging (second pass)

| paper | "we" /1k | "our" /1k | passive constructions /1k | hedge markers /1k |
|---|---:|---:|---:|---:|
| Meng+23 (MEMIT) | 22.4 | 3.4 | 12.0 | 2.8 |
| Li+23 (Othello) | 19.8 | 6.2 | 12.2 | 6.4 |
| Wang+23 (IOI) | 26.1 | 5.0 | 8.4 | 3.7 |
| Burns+23 (CCS) | 31.4 | 4.6 | 4.2 | 7.4 |
| Ilharco+23 (task arithmetic) | 25.7 | 3.6 | 7.5 | 2.0 |
| **reference range** | **19.8–31.4** | **3.4–6.2** | **4.2–12.2** | **2.0–7.4** |
| **ours (current)** | **3.3** | 2.8 | 6.8 | **0.5** |

Hedge markers counted: *may, might, could, suggest(s), appear(s), likely,
possibly, potentially, we believe, seem(s), tend(s) to, largely, partially,
often*. Passive is an approximation (`be`-auxiliary + past participle) and is
useful for comparison across the same detector, not as an absolute rate.

## 3. What the numbers say

1. **Sentence length is already in register.** Mean 19.8 and median 18 sit just
   at or below the short end of the reference band (20.6–25.4 / 19–22), and the
   long-sentence tail (3.9% over 45 words) is comfortably inside 1.5–7.7%. No
   change needed; if anything there is room to let a few sentences run longer.
2. **First-person plural is the biggest divergence in the whole analysis.** At
   **3.3 "we" per 1000 words against a reference floor of 19.8**, the manuscript
   uses roughly **one sixth** of the first-person density of every comparison
   paper. This is the single most recognizable signature of a machine-drafted
   ICLR paper: agentless construction ("the gate is evaluated", "a threshold is
   selected") where the references write "we evaluate the gate", "we select a
   threshold". Fixing this is a mechanical rewrite of roughly 60–80 clauses and
   would do more for perplexity-matching than every other item combined.
3. **Hedging is four times too low.** 0.5 /1k against a 2.0–7.4 band. The
   references routinely write "this suggests that", "may", "we hypothesize",
   "tends to". A draft with near-zero hedging reads as either overclaiming or
   synthetic; reviewers read it as the former.
4. **Semicolons are now fixed.** The earlier 25.5 /1k was a ten-fold outlier
   caused by a global substitution that turned parenthetical dashes into
   semicolons. The current 5.3 /1k is close to the 0.6–3.4 band; trimming another
   dozen gets it inside. Target **at most 3.5 /1k**.
5. **Colons are the new outlier**, at 16.4 /1k against 2.5–12.7. Most are
   run-in paragraph headers and "X: Y" definitional forms. Target **under
   12 /1k**.
6. **Parenthesis density remains high** at 28.0 vs a 7.5–18.7 band. Most
   instances are theorem and table cross-references, which the references also
   carry, but a quarter of them hold clause-length asides that belong in the
   sentence. Target **under 20 /1k**.
7. **Em-dash use is in range** (2.3 vs 0.0–1.9, marginally high) and is the
   natural destination for parenthetical asides moved out of parentheses.
8. **Vocabulary divergence is topical, not stylistic.** The words that separate
   the manuscript from the reference register (*gate, selectivity, transport,
   projection, quantile, dose, axis, mimic*) are the objects the paper is about;
   the references show exactly the same pattern with their own objects
   (*circuit, logit, task vector, heads, probe*). No change needed.

## 4. Per-section analysis

### 4.1 Abstract

| paper | sentences | words |
|---|---:|---:|
| Meng+23 (MEMIT) | 4 | 85 |
| Li+23 (Othello) | 7 | 132 |
| Wang+23 (IOI) | 9 | 181 |
| Burns+23 (CCS) | 7 | 202 |
| Ilharco+23 (task arithmetic) | 9 | 229 |

**Range: 4–9 sentences, 85–229 words; median 7 sentences, 181 words.**

*Opening move.* Every one of the five opens on the **field, not the paper**, in
the present tense, with no citation and no numbers. Four distinct realizations:

- Problem-with-the-status-quo: "Existing techniques for training language models
  can be misaligned with the truth: ..." (CCS)
- Recent-progress-then-however: "Recent work has shown exciting promise in
  updating large language models with new memories ... However, this line of work
  is predominantly limited to updating single associations." (MEMIT)
- Practice statement: "Changing how pre-trained models behave — e.g., improving
  their performance on a downstream task or mitigating biases learned during
  pre-training — is a common practice when developing machine learning systems."
  (task arithmetic)
- Field-goal statement: "Research in mechanistic interpretability seeks to
  explain behaviors of machine learning models in terms of their internal
  components." (IOI)

None opens with "We present" or "In this paper". The pivot to the contribution
arrives at **sentence 2 or 3**, marked by "In this work, we propose", "We
propose", "We develop", "We investigate".

*Quantification.* Numbers appear in the abstract but sparsely — **one to three
concrete figures**, always attached to a scope statement, never a table of
results. Representative: "across 6 models and 10 question-answering datasets, it
outperforms zero-shot accuracy by 4% on average" (CCS); "scale up to thousands of
associations for GPT-J (6B) and GPT-NeoX (20B), exceeding prior work by orders of
magnitude" (MEMIT); "26 attention heads grouped into 7 main classes" (IOI). Three
of five quantify the *scope* of the evaluation (how many models, how many
datasets) before quantifying the *gain*.

*Hedging.* Low but non-zero in the abstract specifically. Hedges attach to the
interpretation, never to the measurement: "Interventional experiments **indicate**
this representation **can** be used to control the output" (Othello); "Our results
provide an **initial step** toward ..." (CCS); "Though these criteria support our
explanation, they also **point to remaining gaps** in our understanding" (IOI).

*Closing move.* Four of five close on significance-plus-limit rather than a
summary: "Our results provide an initial step toward discovering what language
models know, distinct from what they say"; "Overall, our experiments ... show that
task arithmetic is a simple, efficient and effective way of editing models"; "Our
work provides evidence that a mechanistic understanding of large ML models is
feasible, pointing toward opportunities to scale ...". Two abstracts end with a
bare code URL as the final sentence.

### 4.2 Introduction

| paper | words | sentences | bulleted contributions? |
|---|---:|---:|---|
| MEMIT | 357 | 15 | no |
| IOI | 509 | 21 | no |
| Task arithmetic | 595 | 25 | no |
| CCS | 632 | 25 | no |
| Othello | 835 | 41 | no |

**Range: 357–835 words. Zero of the five uses a bulleted contribution list.**
This is the most consistent structural finding in the corpus: `\item` count in
all five introductions is **0**.

Contributions, where enumerated at all, are run **inline into a single sentence**
with parenthesized ordinals:

> "Thus, in summary, our main contributions are that we (i) identify a large
> circuit in GPT-2 small that performs IOI (Figure 2 and Section 3), (ii) provide
> a legible interpretation of all the components of the circuit, based on causal
> interventions (Section 3). We then finally (iii) evaluate our circuit against
> the standards of completeness and minimality through extensive experiments
> (Section 4)." — IOI

> "To sum up, we present four contributions: (1) we provide evidence for an
> emergent world model in a GPT variant ...; (2) we compare the performance of
> linear and non-linear probing approaches ...; (3) ..." — Othello

Two of five (MEMIT, task arithmetic) do not enumerate contributions at all; they
end the introduction on what was measured, or on a code link.

*Motivation pattern.* Uniformly three moves: (a) the capability or practice
exists and matters; (b) the existing approach is limited in a specific, named way
("predominantly limited to updating single associations"; "most previous work
either focuses on simple behaviors in small models or describes complicated
behaviors in larger models with broad strokes"); (c) we do the thing that removes
that specific limitation. The limitation is always **concrete and narrow**, never
"however, challenges remain".

*Typical verbs.* `we use/apply` (66 occurrences corpus-wide), `we find` (33),
`we show` (30), `we evaluate/measure/report` (30), `we propose/introduce` (14),
`we observe` (14), `we hypothesize` (6). Note that **`we use` outnumbers
`we propose` by more than four to one** — these papers describe doing far more
than they describe proposing.

### 4.3 Method / Theory

**Formal environments are almost entirely absent.** Across all five papers:

| environment | total occurrences in all 5 papers |
|---|---:|
| `theorem` | 0 |
| `lemma` | 0 |
| `proposition` | 0 |
| `corollary` | 0 |
| `assumption` | 0 |
| `definition` | 7 (all in IOI) |
| `itemize` | 21 |
| `enumerate` | 10 |

For comparison, our manuscript currently has 4 theorems, 2 propositions, 1
corollary and 1 definition. **This is a genuine structural divergence from the
reference register, and it is the one divergence worth keeping** — the paper's
contribution is a constrained-optimization formulation and the guarantees are the
point. But it should be delivered in the corpus's idiom: the only paper that uses
formal environments (IOI) uses `definition` exclusively, and each definition is
short, named after the object it defines, and immediately followed by a prose
sentence restating it informally.

*How results are introduced when there is no theorem.* MEMIT, whose method is the
most mathematical of the five, derives its closed-form update in running prose
with displayed equations, introducing each symbol in the sentence that first uses
it, and states the solution as "this gives" rather than "Theorem 1 states". The
lesson for us: keep the theorems, but **precede each with the one-sentence prose
statement of what it buys**, and follow it with a sentence in plain language.

*Notation conventions observed.* Lowercase bold or plain italic for vectors
(`v`, `k`), uppercase for matrices (`W`, `K`, `V`), subscripts for layer index
and superscripts for example index, `\mathcal{}` reserved for sets and
distributions, hidden state written `h` with a layer superscript. Symbols are
bound **in the sentence that introduces the equation**, not in a notation table —
none of the five papers has a notation table in the main body.

*Method section naming.* The method section is named either flatly ("Method",
MEMIT) or after the object ("Task Vectors", Ilharco; "Problem Statement and
Framework", Burns; "Discovering the Circuit", Wang). Our
"Method: Projection-Transport Steering" is consistent with the object-named
variant.

### 4.4 Experiments

*Setup description.* Short, front-loaded, and delegated. The setup paragraph
names models, datasets and the metric in two to four sentences, then points to an
appendix for everything else. Appendix cross-references are heavy: 47 in IOI, 22
in task arithmetic, 10 in MEMIT. The main text carries the claim; the appendix
carries the configuration.

*Float references.* **Figures dominate tables by a wide margin.** Counting
`\ref` targets: IOI 87 figure vs 0 table; task arithmetic 32 figure / 9 table;
MEMIT 4 figure / 4 table; Othello and CCS are figure-only in the main body.
Section cross-references are also frequent (31–57 per paper).

The reference style is **integrated, not trailing**: the float number appears
inside the sentence that makes the claim ("shown in Table 2", "as Figure 4
illustrates", 17 occurrences of `shown/see/reported in Table|Figure`), and the
bare-parenthetical form "(Table 3)" at the end of a claim is the minority.
Notably, the construction "Table 3 shows that ..." — with the table as grammatical
subject — occurs **once in the entire corpus**. Prefer "we find X (Table 3)" or
"X is shown in Table 3" over "Table 3 shows X".

*How claims are qualified.* The corpus qualifies interpretation, not measurement.
`suggests that` appears 22 times; `however` 47 times; `moreover` 22; `in contrast`
12; `surprisingly` 6; `significantly` only 8 (and where used, backed by a test).
`as expected` appears **zero** times. `furthermore` appears **twice in five
papers** — it is effectively absent from this register despite being a staple of
machine-generated academic prose.

The characteristic qualifying sentence is a measurement followed by a hedged
interpretation in a separate sentence, not a hedged measurement:

> "Negating a task vector decreases performance on the target task, with little
> change in model behavior on control tasks." (flat measurement)
> ... "This suggests that ..." (separate, hedged)

### 4.5 Discussion / Limitations / Conclusion

| paper | closing section(s) | words | sentences |
|---|---|---:|---:|
| MEMIT | "Discussion and Conclusion" | 201 | 7 |
| Burns | "Discussion" | 378 | 15 |
| Othello | "Conclusion" | 419 | — |
| IOI | "Discussion" | 468 | 24 |
| Ilharco | "Discussion" + "Conclusion" | 663 + 245 | 27 + 10 |

**None of the five papers has a section titled "Limitations".** Limitations are
folded into Discussion, typically as two to four sentences using "do not",
"cannot", "fail", "caveat", "future work" (4 such markers in Burns, 4 in IOI, 2 in
Ilharco, 1 in MEMIT). They are specific and tied to a claim made earlier in the
paper, not generic disclaimers.

*Tone.* Measured and slightly deflationary. The corpus consistently under-sells
in the closing section relative to the abstract: "an initial step toward",
"points to remaining gaps in our understanding", "we do not claim". Three of the
five explicitly name something their method does not do before naming future
work.

*Length.* 200–670 words for the discussion; where a separate conclusion exists it
is short (245 words, 10 sentences) and does not repeat numbers.

Our manuscript's "Limitations and Assumptions" as a standalone section is a
deviation. It is defensible under current ICLR norms (which now encourage an
explicit limitations statement) and I would **keep it**, but it should be written
in the corpus's register: specific, short, each item tied to a numbered claim.

### 4.6 Writer-persona pass, paper by paper and section by section

Each of the five references was re-read from its arXiv source. The notes below
are per paper and per section: what the section does, the tone it does it in,
the design of its evidence, its vocabulary, and the narrative move it makes.
Section 4.7 collects what was carried into the manuscript.

---

#### A. Editing Models with Task Arithmetic (ICLR 2023)

*Abstract.* Eight sentences. Opens on the practice, not on a gap: "Changing
how pre-trained models behave ... is a common practice." Pivots at sentence
two with "In this work, we propose a new paradigm ... centered around task
vectors", defines the object in one sentence, then gives one sentence per
operation in the order the paper will present them, and closes on scope
("several models, modalities and tasks") rather than on a headline number.
Vocabulary is deliberately plain: *steer*, *edit*, *negate*, *add*,
*combine*. No hedge markers at all; the claims are all existence claims the
experiments will demonstrate.

*Introduction.* One paragraph of context with four citation clusters, then
the proposal, then three bolded run-in paragraphs — `Forgetting via
negation.`, `Learning via addition.`, `Task analogies.` — each naming the
operation, pointing at its section ("In Section 3, we negate a task
vector..."), and carrying one number ("maintains 98.9% of the accuracy").
Closes on cost and reuse ("no extra cost at inference time in terms of memory
or compute") and a code link. The narrative shape is *one object, three
operations*, and the introduction is an index to the paper rather than an
argument.

*Method.* Under three pages and almost no notation: a task vector is
$\tau_t=\theta^t_{ft}-\theta_{pre}$, applied as $\theta+\lambda\tau$ with
$\lambda$ "determined using held-out validation sets". The scaling
hyperparameter is disclosed in the method, not hidden in an appendix. Where
the method does not apply, the paper says so in one sentence and defers ("we
could follow Matena & Raffel and merge only the shared weights, but this
exploration is left for future work").

*Experiments.* Each section opens with the claim it tests ("In this section,
we show that negating a task vector is an effective way to reduce its
performance on a target task, without substantially hurting performance
elsewhere"), then motivates why one would want it, then states the control
explicitly ("These interventions should not have a substantial effect ...
Accordingly, we measure accuracy on control tasks"). Baselines include a
**random vector matched in magnitude**, narrated as such ("As an experimental
control, adding a random vector has little impact"). Table captions are full
sentences that state the finding with its number: "Negating task vectors
reduce the accuracy of a pre-trained ViT-L/14 by 45.8 percentage points on the
target tasks, with little loss on the control task." Result paragraphs follow
one template: *As shown in Table N, X is the most effective ... For example,
[number]. In contrast, [baseline] ... while [other baseline] severely ...*

*Discussion.* There is no limitations section; the scope limits are carried
inline as one-sentence deferrals. The closing register is practical rather
than reflective.

---

#### B. Discovering Latent Knowledge Without Supervision, CCS (ICLR 2023)

*Abstract.* Seven sentences. Opens on a failure mode of existing techniques
with two parallel clauses ("if we train models with imitation learning, they
may reproduce errors ...; if we train them to generate text that humans rate
highly, they may output errors ..."), then "We propose circumventing this
issue by ...", "Specifically, we introduce ...", "It works by ...", a result
stated with its scope ("across 6 models and 10 question-answering datasets, it
outperforms zero-shot accuracy by 4% on average"), a second result, and a
close on significance-with-a-limit: "Our results provide an initial step
toward ...".

*Introduction.* A two-paragraph problem build in which every claim carries a
citation and the last sentence of each paragraph generalises the problem
("this is an issue that stems from the misalignment between a training
objective and the truth"; "it likely won't be solved by scaling up models
alone"). Then the proposal, then one paragraph per finding, each ending in a
section pointer. Closes on a proof-of-concept frame.

*Method.* Written as a problem statement first and an algorithm second. The
consistency property the method exploits is stated in words before it is
stated in symbols.

*Experiments.* Subsection titles are assertions: "CCS Is Robust To Misleading
Prompts", "CCS Finds A Task-Agnostic Representation of Truth", "CCS Does Not
Just Recover Model Outputs", "Truth is a Salient Feature". Each opens by
naming the alternative hypothesis it is ruling out ("One possibility is that
CCS can only recover knowledge already contained in a model's outputs"), then
argues by consequence ("First, if CCS were just recovering knowledge in the
model outputs, using the last layer should presumably outperform intermediate
layers. However, ..."). Confounds are disclosed inside the result paragraph,
not in a footnote: the misleading-prefix experiment says plainly that
incorrect labels behaved like correct ones and that "the prefix may instead
actually be reducing accuracy because it is out-of-distribution". Conclusion
verbs are hedged where the inference is indirect — *provides evidence that*,
*suggests that*, *we speculate that* — while the measurements themselves are
flat.

*Discussion.* Section 3.3 is titled "Analyzing CCS" and opens by questioning
the paper's own interpretation: "we have described our motivation as
discovering latent representations of truth ... but in practice CCS just finds
a direction in representation space that attains high accuracy. This raises
the question: in what sense is CCS actually finding 'truth' features?" The
paper's own framing is the thing put on trial.

---

#### C. Mass-Editing Memory in a Transformer, MEMIT (ICLR 2023)

*Abstract.* Five sentences, the shortest in the set. Prior work's promise, its
limitation in one narrow sentence ("predominantly limited to updating single
associations"), the method, the result with scale and models ("thousands of
associations for GPT-J (6B) and GPT-NeoX (20B), exceeding prior work by orders
of magnitude"), the code link. No closing significance sentence.

*Introduction.* Opens on a question — "How many memories can we add to a deep
network by directly editing its weights?" — then motivates by application, then
sharpens the prior-work limitation with a number ("a recent study evaluates on
a maximum of 75"). The figure caption doubles as a summary of the whole
contribution and names where the aggregate metric is defined.

*Related Work.* Bolded run-in topics, each ending by placing this paper inside
the topic: "In this paper, we take on the update problem, asking how the
implicit knowledge encoded within model parameters can be mass-edited."

*Experiments.* Metrics are named, capitalised and abbreviated (Efficacy Score,
Paraphrase Success, Neighborhood Success, Reference Score, Generation
Entropy), and the trade-off is forced into a single aggregate — the Editing
Score, a harmonic mean of three. The unedited model is a row in every table.
The narration explains *where each baseline breaks and why*, including
degenerate wins: "MEND ... curiously, having negligible effect on the model at
n=10,000 (the high specificity score is achieved by leaving the model nearly
unchanged)". A baseline's genuine win is conceded in one sentence with a
mechanism attached: "At small n, ROME achieves better generalization at the
cost of slightly lower specificity ... likely due to that method's hard
equality constraint". Runtime is reported for every method, with the authors'
own implementation criticised in the same breath ("its current implementation
is naive and does not batch ... These computations are actually embarrassingly
parallel").

*Discussion.* Their own method's trade-offs are named by case: "although it
also exhibits a trade-off in editing some relations such as P127 ... and
P641".

---

#### D. Interpretability in the Wild, IOI (ICLR 2023)

*Abstract.* Eight sentences and the closest register to ours. Field, then the
gap in prior work stated as a disjunction ("either focuses on simple behaviors
in small models or describes complicated behaviors in larger models with broad
strokes"), then "In this work, we bridge this gap by ...", the scope as a
count (26 heads, 7 classes), a superlative claim explicitly hedged ("To our
knowledge, the largest end-to-end attempt"), the three evaluation criteria
named, and then the sentence that matters most for us: "Though these criteria
support our explanation, they also point to remaining gaps in our
understanding." It closes on feasibility.

*Introduction.* Standard build, then a bulleted "In particular:" list whose
three items are all *awkward* findings — redundant heads, known structures
used in unexpected ways, and heads writing in the opposite direction of the
answer — introduced as "insights about the challenges of mechanistic
interpretability". Negative results are the contribution, framed as insight.

*Method.* The intervention is named and defined ("a causal intervention that
we call path patching") and the supplementary techniques are listed in one
sentence.

*Experiments and validation.* A subsection is titled as the reader's question:
"Did we miss anything? The Story of the Backup Name Movers Heads". Surprise is
stated plainly ("To our surprise, the circuit still worked (only 5% drop in
logit difference)"), the unexplained mechanism is flagged in one sentence
("We hypothesize ... More work is needed to determine the origin of this
phenomenon"), and Section 4 opens by restating what the previous section did
*not* establish. The three criteria are defined formally, each with a toy
counterexample before application. The adverse measurement is reported with
its number and left open: "However, the third resulted in sets K that had high
incompleteness score: up to 3.09 (87% of the original logit difference). These
greedily-found sets were usually not semantically interpretable ... and
investigating them would be an interesting direction of future work."

*Discussion.* The paper's claim is bounded by its own criteria rather than by
a separate limitations section.

---

#### E. Emergent World Representations, Othello-GPT (ICLR 2023)

*Abstract.* Six sentences. Opens on a puzzle, states the research question as
a literal question ("Do these networks just memorize a collection of surface
statistics, or do they rely on internal representations of the process that
generates the sequences they see?"), then the setting, then the central
finding hedged at the verb — "we uncover evidence of an emergent nonlinear
internal representation" — with the intervention result stated flat
immediately after.

*Introduction.* A two-sided debate: "Some have suggested that training on a
sequence modeling task is inherently limiting ... On the other hand, some
tantalizing clues suggest ...". The closest prior work is then named and its
limitation given in one sentence: "The authors stop short, however, of
exploring the form of any internal representations. Such an investigation will
be the focus of this paper."

*Method.* The intervention is a gradient step on the activation toward a target
class score, given in one display equation, with hyperparameters deferred to
an appendix and the sequential-layer subtlety explained in words ("if we change
activations only at a middle layer, activations at higher layers are directly
affected by pre-intervention information").

*Experiments.* A purpose-built benchmark of 1000 natural and 1000 unnatural
cases, where the unnatural subset is justified as a stress test ("designed to
be a stringent test, since it is by definition far from anything encountered in
the training distribution"). Results are reported against an explicit null
intervention and the conclusion verb is hedged: "suggesting the emergent
representations are causal to model predictions".

*Discussion.* The method is turned into a downstream tool (latent saliency
maps) and validated by a qualitative contrast between two trained models,
again with a hedged verb ("suggests that the visualization technique is
providing useful information").

---

### 4.7 What was carried into the manuscript

1. Section pointers inside every contribution clause (Task Arithmetic, CCS).
2. Results stated with their evaluation scope attached (CCS).
3. The introduction closing on what the work establishes rather than on its
   compute cost (CCS, IOI).
4. Prior work's limitation named once, narrowly, with evidence (MEMIT,
   Othello-GPT), and each Related Work paragraph closing on our position
   (MEMIT).
5. A baseline's genuine win conceded in one sentence with the mechanism
   attached rather than defended (MEMIT). This is the register of the
   head-to-head paragraph.
6. Awkward findings presented as insights about the problem, with their
   numbers, and the unexplained part flagged in one sentence (IOI). This is
   the register of the design-ablation appendix.
7. Table captions that state the finding with its number rather than only
   describing the columns (Task Arithmetic, MEMIT).
8. Hedge the interpretation, never the measurement (Othello-GPT, CCS).

One idea was tested and **not** adopted: MEMIT's single aggregate score. We
computed the harmonic mean of removal, retention and preserved accuracy over
the five evaluation subsets. It does not change the ordering — the tuned
additive dose leads at $0.710$, our score-proportional dose is second at
$0.693$ and our gated local action third at $0.666$ — so introducing it would
add a metric without adding information.

### 4.8 Cross-paper summary (first pass, retained for the statistics)

#### First-pass notes (superseded by 4.6, kept for the record)

**Task Arithmetic (ICLR 2023), Introduction.** Opens on the practice in
present tense with no citation and no number ("Pre-trained models are commonly
used as backbones of machine learning systems"), pivots at sentence three
("In this work, we present a new paradigm..."), then spends one bolded
run-in paragraph per contribution — `Forgetting via negation.`,
`Learning via addition.`, `Task analogies.` — each of which names the
operation, points at the section that demonstrates it ("In Section 3, we
negate..."), and carries one concrete number ("maintains 98.9% of the
accuracy"). It closes on cost and reuse, not on a summary. Our Results section
already uses bolded run-in headers throughout; the contribution paragraph now
carries the same section pointers.

**CCS (ICLR 2023), Abstract and Introduction.** The abstract is seven
sentences: a problem with existing techniques, "We propose circumventing this
issue by...", "Specifically, we introduce...", "It works by...", a result
stated with its scope ("across 6 models and 10 question-answering datasets, it
outperforms zero-shot accuracy by 4% on average"), a second result, and a
close on significance-with-a-limit ("Our results provide an initial step
toward..."). Every result sentence in the introduction ends with a section
pointer, and the introduction closes on a proof-of-concept frame rather than a
triumphal one. Two things were taken from this: results are stated with their
evaluation scope attached, and the contribution paragraph now ends on what the
work establishes rather than on its compute cost.

**MEMIT (ICLR 2023), Abstract and Related Work.** The shortest abstract in the
set, five sentences: prior work's promise, its limitation in one narrow
sentence with a number attached ("predominantly limited to updating single
associations"; the intro sharpens it to "a recent study evaluates on a maximum
of 75"), "We develop MEMIT, a method for...", the result with its scale and
models, the code link. No closing significance sentence at all. The
introduction opens on a question rather than a statement ("How many memories
can we add to a deep network by directly editing its weights?"), and every
Related Work paragraph is a bolded run-in topic that ends by placing this
paper inside it ("In this paper, we take on the update problem, asking
how..."). Two things taken from this: prior work's limitation is named once,
narrowly, with a number, and each related-work paragraph closes on our
position rather than trailing off.

**IOI (ICLR 2023), Abstract and Introduction.** This is the reference whose
situation is closest to ours, because its own evaluation criteria do not fully
support its claim and it says so in the abstract: it names the three criteria
(faithfulness, completeness, minimality), states that they support the
explanation, and then writes "Though these criteria support our explanation,
they also point to remaining gaps in our understanding" before closing on
feasibility. The introduction goes further and presents its awkward findings —
redundant heads, heads writing in the opposite direction of the answer — as a
bulleted list of *insights about the problem*, introduced by "we obtained
several insights about the challenges of mechanistic interpretability". That
is the register our design-ablation appendix now uses: the members that did
not carry are reported as evidence about the selection problem, not as a
catalogue of failures.

**Othello-GPT (ICLR 2023), Abstract and Introduction.** Opens on a puzzle and
states the research question as a literal question, then "We investigate this
question in a synthetic setting by...". The central finding is hedged at the
verb — "we uncover evidence of an emergent nonlinear internal
representation", not "we show" — while the intervention result immediately
after is stated flat. The introduction is built as a two-sided debate and then
names one prior paper's limitation in a single sentence ("The authors stop
short, however, of exploring the form of any internal representations"). The
hedge-the-interpretation-never-the-measurement rule in the checklist below is
this paper's habit.

**What this implies for a paper whose baselines are competitive.** Neither
reference hedges a measurement. Both state the number flat and then bound the
claim in the next sentence — CCS with "an initial step toward", Task
Arithmetic by scoping to the operations it studied. That is the register the
head-to-head paragraph now uses: the tuned additive and CAST numbers are
stated without qualification, and the sentence after them says what those
doses buy the selectivity with, which is a fact about the measurement rather
than a defence of ours.

## 5. Shared vocabulary register (top 40)

Content words and phrases present in at least 3 of the 5 papers, ranked by
corpus frequency. Stopwords removed. This is the domain register our prose
should overlap with.

| rank | term | corpus freq | rank | term | corpus freq |
|---:|---|---:|---:|---|---:|
| 1 | task / tasks | 458 | 21 | training | 57 |
| 2 | model / models | 551 | 22 | find | 57 |
| 3 | token / tokens | 191 | 23 | experiments | 49 |
| 4 | attention | 102 | 24 | effect | 47 |
| 5 | data | 100 | 25 | behavior | 47 |
| 6 | performance | 100 | 26 | hidden | 45 |
| 7 | language | 95 | 27 | random | 45 |
| 8 | results | 94 | 28 | "language models" | 45 |
| 9 | figure | 91 | 29 | test | 44 |
| 10 | between | 87 | 30 | similar | 41 |
| 11 | section | 81 | 31 | single | 41 |
| 12 | dataset | 80 | 32 | question | 40 |
| 13 | layer / layers | 128 | 33 | weights | 40 |
| 14 | example | 74 | 34 | linear | 38 |
| 15 | knowledge | 71 | 35 | methods | 37 |
| 16 | vector / vectors | 69+ | 36 | probability | 35 |
| 17 | text | 62 | 37 | causal | 33 |
| 18 | average | 61 | 38 | loss | 32 |
| 19 | input | 61 | 39 | predictions | 32 |
| 20 | information | 60 | 40 | algorithm / network | 31 / 31 |

Paper-specific register terms worth borrowing where they fit our claims:
*probe / probing* (77), *intervention* (57), *activations* (55), *direction*
(46), *baseline* (30), *ablation* (19), *faithful / faithfulness* (10),
*causal mediator*, *counterfactual*, *specificity*, *generalization*, *fluency*,
*legible*, *completeness*, *minimality*, *zero-shot*, *control task*.

### Flags: words a 2026 LLM draft would get wrong

**Overused by LLM drafts, rare or absent in this corpus** (per five whole papers):

| term | corpus count | verdict |
|---|---:|---|
| `furthermore` | 2 | near-absent; use "moreover" (22) or nothing |
| `crucial` / `pivotal` / `paramount` | 2 / 0 / 0 | avoid entirely |
| `delve` | 0 | avoid entirely |
| `as expected` | 0 | avoid entirely |
| `state-of-the-art` (spelled out) | 2 | rare; name the baseline instead |
| `significantly` | 8 | only with a test |
| `robust` / `robustness` | 12 | acceptable but do not lean on it |
| `it is worth noting` | 0 | avoid entirely |
| `comprehensive` / `holistic` / `nuanced` | ~0 | avoid entirely |
| `leverage` (as verb) | few | prefer "use" (`we use` = 66) |

**Underused by LLM drafts, frequent here:** first-person plural (19.8–31.4 /1k —
we are at 3.3), `we find` (33), `we use` (66), `note that` (24), `suggests that`
(22), `however` (47), `intuitively` (9), `surprisingly` (6), and plain
quantity words `roughly / approximately / about` (36).

## 6. House-style checklist

Concrete rules for the manuscript, in rough order of expected effect on
perplexity and vocabulary divergence from the reference set.

1. **Raise first-person plural to at least 15 "we" per 1000 words** (references:
   19.8–31.4; we are at 3.3). Convert agentless constructions to first person:
   "the gate is calibrated on held-out data" → "we calibrate the gate on held-out
   data". This is the highest-value single edit in this document.
2. **Raise hedge density to 2–5 markers per 1000 words** (references: 2.0–7.4; we
   are at 0.5). Hedge the *interpretation*, never the *measurement*: state the
   number flat, then open the next sentence with "This suggests that".
3. **Cap semicolons at 3.5 per 1000 words** (references: 0.6–3.4; we are at 5.3).
   Use one only to join two independent clauses a reader would otherwise
   mis-parse; prefer a full stop whenever both halves stand alone.
4. **Cap colons at 12 per 1000 words** (references: 2.5–12.7; we are at 16.4).
   Convert run-in "X: Y" headers into sentences.
5. **Cap parentheses at 20 per 1000 words** (references: 7.5–18.7; we are at
   28.0). Keep them for cross-references and numbers; move clause-length asides
   into the sentence or onto an em-dash.
6. **Keep em-dashes at or below 2 per 1000 words** (references: 0.0–1.9; we are
   at 2.3). They are the right destination for asides currently in parentheses,
   so trim elsewhere to make room.
7. **Hold median sentence length at 18–22 words and mean at 20–24**, with
   sentences over 45 words under 8% of the total. We are at median 18, mean 19.8,
   3.9% — in range, with room to lengthen slightly rather than shorten.
8. **Abstract: 7–9 sentences, 180–230 words.** Open on the field in present tense
   with no citation and no number; pivot to "In this work, we ..." at sentence 2
   or 3; carry two or three concrete numbers, at least one of which quantifies
   evaluation *scope* (how many models, how many datasets); close on
   significance-with-a-limit, not on a summary. Never open with "We present" or
   "In this paper".
9. **Introduction: 500–650 words, contributions inline, not bulleted.** Zero of
   the five references bullets its contributions. Use the corpus form: "In
   summary, our contributions are that we (i) ..., (ii) ..., (iii) ...", with a
   section or figure pointer inside each clause.
10. **State the prior work's limitation concretely and narrowly.** "Predominantly
    limited to updating single associations", not "however, challenges remain".
    One named limitation, one sentence.
11. **Prefer `we use` to `we propose`.** The corpus ratio is better than 4:1 in
    favor of doing over proposing. Describe what was done; let the contribution
    be inferred.
12. **Precede every theorem with a one-sentence plain statement of what it buys,
    and follow it with a plain restatement.** The reference corpus contains zero
    theorems and only IOI's short `definition` blocks; our formal environments are
    a deliberate deviation and must be cushioned in prose to stay legible to this
    audience.
13. **Bind every symbol in the sentence that introduces its equation.** No
    notation table in the main body — none of the five references has one. Never
    reuse a symbol for two objects (the draft's `d` collision between Cohen's d
    and model width is exactly the failure mode).
14. **Reference floats inside the claim sentence, not as a trailing
    parenthetical.** Write "we find X (Table 3)" or "X is shown in Table 3", not
    "Table 3 shows X" — the table-as-subject construction occurs once in the
    entire reference corpus.
15. **Push configuration to the appendix and point at it.** The references carry
    10–47 appendix cross-references each; the main-text setup paragraph is two to
    four sentences naming only models, data and metric.
16. **Quantify every comparison** ("1.88x further", "from .396 to .128", "by 4% on
    average across 6 models and 10 datasets"); never use "significantly" without a
    stated test. The corpus uses "significantly" only 8 times in ~37k words.
17. **Report the scope of the evaluation before the size of the gain**, in the
    abstract and at the top of each results subsection.
18. **Keep "Limitations and Assumptions" as a section but write it in the corpus
    register:** 150–300 words, two to four specific items, each tied to a claim
    made earlier by number, each naming what the method does *not* do. The
    references fold this into Discussion at 200–670 words with a deflationary
    tone.
19. **Close on an under-sell.** "An initial step toward", "points to remaining
    gaps" — the corpus consistently claims less in the conclusion than in the
    abstract. A separate conclusion, if kept, runs ~250 words and repeats no
    numbers.
20. **Ban the LLM register outright:** `furthermore`, `crucial`, `pivotal`,
    `delve`, `as expected`, `it is worth noting`, `comprehensive`, `holistic`,
    `nuanced`, `underscores`, `showcases`. Their combined count in five full ICLR
    papers is under five occurrences.
21. **Prefer "use" to "leverage", "we find" to "our findings indicate", "about"
    to "approximately"** where either fits. `we find` (33), `we use` (66) and
    `roughly/approximately/about` (36) are all high-frequency in the corpus.
22. **One claim per paragraph, carried by the first sentence.**
23. **Name a method once in full, then by its short name**; do not alternate
    between "projection-transport steering" and "PTS" at random.
24. **Use "however" freely (corpus: 47 occurrences, ~1.3 /1k) and "moreover"
    sparingly (22, ~0.6 /1k); do not use "furthermore" at all.**
25. **Do not stack three or more modifiers before a noun**, and do not use the
    rule-of-three list as a default rhetorical shape — the corpus uses two-item
    contrasts far more often than three-item lists.
