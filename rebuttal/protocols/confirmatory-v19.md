# Confirmatory protocol v19: answer-level readout, fresh split

Frozen before any steered rollout on the confirmatory split was generated.

## Readout (primary): M5

The model writes its multiple-choice reasoning trace and a `\boxed{}` letter.
Correctness is the letter against the gold answer. Confidence is P(YES) for
the follow-up turn "You answered X. Is that answer correct? Reply with only
YES or NO.", read from the next-token logits in the same conversation. A trace
is confident iff P(YES) >= 0.5. When an arm steers a question, the hook is
active for the reasoning trace and for the confidence query, so the answer and
the confidence both come from the edited model. States: OCW (confident, wrong),
NCW, NCR, CR (confident, right).

Secondary readouts: M4 (per-option yes/no verification, the readout of the
earlier manuscript) and the letter-logit confidence of the same generated
answer (M2).

## Splits (MMLU test, ids fixed by `label_pool.split_records`)

| split | n | role |
|---|---|---|
| extraction | 400 | directions, projection statistics, transport targets |
| detector | 4000 | detector training (with extraction) |
| tuning | 1200 | every method's configuration, one rule for all |
| confirm | 3000 | this protocol; disjoint from all of the above and from the five legacy evaluation subsets |

## Selection rule (applied on the tuning split, identical for every method)

Highest pooled selectivity among configurations whose accuracy change is at
least -0.01; ties by accuracy change.

Amendment A (before any confirmatory steered rollout). The first tuning pass
showed that under M5 ablation converts fewer overconfident-wrong answers than a
shift of the confidence coordinate, so every gated family now searches the
same kind of grid, and the comparison between gated families is a comparison
of what the decision reads:

| family | decision reads | grid | budget |
|---|---|---|---|
| additive | nothing | dose in {-0.125,...,-1.5} x mean norm | 8 |
| CAST-style, trace condition | extraction-split OCW-CR direction on the finished trace | 4 CR-quantile thresholds x shift {-0.25,-0.375,-0.5,-0.75} | 16 |
| CAST, prompt condition | logistic probe on prompt activations, fitted on the detector pool | 4 quantiles {.3,.4,.5,.6} x the same 4 shifts | 16 |
| MiMiC | nothing | layer {10,14,18} x shrinkage {1e-4,1e-2,1e-1} | 9 |
| Linear-AcT | nothing | layer {10,14,18} x strength {.25,.5,1} | 9 |
| PTS post-hoc | logistic probe on the finished unsteered trace, fitted on the detector pool | 4 quantiles x action {ablation, shift -0.25, -0.375, -0.75} | 16 |
| PTS single pass | logistic probe on the first t tokens | quantile {.3,.5} x t {16,32,64,128} x action {ablation, shift -0.375} | 16 |

Gated configurations are scored on the tuning split by composing the ungated
rollouts of the same split with the gate flags (Proposition 1 makes this exact
up to batched-decoding nondeterminism); single-pass configurations are real
runs. The confirmatory arms are real runs.

Amendment B (before any confirmatory steered rollout). A CAST baseline
faithful to the original condition mechanism is added: the condition vector is
the difference of mean prompt activations between OCW and CR questions of the
detector pool, the gate is its cosine similarity with the mean prompt
activation, and the grid is condition layer {14,16} x cosine quantile {.3,.5} x
shift {-0.25,-0.375,-0.5,-0.75} (16). The logistic probe on prompt activations
is PTS with the decision taken at the prompt (t = 0); it is reported as a
secondary PTS variant, not as a baseline. The primary comparison family is
therefore seven baselines against PTS post-hoc.

## Arms on the confirmatory split

Selected configurations are recorded in `results/v4_parity_m5/selection.json`
and copied into `results/v4_confirm/protocol.json` before launch.

## Endpoints and tests

- Primary: pooled M5 selectivity, PTS post-hoc against each of prompting, tuned
  additive, directional ablation, tuned CAST-style (trace condition), tuned
  CAST (prompt condition, diff-in-means and cosine), tuned MiMiC, tuned
  Linear-AcT. Question-level paired
  bootstrap, 10,000 resamples, two-sided; Holm adjustment over the seven
  comparisons. Superiority is claimed only where
  the Holm-adjusted p < 0.05.
- Accuracy: non-inferiority of every arm against the unsteered model at margin
  -0.02 (lower bound of the two-sided 95% bootstrap interval of the change).
- Secondary, reported without a claim: removal, retention, ECE, M4 and M2
  selectivity, PTS single pass against PTS post-hoc, detector+additive against
  PTS post-hoc.

## Replication

The same protocol, with its own tuning split and detector, is run on
Qwen2.5-7B-Instruct (action layer 14 of 28).
