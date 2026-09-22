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
least -0.01; ties by accuracy change. Budgets: additive 8 doses; CAST-style 4
thresholds x 2 doses; MiMiC 3 layers x 3 shrinkages; Linear-AcT 3 layers x 3
strengths; PTS post-hoc 4 detector quantiles x 2 actions; PTS single pass 2
quantiles x 4 prefix budgets; detector+additive 4 quantiles x 2 doses. Gated
configurations are scored on the tuning split by composing the ungated rollouts
of the same split with the gate flags; the confirmatory arms are real runs.

## Arms on the confirmatory split

Selected configurations are recorded in `results/v4_parity_m5/selection.json`
and copied into `results/v4_confirm/protocol.json` before launch.

## Endpoints and tests

- Primary: pooled M5 selectivity, PTS post-hoc against each of prompting, tuned
  additive, directional ablation, tuned CAST-style, tuned MiMiC, tuned
  Linear-AcT. Question-level paired bootstrap, 10,000 resamples, two-sided;
  Holm adjustment over the six comparisons. Superiority is claimed only where
  the Holm-adjusted p < 0.05.
- Accuracy: non-inferiority of every arm against the unsteered model at margin
  -0.02 (lower bound of the two-sided 95% bootstrap interval of the change).
- Secondary, reported without a claim: removal, retention, ECE, M4 and M2
  selectivity, PTS single pass against PTS post-hoc, detector+additive against
  PTS post-hoc.

## Replication

The same protocol, with its own tuning split and detector, is run on
Qwen2.5-7B-Instruct (action layer 14 of 28).
