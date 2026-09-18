# IRC source and runtime integration audit

2026-09-07. Engineering progress; the executable benchmark and behavioral claim remain unpassed. This audit does not change the reset's success thresholds or reopen a closed route.

## Data and measurement

The pinned EleutherAI MATH snapshot contains 7,500 training and 5,000 test records. Its complete multiset of `problem`, `level`, `type`, and `solution` matches all 12,500 records in the mirror linked by the MATH authors: zero missing or extra records. The earlier EleutherAI loader maps the original archive's train/test directories directly. The original archive itself was not recovered. This is a checked conversion-provenance chain, not a fresh byte comparison with that archive. The full-record train/test intersection is empty; problem-only duplicates, near duplicates and historical exposure still require a separate audit. Evidence: `irc-source-audit-v1.json`, SHA-256 `8c0d0c6cfe2f0bd95e0455f3995cf63b30598415e427f30202ce5827ec7c6b07`.

The general-MATH adapter calls the pinned author's last-box extractor and Math-Verify 0.9.0. Tests cover fractions, expressions, ordered tuples, unordered sets, multiple boxes, missing answers, text answers, punctuation and budget exhaustion. Four additional tests exposed variable destruction by unit normalization, including the false equality `3m = 3`; symbol-preserving normalization fixes those cases without changing the old AIME configuration. AST hashes of all 11 pre-existing functions match the unchanged remote source.

The complete training-reference self-check succeeds on 7,484 records and records 16 failures. These include missing/empty boxed references, unsupported formatting and a time expression parsed as an undefined symbolic value. No record was excluded, allocated or relabeled. No test reference was graded and no MATH model completion was generated. Self-equivalence is not evidence that every prediction/reference pair is scored correctly: handling legitimate units and heterogeneous non-algebraic answers remains unresolved. The final evaluator and invalid-reference contract must be frozen before scientific outputs, without another project-specific answer grammar.

## ReFT integration

The actual `LoreftIntervention` at `stanfordnlp/pyreft@dafd0995a366d7b47160a337dcc388eda7431821` fits the existing `TorchLayerAction` contract directly. No ReFT implementation was copied into project code. Seven CPU tests use real tiny Qwen3/Llama implementations: ranks 4 and 8 receive finite nonzero adapter gradients, frozen base parameters remain unchanged, saved/reloaded adapters reproduce logits within `rtol=1e-6, atol=1e-7`, identity hooks preserve logits exactly, and bfloat16 action outputs retain their dtype. `pyvene==0.1.8` is pinned in the experiment dependency group. These checks do not establish CUDA training, a useful fitted action, or parity with the full upstream training recipe.

## Branch contract and failure retained

The new branch module uses Transformers' cache and sampling processors with per-row Torch generators. A returned state retains a pending final sampled token; its cache contains the preceding processed prefix. Branch continuation copies that state once, preserves independent RNG state, supports left padding, and treats EOS as absorbing. The existing hook supplies interventions. Twenty-one CPU tests cover branch/order isolation, same-engine identity, split/resume replay, cached/uncached logits, finite action windows, mixed EOS rows and invalid inputs. Two additional tests exercise the probe's command-line path on saved real tiny models.

The first three real-weight probes failed before weight loading because default Python site initialization combined project Torch 2.9.1 with the container's incompatible torchvision. `irc-runtime-attempt1-failure.json` retains all three errors. The unchanged probe was relaunched with Python `-S` and only the declared project packages on its import path. A real Qwen3-class CUDA forward passed before that relaunch. This is an environment correction, not a change to the model, action, workload or scientific gate.

## Actual Qwen3-8B/H100 probe

All five Qwen3-8B weight files were read and SHA-256 verified against their pinned cache blob identities. Source hashes matched before execution and after receipt transfer. Three independent resident workers ran concurrently on fsk42 GPUs 0, 1 and 2; fsk40 was left untouched because another user's jobs occupied all eight devices.

The workload is synthetic token input, not MATH: 128 padded prefix positions and 64 generated tokens per row, bfloat16, SDPA, temperature 0.6, top-p 0.95, top-k 20, two intra-op threads and one inter-op thread. Each batch condition was measured once after a two-token warmup. The table is a kernel/workload observation, not a serving or research-throughput estimate; no uncertainty interval is justified by one timing per condition.

| Batch | Generated tokens | Reference seconds | Tokens/second | Peak allocated GiB | Identity / split replay / parent cache |
|---|---:|---:|---:|---:|---|
| 1 | 64 | 1.4848 | 43.10 | 15.47 | pass / pass / unchanged |
| 4 | 256 | 1.7448 | 146.73 | 16.05 | pass / pass / unchanged |
| 8 | 512 | 1.7843 | 286.94 | 16.76 | pass / pass / unchanged |

Source: `artifacts/engineering/irc_branch_probe_v1/batch{1,4,8}_attempt2.json`. All rows reached the synthetic 64-token limit; none terminated before the branch cut. Budget exhaustion is not treated as a scientific correctness result here because no task score is computed. Cross-batch token equivalence is not claimed from these GPU receipts. The within-engine identity and split checks passed separately at every batch size.

Batch 8 has 6.657 times the observed batch-1 throughput on this short workload. It is the leading option for the prospective real-question throughput check, not a final batch freeze. Long-context cache growth, ReFT overhead, real output-length variation, CPU scoring, persistent scheduling and eight-worker contention remain unmeasured. The measured Python-process times sum to 0.01951 H100-hours, including loading and checks but excluding container startup, imports before the timer, weight hashing, transfer and the failed import attempts. This is not complete end-to-end experiment cost.

The GPU environment used Torch 2.9.1+cu128 and Transformers 5.16.1 from the read-only historical environment under container Python 3.12.3. The local tests used Torch 2.14.0. The probe is therefore not a claim that the remote machine installed the current `uv.lock`; a clean, declared scientific runtime remains a prerequisite. All three workers exited successfully, receipts matched remote hashes, containers were removed, and no compute process remained on fsk42 at the final check.

## Decision and next dependency

Preserve the branch implementation and upstream ReFT wiring. Keep C67 unverified. The complete local suite passes 312 tests with one pre-existing Torch deprecation warning; critical Ruff checks pass. The original PDF/TeX, all old AIME function bodies and the prospective benchmark's bytes are unchanged.

Next: finalize standard heterogeneous-answer scoring and reference handling, materialize the historical-exposure/duplicate-cluster split manifest, validate ReFT in the chosen CUDA environment, and run the declared 16-question real-input runtime packet. Only then fit/select the fixed adapter and collect the action-response pilot. The response head, full-rollout development, strong-comparator frontier, independent model replication and three-family/three-construct paper gates remain open work. No SOTA or submission-ready result is inferred from this audit.

## Prose and result audit

Mode: `DRAFT_FROM_EVIDENCE`, audit notes only. Paragraph groups map respectively to the source-audit JSON, pinned upstream source and conformance tests, retained failed-launch JSON, three transferred GPU receipts and the unchanged prospective benchmark. The source/probe programs and all listed inputs are hash-bound in the integration receipt. The table was recomputed from the receipt counts/times; GiB means bytes divided by `2^30`. No plot was generated because three exact workload rows suffice. Adverse evidence and all unsupported extrapolations are retained. This section passes the audit-note provenance/self-check, not the fair-strong paper gate.
