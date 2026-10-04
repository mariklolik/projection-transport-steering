# Reproducibility

The default branch contains the core multiple-choice PTS implementation, its model adapters, a record-analysis script, the README figures, and the paper. Optional experiment suites and all saved artifacts live on the [complete reproduction branch](https://github.com/mariklolik/projection-transport-steering/tree/archive/reproduction-full-2026-10-04). The [research archive](https://github.com/mariklolik/projection-transport-steering/tree/archive/pr2-research-2026-10-04) preserves notes, protocols, generators, and historical evidence.

## CPU verification

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. The environment is locked; smoke checks import the retained modules and run their numerical self-tests without loading model weights.

    make sync
    make smoke

Download the released records and fitted overconfidence directions from the immutable archived commit. Downloads are ignored by Git and do not expand the tracked default-branch tree.

    make artifacts
    make records

`make records` recomputes Gemma selectivity and paired bootstrap intervals from the released per-question records. Other released confirmatory splits:

    uv run --frozen python results/recompute_from_records.py v5q_confirm det_q40_ablate
    uv run --frozen python results/recompute_from_records.py v6a_confirm det_q30_alpha-0.25

## Core model reproduction

Model runs require a CUDA GPU and Hugging Face access to the selected model. Gemma-2-2B is the default. `make artifacts` supplies the released fitted directions and detector; the baseline and steering commands evaluate the disjoint 3,000-question confirmation split with the M5 readout.

    make baseline
    make steer

The steering target runs the selected post-hoc PTS configuration `det_q60_alpha-0.375` and its ungated action `plain_alpha-0.375`. Paired analysis:

    uv run --frozen python -m behaviour_specific.overconfidence.analyze_pooled \
      --dirs v4_confirm --conds det_q60_alpha-0.375,plain_alpha-0.375 \
      --ref det_q60_alpha-0.375 --readouts m5 --out v4_confirm/analysis.json

To refit projection statistics and action-layer moments, run `make extract`. To retrain the decisions, `label_pool` creates the extraction, detector, tuning, and held-out pools; `fit_detector` trains trace, prompt, prefix, and CAST decisions. `select_configs` applies the original common tuning rule. Exact split definitions, training/held-out arguments, grids, and frozen selections are in the [archived protocols](https://github.com/mariklolik/projection-transport-steering/tree/archive/pr2-research-2026-10-04/rebuttal/protocols); command defaults alone are not the complete paper protocol.

Set `MODEL_IMPL=qwen2_5_7b_it` and `PTS_DIRECTIONS=behaviour_specific/overconfidence/directions/qwen2_5_7b_it` for Qwen. Model adapters for Gemma-2-9B and Mistral-7B are also retained. Core CLI arguments are available with `--help` on `label_pool`, `fit_detector`, `select_configs`, `steer`, and `analyze_pooled`.

## Complete paper reproduction

The archive branches retain the original five-subset pipeline, toxicity and PCHI suites, capability tests, latency benchmarks, SAE experiments, legacy runners, complete released artifacts, and table/figure generators. Check out the [full reproduction branch](https://github.com/mariklolik/projection-transport-steering/tree/archive/reproduction-full-2026-10-04) to run those suites, following its guide and the frozen protocols.

The helper modules retained on main preserve their original callable implementations. Obsolete standalone runner blocks were removed; the model entry point is `steer`.

## Paper

`paper/` contains only `paper.tex` and `paper.pdf`. Tables, plot data, and local style resources are embedded in the TeX source. Compile with TeX Live/TikZ/PGFPlots in a temporary directory:

    build=$(mktemp -d)
    cp paper/paper.tex "$build/"
    (cd "$build" && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && pdflatex -interaction=nonstopmode -halt-on-error paper.tex)
