# Reproducibility

Python 3.11+, [uv](https://docs.astral.sh/uv/), and the committed `uv.lock` define the environment. Model runs require a CUDA GPU and Hugging Face access to the selected models. CPU checks and released-record analysis do not load model weights.

    uv sync --frozen
    make smoke

## Released records

`results/release/` contains the multiple-choice, toxicity, fresh toxicity, and PCHI per-example records. Fitted overconfidence directions are in `behaviour_specific/overconfidence/directions/`; toxicity decisions and selected configurations are under `results/toxicity/`. The frozen multiple-choice selections and protocols are under `results/v4_*`, `results/v5q_*`, and `results/v6a_*`.

Recompute selectivity and paired bootstrap intervals directly from the multiple-choice records:

    uv run python results/recompute_from_records.py v4_confirm det_q60_alpha-0.375
    uv run python results/recompute_from_records.py v5q_confirm det_q40_ablate
    uv run python results/recompute_from_records.py v6a_confirm det_q30_alpha-0.25

The record split and arm names are taken verbatim from the release. Research summaries and the earlier five-subset records remain on the archive branch linked below.

## Original pipeline

    make extract
    make compare SEED=7
    make steer SEED=7

Repeat evaluation with seeds 7, 11, 23, 31, and 47. These seeds resample questions; generation is greedy. Runs write outputs under `results/`.

The projection-local and gated implementations can also be run directly:

    uv run python -m behaviour_specific.overconfidence.steer_v3_optimal --benchmark mmlu --n 300 --seed 7
    uv run python -m behaviour_specific.overconfidence.steer_v5_sweep --gates ocwcr_crq50,lda_allq50 --actions ablate,clamp_q30,clamp_q50
    uv run python -m behaviour_specific.overconfidence.analyze_steering_v2 --steer-dir steering_v2

## Confirmatory experiments

`label_pool` defines the disjoint extraction, detector, tuning, and confirmation splits. `fit_detector` fits the decisions; `steer_v7_tuned` generates the action/gate arms; `select_configs` applies the common tuning rule; `analyze_pooled` computes paired statistics. Use each module's `--help` for its arguments.

    for split in extraction tuning detector confirm; do
      uv run python -m behaviour_specific.overconfidence.label_pool --split "$split"
    done
    uv run python -m behaviour_specific.overconfidence.fit_detector
    uv run python -m behaviour_specific.overconfidence.fit_detector --kind prefix
    uv run python -m behaviour_specific.overconfidence.fit_detector --kind prompt
    uv run python -m behaviour_specific.overconfidence.fit_detector --kind cast

For Qwen, set `MODEL_IMPL=qwen2_5_7b_it` and `PTS_DIRECTIONS=behaviour_specific/overconfidence/directions/qwen2_5_7b_it`. Model adapters also support Gemma-2-9B and Mistral-7B. The full grids, selected arm names, benchmark settings, and GPU job commands are preserved in the [frozen protocols](https://github.com/mariklolik/projection-transport-steering/tree/archive/pr2-research-2026-10-04/rebuttal/protocols). Follow those exact settings to reproduce the reported experiment; CLI defaults alone do not specify the paper's full grid.

For toxicity, download `allenai/real-toxicity-prompts` into the Hugging Face cache before running `behaviour_specific.toxicity.run`. Its stages are `base`, `fit`, `steer`, and `transport`. `behaviour_specific.toxicity.analyze` and `fresh` analyze the original and fresh confirmation splits; the latter uses `results/release/fresh_selection.json`. `behaviour_specific.pchi.run` exposes `generate`, `feats`, `fit`, `eval`, and `analyze` stages. Exact arms and settings are in the same frozen protocols.

## Paper

`paper/` contains only `paper.tex` and `paper.pdf`. The source embeds its style files and CSV plot data and includes the generated tables directly. A TeX Live installation with TikZ/PGFPlots can compile it in a temporary directory:

    build=$(mktemp -d)
    cp paper/paper.tex "$build/"
    (cd "$build" && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && pdflatex -interaction=nonstopmode -halt-on-error paper.tex && pdflatex -interaction=nonstopmode -halt-on-error paper.tex)

Compilation writes its embedded resources and normal TeX build files into that directory.

## Research archive

The [complete pre-cleanup repository](https://github.com/mariklolik/projection-transport-steering/tree/archive/pr2-research-2026-10-04) preserves research notes, protocols, audit reports, intermediate results, table/figure generators, earlier manuscripts, and the original README evidence. PR #2's [submission branch](https://github.com/mariklolik/projection-transport-steering/tree/iclr2027-submission) is also retained. These materials are intentionally outside the default branch.
