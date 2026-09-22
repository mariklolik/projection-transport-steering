# Reproducibility

## Environment

- Python 3.11 or newer
- uv
- CUDA-capable GPU for model runs
- Hugging Face access to the model identifiers used by the scripts
- TeX Live with pdflatex for the paper

Install the locked environment:

    uv sync

Run the CPU-only numerical and import checks:

    make smoke

## Main experiments

The pipeline is split into direction extraction, confidence-method comparison, and steering:

    make extract
    make compare SEED=7
    make steer SEED=7

For the five MMLU question resamples:

    for seed in 7 11 23 31 47; do
      make compare SEED="$seed"
      make steer SEED="$seed"
    done

The optimal gate-times-transport implementation is:

    uv run python -m behaviour_specific.overconfidence.steer_v3_optimal --benchmark mmlu --n 300 --seed 7

Results are written under results/. Released fitted directions and projection statistics are under behaviour_specific/overconfidence/directions/.

## Added evidence

The depth sweep, the sequential-gate calibration, the equal-budget tuning, the
capability and on-manifold diagnostics and the end-to-end latency measurement
are listed with their commands in STEERING.md. `rebuttal/run/dispatch.py` fans
a file of job chains out over several hosts' GPUs, and
`rebuttal/run/finalize.sh` re-analyses every run directory, pulls the outputs
back and rebuilds the PDF.

## Tables and paper

    uv run python paper/make_tables.py
    uv run python paper/make_multiseed.py
    uv run python paper/make_rebuttal_tables.py
    uv run python paper/make_figdata.py
    uv run python paper/check_numbers.py    # every quantity the prose asserts
    uv run python paper/style_check.py      # punctuation and register
    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error iclr2027.tex
    pdflatex -interaction=nonstopmode -halt-on-error iclr2027.tex

The committed PDF is the two-pass compilation of the committed paper/iclr2027.tex.

## Post-review evidence

rebuttal/protocols/ contains the frozen method and benchmark contracts. rebuttal/evidence/ contains the split manifests, selected configurations, per-question outputs, bootstrap analyses, and runtime receipts. rebuttal/audits/ records the analysis decisions.

The headline confirmatory result is stored in:

    rebuttal/evidence/truthfulqa_olmo2_daps_v1/confirmatory/initial-analysis.json

The seven-control family is stored in:

    rebuttal/evidence/truthfulqa_olmo2_daps_v1/confirmatory/control-analysis.json

The post-review packet is preserved as an evidence-level reproduction of the reported statistics. It does not include model weights.
