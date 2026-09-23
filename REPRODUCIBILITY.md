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

## Confirmatory protocol (answer-level readout M5)

The paper's main results come from the protocol in `rebuttal/protocols/confirmatory-v19.md`.
Each step below runs per shard; `rebuttal/run/dispatch.py` fans the shards out.

    P="uv run python -m behaviour_specific.overconfidence"
    # 1. unsteered answers, M2/M4/M5 labels and activation features for every split
    for split in extraction tuning detector confirm; do $P.label_pool --split $split; done
    # 2. decisions: trace, prefix and prompt probes, and the CAST condition vector
    $P.fit_detector; $P.fit_detector --kind prefix; $P.fit_detector --kind prompt; $P.fit_detector --kind cast
    # 3. ungated runs of every configuration on the tuning split, then one selection rule for all methods
    $P.steer_v7_tuned --split tuning --methods plain_alpha-0.25 --readouts m5   # etc., see rebuttal/run/queue54.jobs
    $P.select_configs --dir v4_tuning --out v4_parity_m5
    # 4. confirmatory arms, then paired analysis and the identity check
    python3 rebuttal/run/make_confirm_jobs.py --selection results/v4_parity_m5/selection.json --outdir v4_confirm \
        --parity-dir v4_parity_m5 --jobs confirm.jobs --protocol results/v4_confirm/protocol.json
    $P.analyze_pooled --dirs v4_confirm --conds <arms> --ref <PTS arm> --readouts m5,m5m2,m4 --out v4_pooled/confirm_gemma.json
    $P.law_confirm --checks <arm:decision:ungated:config,...> --out v4_pooled/gate_law_confirm_m5.json
    $P.decision_auroc --dirs v4_confirm --out v4_pooled/decision_auroc_gemma.json
    # 5. capability, transfer and examples
    $P.capability_v4 --task humaneval; $P.capability_v4 --task openended; $P.capability_v4 --task wikitext
    $P.dump_examples

For Qwen2.5-7B-Instruct set `MODEL_IMPL=qwen2_5_7b_it` and
`PTS_DIRECTIONS=behaviour_specific/overconfidence/directions/qwen2_5_7b_it`, fit the
directions with `extract_projection_stats --layer 14` and `diag_layers --layers 10,14,18
--dump-moments`, and use `v5q_*` output directories. The SHA-256 hashes of the selected
configurations and fitted detectors, recorded before the confirmatory runs, are in
`rebuttal/protocols/confirmatory-v19-*.sha256`. Tables and figures:

    python3 paper/make_confirm_tables.py; python3 paper/make_v4_figdata.py; python3 paper/make_examples.py

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
