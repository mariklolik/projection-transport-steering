# Projection-Transport Steering (PTS)

Code, directions, and per-question rollouts for the paper
"Projection-Transport Steering: Conditional Control of Reasoning Behavior in
LLMs" (paper/iclr2027.pdf, built from paper/iclr2027.tex).

## Setup

```bash
uv sync                       # creates the environment from pyproject/uv.lock
export HF_HOME=...            # gemma-2 weights are gated; accept terms on HF
export MODEL_IMPL=gemma_2_2b_it   # or gemma_2_9b_it (layer 24 instead of 14)
```

## Reproduce (one A100/H100; ~6 GPU-h for the full 2B suite)

```bash
P="uv run python"
# 1. fit: directions, projection stats, gate calibration (~1 GPU-h)
$P -m behaviour_specific.overconfidence.extract_projection_stats --n 400 --layer 14
$P -m behaviour_specific.overconfidence.extract_joint_stats --layer 14
$P -m behaviour_specific.overconfidence.extract_probe_gate --layer 14

# 2. steering conditions (baseline/additive/ablate/clamp/OT + think-only)
$P -m behaviour_specific.overconfidence.steer_v2 --n 300 --layer 14
# 3. conditional variants and the headline gated ablation
$P -m behaviour_specific.overconfidence.steer_v2_gated --taus cr_q50,cr_q70 --actions otq_cal,clamp_q50
$P -m behaviour_specific.overconfidence.steer_v5_sweep --gates ocwcr_crq50,lda_allq50 --actions ablate,clamp_q30,clamp_q50
# 4. head-to-head baselines (prompting, CAST-style, MiMiC, Linear-AcT, probe-gated)
$P -m behaviour_specific.overconfidence.steer_v4_sota

# 5. analysis: surgical metrics, transitions, AURC, markdown report
$P -m behaviour_specific.overconfidence.analyze_steering_v2 --steer-dir steering_v2
```

## Depth, single-pass gate, and the added evidence

```bash
# where to read the gate and where to act (writes per-layer fits and moments)
$P -m behaviour_specific.overconfidence.diag_layers --layers 4,7,10,12,14,16,18,22,25 --dump-moments
# the running-mean deviation scale the sequential boundary needs
$P -m behaviour_specific.overconfidence.calibrate_gate --gate-layer 16
# single-pass conditional steering: gate read at 16, action at 14, no regeneration
$P -m behaviour_specific.overconfidence.steer_v6_online --outdir steering_v2

# equal-budget tuning for every method on a held-out split
for m in additive cast mimic act ours online; do
  $P -m behaviour_specific.overconfidence.sweep_parity --method $m --n 400; done
# capability away from the fitted behavior, and whether the edit stays on-manifold
for t in wikitext humaneval openended; do
  $P -m behaviour_specific.overconfidence.eval_openended --task $t; done
$P -m behaviour_specific.overconfidence.judge_openended
$P -m behaviour_specific.overconfidence.diag_manifold --benchmark arc --rollouts steering_v2_arc
# end-to-end wall clock, post-hoc gate charged for its second pass
$P -m behaviour_specific.overconfidence.bench_serving
```

`rebuttal/run/dispatch.py` fans a file of job chains out over the GPUs of
several hosts and is how the grids above were run.

Other benchmarks: add `--benchmark arc|gsm8k|gpqa` (transfer runs reuse the
MMLU-fitted files in `behaviour_specific/overconfidence/directions/`).
Multi-seed: repeat steps 2-5 with `--seed 11 23 31 47 --outdir steering_v2_s<seed>`,
then `python3 paper/make_multiseed.py`.

## Artifacts

- `behaviour_specific/overconfidence/directions/` — fitted directions, token
  quantile grids, gate parameters (19 KB per behavior at serve time).
- `results/steering_v2*/` — analysis JSONs and markdown reports per benchmark
  and seed; `results/steering_v2_rollouts.tgz` — full per-question rollouts
  (raw traces before/after every condition).
- `paper/` — LaTeX source, generated tables (`gen/`), figure data (`pic/viz/`).
