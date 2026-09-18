# Projection-Transport Steering (PTS)

Code, directions, and per-question rollouts for the paper
"Projection-Transport Steering: Provably Minimal-Perturbation Control of
Reasoning Behavior in LLMs" (paper/paper.pdf, built from paper/paper.tex).

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
