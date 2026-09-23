#!/usr/bin/env bash
# Last mile: re-analyse every run directory on the cluster, dump the figure
# coordinates, pull everything back, regenerate all tables and the PDF.
#   bash rebuttal/run/finalize.sh [host]
set -uo pipefail
HOST="${1:-avi-gn-fsk40}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ssh -n -o BatchMode=yes "$HOST" "cd ~/pts-rebuttal && export PYTHONPATH=. HF_HUB_OFFLINE=1 && \
  for d in v3_mmlu_s7 v3_mmlu_s11 v3_mmlu_s23 v3_mmlu_s31 v3_mmlu_s47 v3_arc v3_gsm8k \
           v3_9b_s7 v3_9b_s11 v3_9b_s23; do \
    [ -f results/\$d/rollouts/baseline_m4__shard0.jsonl ] && \
      .venv/bin/python -m behaviour_specific.overconfidence.analyze_steering_v2 --steer-dir \$d \
        > logs/analyze_\$d.log 2>&1 && echo \"analysed \$d\"; \
  done; \
  .venv/bin/python -m behaviour_specific.overconfidence.exp1_trace_analysis --steer-dir v3_mmlu_s7 \
    > logs/trace_s7.log 2>&1 && echo 'trace analysis done'"

bash rebuttal/run/pull_and_build.sh "$HOST"
