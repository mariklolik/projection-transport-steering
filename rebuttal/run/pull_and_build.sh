#!/usr/bin/env bash
# Pull the run outputs from the cluster, regenerate every generated table and
# figure fragment, and rebuild the PDF.
#   bash rebuttal/run/pull_and_build.sh [host]
set -euo pipefail
HOST="${1:-avi-gn-fsk39}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

for d in v3_mmlu_s7 v3_mmlu_s11 v3_mmlu_s23 v3_mmlu_s31 v3_mmlu_s47 v3_arc v3_gsm8k \
         v3_9b_s7 v3_9b_s11 v3_9b_s23 v3_parity v3_openended v3_serving v3_layers \
         v3_layers_9b v3_manifold viz; do
  mkdir -p "results/$d"
  rsync -az --include='*/' --include='*.json' --include='*.csv' --include='*.md' \
        --exclude='*' -e "ssh -o BatchMode=yes" "$HOST:pts-rebuttal/results/$d/" "results/$d/" || true
done

python3 paper/make_multiseed.py
python3 paper/make_tables.py
python3 paper/make_rebuttal_tables.py
[ -f results/viz/plane_mmlu.csv ] && python3 paper/make_figdata.py || echo "no viz dump yet"
python3 paper/check_numbers.py > rebuttal/numbers.txt
python3 paper/style_check.py

cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1
grep -c "Warning.*undefined" paper.log || true
echo "pages: $(pdftotext paper.pdf - 2>/dev/null | tr -cd '\f' | wc -c)"
