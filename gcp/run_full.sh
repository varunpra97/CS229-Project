#!/usr/bin/env bash
# STEP 2 — the real run on the ENTIRE dataset (no subsampling), 3 methods x 3 seeds
# x 3 epochs per task, then per-task plots. Run this only after gcp/run_smoke.sh
# looks correct.
#
#   tmux new -s full
#   cd ~/pacbayes-peft && bash gcp/run_full.sh <task> [<task2> ...]
#   e.g.  bash gcp/run_full.sh mnli rte
#
# This is long (hours — see the cost table in gcp/README.md). Recommended:
#   SHUTDOWN=1 bash gcp/run_full.sh mnli rte      # power off the VM when done
#
# Outputs:
#   results/runs/*.pkl          (raw per-run angles/metrics; feeds aggregate.py)
#   results/report_<task>.pdf   (the plots for each task)
set -euo pipefail
cd "$(dirname "$0")"

# Full data = MAX_TRAIN unset. 3 seeds + 3 epochs are the paper defaults.
SEEDS="${SEEDS:-0 1 2}" \
EPOCHS="${EPOCHS:-3}" \
BATCH="${BATCH:-32}" \
OUTDIR="${OUTDIR:-results/runs}" \
SUFFIX="${SUFFIX:-}" \
  bash run_task.sh "$@"

echo
echo ">>> Full run complete. Plots: results/report_<task>.pdf"
echo ">>> Raw runs in results/runs/ feed the team-wide analysis (scripts/aggregate.py)."
