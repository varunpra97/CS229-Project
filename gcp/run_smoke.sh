#!/usr/bin/env bash
# STEP 1 — cheap smoke run. Verify the whole pipeline (train -> angles -> plots)
# works on your GPU VM WITHOUT spending real credits, using a small slice of each
# task's training data.
#
#   tmux new -s smoke
#   cd ~/pacbayes-peft && bash gcp/run_smoke.sh <task> [<task2> ...]
#   e.g.  bash gcp/run_smoke.sh mnli rte
#
# Defaults: 2000 train examples/task, 1 epoch, 1 seed -> finishes in a few minutes
# and costs cents. Eval still runs on the FULL validation split, so accuracies are
# real (just low, because training is short). Outputs are kept SEPARATE from the
# full run:
#   results/runs_smoke/*.pkl        (raw)
#   results/report_<task>_smoke.pdf (plots)
#
# Override any default inline, e.g.:  MAX_TRAIN=1000 SEEDS="0 1" bash gcp/run_smoke.sh rte
set -euo pipefail
cd "$(dirname "$0")"

MAX_TRAIN="${MAX_TRAIN:-2000}" \
SEEDS="${SEEDS:-0}" \
EPOCHS="${EPOCHS:-1}" \
BATCH="${BATCH:-32}" \
OUTDIR="${OUTDIR:-results/runs_smoke}" \
SUFFIX="${SUFFIX:-_smoke}" \
  bash run_task.sh "$@"

echo
echo ">>> Smoke check passed. Inspect results/report_<task>_smoke.pdf."
echo ">>> If it looks right, launch the real run:  bash gcp/run_full.sh $*"
