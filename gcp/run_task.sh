#!/usr/bin/env bash
# Run one or more GLUE tasks end-to-end on a GPU VM, FULL dataset:
#   3 methods (LoRA, DoRA, MAP) x 3 seeds (0,1,2) x full training data,
#   then build the per-task directional-update report PDF for each task.
#
# Usage (run inside tmux so it survives SSH disconnects):
#   tmux new -s run
#   cd ~/pacbayes-peft && bash gcp/run_task.sh <task> [<task2> ...]
#
# Examples:
#   bash gcp/run_task.sh mnli rte          # one teammate's two tasks
#   bash gcp/run_task.sh sst2              # a single task
#
# This is the shared engine. You normally call one of the two wrappers instead:
#   gcp/run_smoke.sh  -- limited data, fast & cheap (verify everything works)
#   gcp/run_full.sh   -- entire dataset, the real run
#
# Optional env overrides:
#   SEEDS="0 1 2"   seeds to run (default 3 seeds, as the paper specifies)
#   EPOCHS=3        training epochs (default 3)
#   BATCH=32        per-device batch (32 fits a 16 GB T4 at fp16; use 64 on L4/A100)
#   MAX_TRAIN=      cap train examples per task (empty = FULL data; set e.g. 2000 to limit)
#   BUCKET=gs://my-bucket   if set, syncs results/ there at the end (for pooling)
#   SHUTDOWN=1      power the VM off when finished (stops billing)
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS="$*"
if [[ -z "$TASKS" ]]; then
  echo "usage: bash gcp/run_task.sh <task> [<task2> ...]"
  echo "tasks: sst2 mrpc rte qnli qqp mnli"; exit 1
fi
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"
OUTDIR="${OUTDIR:-results/runs}"     # where per-run pickles go (full vs smoke differ)
SUFFIX="${SUFFIX:-}"                  # appended to report_<task><SUFFIX>.pdf

# Cap training set if MAX_TRAIN is set; otherwise use the entire dataset.
MAXARG=""; DATA_DESC="FULL data"
if [[ -n "${MAX_TRAIN:-}" ]]; then
  MAXARG="--max-train $MAX_TRAIN"; DATA_DESC="LIMITED to $MAX_TRAIN train examples/task"
fi

mkdir -p "$OUTDIR"
echo "=== run_task: tasks=[$TASKS] seeds=[$SEEDS] epochs=$EPOCHS batch=$BATCH ($DATA_DESC) -> $OUTDIR ==="

# Already-finished runs in $OUTDIR are skipped, so a preempted/restarted VM resumes
# where it left off. Smoke and full runs use DIFFERENT $OUTDIRs so they never collide.
python -m src.experiment \
  --tasks $TASKS --methods lora dora map --seeds $SEEDS \
  --epochs "$EPOCHS" --batch-size "$BATCH" --out-dir "$OUTDIR" $MAXARG \
  2>&1 | tee -a results/sweep.log

# Per-task plots (read from $OUTDIR, write report_<task><SUFFIX>.pdf).
for t in $TASKS; do
  python scripts/report_task.py "$t" "$OUTDIR" "results/report_${t}${SUFFIX}.pdf" \
    2>&1 | tee -a results/sweep.log
done

echo "=== DONE. Plots: results/report_<task>${SUFFIX}.pdf | raw runs: $OUTDIR/*.pkl ==="

if [[ -n "${BUCKET:-}" ]]; then
  echo "Syncing results/ to $BUCKET ..."
  gsutil -m rsync -r results "$BUCKET/results"
fi
if [[ "${SHUTDOWN:-}" == "1" ]]; then
  echo "Powering off in 30s (Ctrl-C to cancel) ..."; sleep 30; sudo shutdown -h now
fi
