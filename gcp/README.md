# Running the full GLUE sweep on GCP (team of 3 machines)

This runs the paper protocol — RoBERTa-base fine-tuned on **all 6 GLUE tasks**,
each with **LoRA, DoRA, and MAP**, **3 seeds**, on the **full dataset** — split
across three GPU VMs (one per teammate, **2 tasks each**). Each machine produces
the per-task plots for the tasks it ran.

---

## 0. Task assignment (2 tasks per machine)

Tasks are paired big+small so the three machines finish in roughly similar time.

| Machine | Owner       | Tasks            | Command on the VM                     |
|---------|-------------|------------------|---------------------------------------|
| **A**   | teammate 1  | **MNLI + RTE**   | `bash gcp/run_task.sh mnli rte`       |
| **B**   | teammate 2  | **QQP + MRPC**   | `bash gcp/run_task.sh qqp mrpc`       |
| **C**   | teammate 3  | **QNLI + SST-2** | `bash gcp/run_task.sh qnli sst2`      |

Each `run_task.sh` call trains `3 methods × 3 seeds = 9 runs per task` on the
**full** training set, then writes `results/report_<task>.pdf`.

> Want a different split? Any task can run alone, e.g. `bash gcp/run_task.sh qqp`.

---

## 1. One-time local setup (each teammate, on their laptop)

Install the Google Cloud CLI and point it at the project that holds your credits:

```bash
# install gcloud: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID          # the project with your credits
gcloud services enable compute.googleapis.com
```

**GPU quota:** new projects often have 0 GPU quota. Request at least 1 of your
chosen GPU in your zone at
`IAM & Admin → Quotas` (filter "GPUs (all regions)" and the specific
"NVIDIA T4 GPUs"). Approval is usually quick.

---

## 2. Create the GPU VM (each teammate)

We use Google's **Deep Learning VM** (PyTorch image) — CUDA + a GPU build of
PyTorch are preinstalled, so there are no driver headaches.

```bash
# --- pick your names/zone ---
export VM=peft-gpu               # any name
export ZONE=us-central1-a        # a zone with T4 capacity

# T4 (16 GB, cheapest). Good default.
gcloud compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --metadata="install-nvidia-driver=True"
```

**Faster option (recommended for MNLI/QQP):** swap the machine line for an **L4**
(24 GB, ~2× faster than T4) — L4 is built into `g2` machine types, so drop the
`--accelerator` line:

```bash
  --machine-type=g2-standard-8 \
  # (remove the --accelerator=... line; keep everything else)
```

The first boot installs the NVIDIA driver (a few minutes). SSH in:

```bash
gcloud compute ssh "$VM" --zone="$ZONE"
```

---

## 3. Set up the repo on the VM (each teammate)

```bash
# on the VM
git clone https://github.com/varunpra97/CS229-Project.git pacbayes-peft
cd pacbayes-peft
bash gcp/setup_vm.sh        # checks the GPU, installs transformers/peft/datasets/... (NOT torch)
```

`setup_vm.sh` prints the GPU name and confirms `torch.cuda.is_available() == True`.

---

## 4. Run your tasks (each teammate)

Run inside **tmux** so the job survives an SSH disconnect:

```bash
tmux new -s run
cd ~/pacbayes-peft

# Machine A:
bash gcp/run_task.sh mnli rte
# Machine B:   bash gcp/run_task.sh qqp mrpc
# Machine C:   bash gcp/run_task.sh qnli sst2
```

Detach with `Ctrl-b d`; reattach later with `tmux attach -t run`.
Watch progress: `tail -f results/sweep.log`.

This trains on the **entire dataset** (no subsampling). When it finishes you'll
have, for each of your tasks:

- `results/report_<task>.pdf` — the 4-page directional-update report (the plots)
- `results/runs/<task>_<method>_seed<n>.pkl` — raw per-run angles/metrics

**Auto power-off when done** (saves credits) — set `SHUTDOWN=1`:

```bash
SHUTDOWN=1 bash gcp/run_task.sh mnli rte
```

**Resume:** finished runs are skipped automatically, so if a VM is interrupted
just re-run the same command — it picks up where it left off.

---

## 5. Get the plots back to your laptop (each teammate)

From your **laptop** (not the VM):

```bash
gcloud compute scp --zone="$ZONE" --recurse \
  "$VM:~/pacbayes-peft/results/report_*.pdf" ./gcp-results/
# optional: also grab the raw runs for the team-wide pooled analysis (step 6)
gcloud compute scp --zone="$ZONE" --recurse \
  "$VM:~/pacbayes-peft/results/runs" ./gcp-results/runs/
```

---

## 6. (Optional) Team-wide pooled analysis — P1 / P3 across all 6 tasks

The cross-task predictions (P1: complexity↔error Spearman correlation; P3:
MLP/attention variance) need **all 6 tasks' runs in one place**. Two ways:

**Via a shared GCS bucket** (cleanest). Create it once, then each teammate adds
`BUCKET=gs://...` to their run command:

```bash
gsutil mb -l us-central1 gs://YOUR-TEAM-bucket
SHUTDOWN=1 BUCKET=gs://YOUR-TEAM-bucket bash gcp/run_task.sh mnli rte
```

Then on any one machine (or your laptop with the repo + deps):

```bash
mkdir -p results/runs
gsutil -m rsync -r gs://YOUR-TEAM-bucket/results/runs results/runs
python scripts/aggregate.py
#   -> results/report_<task>.pdf      (all 6)
#   -> results/report_aggregate.pdf   (accuracy table, P1 scatter, P3 table)
#   -> results/aggregate_summary.csv
```

**Or just collect the `results/runs/*.pkl` from each teammate** into one
`results/runs/` and run `python scripts/aggregate.py`.

---

## 7. ⚠️ Delete the VM when finished (each teammate)

A running GPU VM bills continuously. After you've copied your results off:

```bash
gcloud compute instances delete "$VM" --zone="$ZONE"
```

(`SHUTDOWN=1` only *stops* the VM — that halts GPU billing but the disk still
costs a little. `delete` removes everything.)

---

## Cost & time guidance

Dataset sizes (train examples) drive runtime: MNLI ≈ 393k, QQP ≈ 364k,
QNLI ≈ 105k, SST-2 ≈ 67k, MRPC ≈ 3.7k, RTE ≈ 2.5k. Each task does
`9 runs × 3 epochs` over its full set.

| GPU  | Machine type     | ~$/hr (on-demand) | Big-task pair (MNLI+RTE / QQP+MRPC) | Medium pair (QNLI+SST-2) |
|------|------------------|-------------------|--------------------------------------|--------------------------|
| T4   | n1-standard-8    | ~$0.35 + GPU      | ~18–24 h                             | ~8–10 h                  |
| L4   | g2-standard-8    | ~$0.7–1.0         | ~9–12 h                              | ~4–5 h                   |
| A100 | a2-highgpu-1g    | ~$3–4             | ~3–5 h                               | ~1.5–2 h                 |

Tips to cut cost/time:
- **Use L4** for MNLI/QQP — best speed-per-dollar here.
- Increase throughput with a bigger batch: `BATCH=64 bash gcp/run_task.sh ...`
  (L4/A100 have the memory).
- **Sanity-check first** with one cheap seed/epoch before the full run:
  `SEEDS="0" EPOCHS=1 bash gcp/run_task.sh rte` (RTE is tiny, ~minutes).
- Always `SHUTDOWN=1` (or delete the VM) so an idle GPU doesn't burn credits
  overnight.
