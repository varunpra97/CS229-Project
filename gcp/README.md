# Running the full GLUE sweep on GCP (team of 3 machines)

This runs the paper protocol — RoBERTa-base fine-tuned on **all 6 GLUE tasks**,
each with **LoRA, DoRA, and MAP**, **3 seeds**, on the **full dataset** — split
across three GPU VMs (one per teammate, **2 tasks each**). Each machine produces
the per-task plots for the tasks it ran.

Each machine runs in **two steps**: a cheap **smoke** run that proves the
pipeline works without burning credits, then the **full** run on the entire
dataset.

---

## What dataset is used?

**GLUE** (General Language Understanding Evaluation), pulled from the Hugging Face
Hub as `nyu-mll/glue` in `src/data.py`. Each run fine-tunes RoBERTa-base on one of
the six tasks below. Training uses the task's `train` split; **accuracy is measured
on the `validation` split** (GLUE's test labels are private). For MNLI we evaluate
on `validation_matched`.

| Task  | What it is                    | Train size | Eval (validation) |
|-------|-------------------------------|-----------:|------------------:|
| MNLI  | NLI (3-class)                 |   ~393,000 |            ~9,800 |
| QQP   | duplicate-question detection  |   ~364,000 |           ~40,400 |
| QNLI  | question/answer entailment    |   ~105,000 |            ~5,500 |
| SST-2 | sentiment (binary)            |    ~67,000 |               872 |
| MRPC  | paraphrase detection          |     ~3,700 |               408 |
| RTE   | textual entailment            |     ~2,500 |               277 |

The smoke step caps **training** to a small slice (default 2,000 examples) so it
finishes in minutes; the full step uses the entire `train` split.

---

## 0. Task assignment (2 tasks per machine)

Tasks are paired big+small so the three machines finish in roughly similar time.
Each owner runs **smoke first, then full** for their two tasks.

| Machine | Owner       | Tasks            | Smoke (step 4a)                    | Full (step 4b)                   |
|---------|-------------|------------------|------------------------------------|----------------------------------|
| **A (Varun)**   | teammate 1  | **MNLI + RTE**   | `bash gcp/run_smoke.sh mnli rte`  | `bash gcp/run_full.sh mnli rte`  |
| **B (Soham)**   | teammate 2  | **QQP + MRPC**   | `bash gcp/run_smoke.sh qqp mrpc`  | `bash gcp/run_full.sh qqp mrpc`  |
| **C (Stephen)**   | teammate 3  | **QNLI + SST-2** | `bash gcp/run_smoke.sh qnli sst2` | `bash gcp/run_full.sh qnli sst2` |

Each **full** call trains `3 methods × 3 seeds = 9 runs per task` on the entire
training set, then writes `results/report_<task>.pdf`.

> Want a different split? Any task can run alone, e.g. `bash gcp/run_full.sh qqp`.

---

## 1. One-time local setup (each teammate, on their laptop)

Install the Google Cloud CLI and point it at the project that holds your credits:

```bash
# install gcloud: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID          # the project with your credits
gcloud services enable compute.googleapis.com
```

**GPU quota:** new projects ship with **0 GPU quota**, so the VM create will
fail until you raise it. This is the most common blocker — see the detailed
walkthrough in step 2 below (you must raise **both** the global *and* the
regional quota, and billing must be enabled first).

---

## 2. Create the GPU VM (each teammate)

We use Google's **Deep Learning VM** (PyTorch image) — CUDA + a GPU build of
PyTorch are preinstalled, so there are no driver headaches.

> **⚠️ Prerequisite: request GPU quota first.** New GCP projects ship with a
> **global GPU quota of 0** (`GPUS_ALL_REGIONS = 0`), so the create command will
> fail with `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally` no matter
> which zone or GPU you pick. Walk through these in order:
>
> **1. Confirm billing is enabled** (GPU quota requests on free-trial accounts
> are usually rejected):
>
> ```bash
> PROJECT=$(gcloud config get-value project)
> gcloud billing projects describe "$PROJECT" \
>   --format="value(billingEnabled, billingAccountName)"   # want: True  billingAccounts/XXXX
> ```
>
> (If `gcloud billing` says the component isn't installed, run
> `gcloud components install beta` or just check billing in the Console.)
>
> **2. Raise both quotas** on the quotas page —
> `https://console.cloud.google.com/iam-admin/quotas?project=<YOUR_PROJECT>`:
>
> - **`GPUs (all regions)`** (`compute.googleapis.com/gpus_all_regions`) → **1** — this is the global cap that defaults to 0.
> - the **per-GPU regional** quota in the region you'll use — **`NVIDIA L4 GPUs`** (or `NVIDIA T4 GPUs`) in e.g. `us-east1` → **1**.
>
> **Both the global *and* the regional quota must be ≥ 1**, or the create still
> fails. Submit with a short justification (e.g. *"ML coursework — single GPU
> for fine-tuning"*); approval is often automatic within minutes but can take up
> to ~2 business days. You'll get an email when it's granted.
>
> **3. Verify the increase landed** before retrying — the global limit should
> read `1.0`:
>
> ```bash
> gcloud compute project-info describe --flatten="quotas[]" \
>   --format="value(quotas.metric, quotas.limit)" | grep -i gpus_all_regions
> ```

> **Pick a current PyTorch image family.** Google retired the old
> `pytorch-latest-gpu` family; the published families are now versioned. List
> what's available before creating the VM and use one of the names it prints:
>
> ```bash
> gcloud compute images list --project deeplearning-platform-release \
>   --filter="family~pytorch" --format="value(family)" | sort -u
> ```
>
> The commands below use `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, which was
> current as of this writing — substitute whatever the list command shows.

```bash
# --- pick your names/zone ---
export VM=peft-gpu               # any name
export ZONE=us-central1-a        # a zone with T4 capacity

# T4 (16 GB, cheapest) — good default.
gcloud compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --metadata="install-nvidia-driver=True"
```

> **zsh tip:** if you paste comment lines and see `zsh: no matches found: ...`,
> run `setopt interactive_comments` once (zsh doesn't treat `#` as a comment
> interactively by default). The error is harmless — it just skips the comment.

**Faster option (recommended for MNLI/QQP, and often more available than T4):**
use an **L4** (24 GB, ~2× faster than T4). L4 is built into `g2` machine types,
so there's **no `--accelerator` line** — just use `g2-standard-8`:

```bash
export VM=peft-gpu
export ZONE=us-east1-d        # a zone where you hold L4 quota; us-east1-c has had L4 capacity
                             # (note the hyphen: us-east1-c, not us-east1c)

gcloud compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --metadata="install-nvidia-driver=True"
```

> **`ZONE_RESOURCE_POOL_EXHAUSTED` / `..._WITH_DETAILS`?** That zone is
> temporarily out of the GPU you asked for — a transient capacity issue, not a
> config error. The error usually **names other zones that currently have
> capacity** (`zonesAvailable: us-east1-c,us-east1-b`) — just set `ZONE` to one
> of those and re-run. T4 and L4 are offered in most `us-central1`, `us-east1`,
> `us-east4`, `us-west1`, and `us-west4` zones. **Use a zone name with the
> hyphen** (`us-east1-c`), not `us-east1c` — the latter gives a misleading
> `Permission denied on 'locations/...' (or it may not exist)` error.

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

## 4a. STEP 1 — smoke run (cheap, do this first)

Verify the whole pipeline on your GPU **before** spending credits on the full
data. This caps training to 2,000 examples/task, 1 epoch, 1 seed — a few minutes,
cents of cost. Run inside **tmux** so it survives an SSH disconnect:

```bash
tmux new -s smoke
cd ~/pacbayes-peft

# Machine A:
bash gcp/run_smoke.sh mnli rte
# Machine B:   bash gcp/run_smoke.sh qqp mrpc
# Machine C:   bash gcp/run_smoke.sh qnli sst2
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t smoke`. Watch progress:
`tail -f results/sweep.log`.

Outputs (kept **separate** from the full run so they never collide):
- `results/report_<task>_smoke.pdf` — the plots, to eyeball that everything works
- `results/runs_smoke/*.pkl` — raw smoke runs

Accuracies will be low (training is deliberately tiny) — that's expected. You're
only checking the job runs green and the PDFs render.

---

## 4b. STEP 2 — full run (entire dataset)

Once the smoke PDF looks right, launch the real run on the **entire dataset**
(3 methods × 3 seeds × 3 epochs per task). This is long (hours — see the cost
table at the bottom), so use `SHUTDOWN=1` to power the VM off when it finishes:

```bash
tmux new -s full
cd ~/pacbayes-peft

# Machine A:
SHUTDOWN=1 bash gcp/run_full.sh mnli rte
# Machine B:   SHUTDOWN=1 bash gcp/run_full.sh qqp mrpc
# Machine C:   SHUTDOWN=1 bash gcp/run_full.sh qnli sst2
```

When it finishes you'll have, for each of your tasks:

- `results/report_<task>.pdf` — the 4-page directional-update report (the plots)
- `results/runs/<task>_<method>_seed<n>.pkl` — raw per-run angles/metrics

**Resume:** finished runs are skipped automatically, so if a VM is interrupted
just re-run the same `run_full.sh` command — it picks up where it left off.

**Tuning knobs** (env vars on either script): `BATCH=64` (more throughput on
L4/A100), `SEEDS="0"` (one seed for a 3× cheaper first pass), `EPOCHS=2`,
`MAX_TRAIN=50000` (cap the giant tasks). Example:
`BATCH=64 SHUTDOWN=1 bash gcp/run_full.sh qqp`.

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
- **Always smoke-test first** (`gcp/run_smoke.sh`) — it's minutes and cents, and
  catches setup problems before you commit to the long full run.
- **Use L4** for MNLI/QQP — best speed-per-dollar here.
- Increase throughput with a bigger batch: `BATCH=64 bash gcp/run_full.sh ...`
  (L4/A100 have the memory).
- Cheaper first pass: `SEEDS="0" bash gcp/run_full.sh ...` (one seed instead of
  three), or cap the giant tasks with `MAX_TRAIN=50000`.
- Always `SHUTDOWN=1` (or delete the VM) so an idle GPU doesn't burn credits
  overnight.

## Script reference

| Script               | What it does                                                        |
|----------------------|---------------------------------------------------------------------|
| `gcp/setup_vm.sh`    | Installs deps on the Deep Learning VM, verifies the GPU             |
| `gcp/run_smoke.sh`   | STEP 1: limited data (2k/task, 1 epoch, 1 seed) → `report_*_smoke.pdf` |
| `gcp/run_full.sh`    | STEP 2: entire dataset, 3 methods × 3 seeds × 3 epochs → `report_*.pdf` |
| `gcp/run_task.sh`    | Shared engine the two wrappers call (override via env vars)         |
| `scripts/report_task.py` | Build the plot PDF for one task from its runs                   |
| `scripts/aggregate.py`   | Pool all tasks' runs → cross-task P1/P3 report + CSV            |
