# PAC-Bayes PEFT Experiments

Minimal pipeline to validate the predictions from *"PAC-Bayesian Generalization
Bounds for Weight-Decomposed Low-Rank Adaptation"*. Fine-tunes RoBERTa-base on
GLUE tasks with LoRA / DoRA, extracts directional angles, and computes the
PAC-Bayes directional complexity estimator.

## What this does (Phase 0 scope)

1. Fine-tune RoBERTa-base on one GLUE task with a chosen PEFT method.
2. Merge adapter into base weights, compare pretrained vs fine-tuned columns.
3. Extract per-column angles `theta_j` and the global angle `Theta_global`.
4. Compute the complexity estimators `C_dir` and the non-asymptotic factor `A_d(kappa)`.
5. Write a row to `results/results.csv`.

Priorities for the first pass: **P1** (C_dir predicts test error) and **P4**
(non-asymptotic vs asymptotic bound), which come almost for free.

## Setup (local CPU first — costs nothing)

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Smoke test on CPU with a tiny config (1 task, ~50 steps)
python -m src.run --config configs/smoke.yaml
```

The smoke config trains for only a handful of steps on a tiny data subset so you
can confirm the whole loop works (train -> merge -> angles -> CSV) without a GPU
and without waiting. Expect it to finish in a few minutes on a laptop CPU.

## Once the smoke test passes

Move to a real run on Colab (free T4) or GCP:

```bash
python -m src.run --config configs/sst2_dora.yaml
```

Swap method/task by editing the config or overriding on the CLI:

```bash
python -m src.run --config configs/sst2_dora.yaml --method lora --task mrpc
```

## Files

- `src/run.py`        — entry point; orchestrates one (method, task, seed) run
- `src/train.py`      — fine-tuning loop (HF Trainer)
- `src/angles.py`     — angle extraction + complexity estimators (the heart)
- `src/bounds.py`     — A_d(kappa) and PAC-Bayes bound values (P4, pure CPU)
- `src/data.py`       — GLUE loading + tokenization
- `configs/`          — YAML configs (smoke + per-task)
- `scripts/analyze_p1.py` — pool results.csv, compute Spearman correlation (P1)

## Budget reminder

Debug everything here on CPU. Only use a paid GPU for the final batch, and
always shut the VM down. Set a GCP budget alert at $25 and $40.
