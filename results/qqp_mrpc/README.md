# QQP + MRPC results (Machine B / Soham)

RoBERTa-base, rank-8 PEFT, GLUE, 3 seeds (0/1/2), batch 64. Produced on a GCP L4 VM
(`peft-gpu`, us-east1-b). These are the QQP+MRPC slice only; the all-task aggregates
(`report_mnli/qnli/rte/sst2.pdf`, `aggregate_summary.csv`, `bound_values.csv`) at the
`results/` top level are the team's combined artifacts and are left untouched.

## Folder map

| folder | n pkls | what |
|---|---|---|
| `runs_v1_3ep/` | 18 | **Original** sweep. Methods: LoRA, DoRA, MAP (original Cartesian adapter). QQP + MRPC, 3 epochs. Untouched baseline. |
| `runs_v2_3ep/` | 24 | **Rerun.** Adds `map_geo` (polar-reparam MAP) as a 4th method **and** train-error instrumentation (`train_accuracy`, `n_train`). Same 3-epoch budget. Contains both **QQP@3ep** and **MRPC@3ep** (task is encoded in the filename). |
| `runs_v2_mrpc_10ep/` | 12 | MRPC at **10 epochs**, all 4 methods. Added to test whether original MAP's collapse is a training-budget issue or a parametrization issue. |
| `smoke/` | 6 | Single-seed smoke runs (sanity only). |
| `reports/` | 5 PDFs | `report_{qqp,mrpc}.pdf` (v1), the `_smoke` variants, and `report.pdf`. |
| `logs/` | 5 | `full.log` (v1), `v2_full.log` / `v2_mrpc.log` (v2 sweeps), `sweep.log`, `smoke.log`. |
| `aggregates/` | 3 | `results.csv`, `summary.csv`, `directional.pkl` — VM-generated v1 aggregates (regenerable from the pkls). |

pkl schema: `accuracy, train_accuracy, train_loss, n_train, layers, method, seed, task, cfg, trainable_params`.

## Results (accuracy, mean over seeds)

| task / budget | LoRA | DoRA | MAP (orig) | MAP-geo (new) |
|---|---|---|---|---|
| QQP @3ep | .8997 | .9005 | .8688 | **.8908** |
| MRPC @3ep | .8407 | .8570 | .6838 *(collapsed, σ=0)* | **.7034** |
| MRPC @10ep | .8873 | .8832 | .7312 *(still plateaued)* | **.8587** |

## How the rerun (v2) affected the data

1. **Reproduced v1 exactly** for the shared methods (QQP DoRA .9005, MAP .8688; MRPC
   LoRA .8407, DoRA .8570, MAP .6838) — the pipeline is deterministic, so v2 is purely
   additive, not corrective. Originals are preserved untouched in `runs_v1_3ep/`.
2. **Added `map_geo`**, a theory-faithful polar (ρ, φ) reparametrization of MAP (init =
   W₀ exactly, φ = global angle Θ). It beats original MAP in every regime
   (+2.2 QQP, +2.0 MRPC@3ep, **+12.7 MRPC@10ep**).
3. **The 10-epoch MRPC sweep is the decisive new data:** original MAP stays stuck at
   .7312 with 3× the budget while `map_geo` reaches .8587 — proving the collapse is the
   parametrization (∂Θ/∂β ≈ 1/‖W₀‖ throttling), not under-training.
4. **Instrumentation** (`train_accuracy`, `n_train`) enables the corrected P1 partial
   correlation and the empirical P4 McAllester/Catoni bound computation, neither of
   which v1 pkls can support.

## Reproduce the analysis

```
python scripts/analyze_predictions.py --runs-dir results/qqp_mrpc/runs_v2_3ep
python scripts/analyze_predictions.py --runs-dir results/qqp_mrpc/runs_v2_mrpc_10ep
python scripts/analyze_predictions.py --runs-dir results/qqp_mrpc/runs_v1_3ep
```

Prediction status on this 2-task slice: **P4** confirmed (76.8× overestimate at κ=10);
**P1** under-powered (needs the full 6-task pool; sign turns +0.46 at MRPC@10ep
convergence); **P3** contradicted on both tasks (MLP/Attn var ratio 0.34–1.16 < 1.5).
