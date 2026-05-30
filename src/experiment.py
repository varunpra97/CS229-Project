"""
Directional-update comparison experiment: LoRA vs DoRA vs MAP on one GLUE task.

For each method we fine-tune RoBERTa-base on the SAME data subset, then compare
the pretrained vs fine-tuned weights of the target modules through the paper's
hyperspherical lens:

  * per-column angles      theta_j   = angle(w0_j, what_j)          (DoRA / local view)
  * global angle           Theta_g   = angle(vec(W0), vec(What))    (MAP / global view)
  * directional complexity C_DoRA = kappa * sum_j (1 - cos theta_j)
                           C_MAP  = kappa * (1 - cos Theta_g)
  * effective sparsity     s = |{j : theta_j > tau}|

All three methods are scored with BOTH estimators (the estimators are functions
of the resulting angles, applied regardless of how the update was produced), so
we can see how a column-wise update (DoRA) vs a global update (MAP) distribute
their directional change.

Outputs:
  results/directional.pkl  -- raw angles + per-layer/per-method metrics (for plots)
  results/summary.csv      -- one row per (method, layer), human-readable

Run:  python -m src.experiment
"""

from __future__ import annotations

import csv
import os
import pickle

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import load_glue
from .train import build_model, make_metric
from .map_adapter import build_map_model, map_layer_weights
from .run import snapshot_base_weights, merged_weights, align_names
from .angles import column_angles, global_angle

# ----- experiment configuration -------------------------------------------------
CFG = dict(
    model_name="roberta-base",
    task="sst2",
    methods=["lora", "dora", "map"],
    rank=8,
    seed=1,
    subset=4000,        # train examples (eval uses the full validation split)
    epochs=3,
    batch_size=16,
    max_len=128,
    lr=2.0e-4,
    kappa=10.0,
    tau=0.01,
)
RESULTS_DIR = "results"


def layer_type(name: str) -> str:
    """attn = Q/K/V/attention-output.dense ; mlp = intermediate.dense / FFN output.dense."""
    if "attention" in name:
        return "attn"
    if "intermediate" in name or "output.dense" in name:
        return "mlp"
    return "other"


def extract_layers(model, method: str) -> dict[str, tuple]:
    """Return {layer_name: (W0, W_final)} for the target modules of `model`."""
    if method == "map":
        return map_layer_weights(model)
    # peft (lora / dora): read frozen base weights and the merged adapter weights.
    base = snapshot_base_weights(model)
    merged = merged_weights(model)
    pairs = align_names(base, merged)
    return {mname: (base[bname], merged[mname]) for bname, mname in pairs}


def build(method: str, num_labels: int):
    if method == "map":
        return build_map_model(CFG["model_name"], num_labels, rank=CFG["rank"])
    return build_model(CFG["model_name"], num_labels, method, CFG["rank"])


def train_method(method: str, tok, train_ds, eval_ds, num_labels: int):
    """Fine-tune one method on the shared datasets; return (model, eval_metrics)."""
    set_seed(CFG["seed"])
    model = build(method, num_labels)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    args = TrainingArguments(
        output_dir=os.path.join("out_exp", method),
        per_device_train_batch_size=CFG["batch_size"],
        per_device_eval_batch_size=CFG["batch_size"],
        learning_rate=CFG["lr"],
        num_train_epochs=CFG["epochs"],
        lr_scheduler_type="cosine",
        seed=CFG["seed"],
        logging_steps=50,
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=make_metric(CFG["task"]),
    )
    trainer.train()
    metrics = trainer.evaluate()
    metrics["trainable_params"] = n_train
    return model, metrics


def analyze(model, method: str) -> dict:
    """Compute per-layer angles and complexity terms for a trained model."""
    kappa, tau = CFG["kappa"], CFG["tau"]
    layers = {}
    for name, (W0, Wf) in extract_layers(model, method).items():
        thetas = column_angles(W0, Wf)            # (d_in,)
        tg = global_angle(W0, Wf)
        d_in = int(W0.shape[1])
        layers[name] = dict(
            theta=thetas.astype(np.float32),
            theta_global=float(tg),
            layer_type=layer_type(name),
            d=d_in,
            s=int(np.sum(thetas > tau)),
            c_dora=float(kappa * np.sum(1.0 - np.cos(thetas))),   # local sum
            c_map=float(kappa * (1.0 - np.cos(tg))),              # global single
        )
    return layers


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[exp] device mps={torch.backends.mps.is_available()} | "
          f"task={CFG['task']} subset={CFG['subset']} methods={CFG['methods']}")

    tok = AutoTokenizer.from_pretrained(CFG["model_name"])
    train_ds, eval_ds, num_labels = load_glue(
        CFG["task"], tok, max_len=CFG["max_len"], subset=CFG["subset"]
    )
    # Same eval set is the full validation split (load_glue caps it by subset);
    # reload eval at full size so accuracy is comparable to the paper.
    _, eval_full, _ = load_glue(CFG["task"], tok, max_len=CFG["max_len"], subset=None)
    print(f"[exp] train={len(train_ds)} eval={len(eval_full)} examples")

    results = {"cfg": CFG, "methods": {}}
    for method in CFG["methods"]:
        print(f"\n[exp] ===== fine-tuning {method.upper()} =====")
        model, metrics = train_method(method, tok, train_ds, eval_full, num_labels)
        acc = metrics.get("eval_accuracy", float("nan"))
        print(f"[exp] {method}: eval_accuracy={acc:.4f} "
              f"trainable={metrics['trainable_params']:,}")
        layers = analyze(model, method)
        results["methods"][method] = dict(
            accuracy=acc,
            trainable_params=metrics["trainable_params"],
            layers=layers,
        )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ----- persist raw results for plotting -------------------------------------
    with open(os.path.join(RESULTS_DIR, "directional.pkl"), "wb") as f:
        pickle.dump(results, f)

    # ----- human-readable summary CSV -------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    fields = ["method", "layer", "layer_type", "accuracy", "d", "s", "s_over_d",
              "theta_mean", "theta_var", "theta_max", "theta_global",
              "c_dora", "c_map"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, mres in results["methods"].items():
            for name, L in mres["layers"].items():
                th = L["theta"]
                w.writerow(dict(
                    method=method, layer=name, layer_type=L["layer_type"],
                    accuracy=mres["accuracy"], d=L["d"], s=L["s"],
                    s_over_d=L["s"] / L["d"],
                    theta_mean=float(np.mean(th)), theta_var=float(np.var(th)),
                    theta_max=float(np.max(th)), theta_global=L["theta_global"],
                    c_dora=L["c_dora"], c_map=L["c_map"],
                ))
    print(f"\n[exp] wrote {os.path.join(RESULTS_DIR, 'directional.pkl')} and {csv_path}")

    # ----- compact metric table to stdout ---------------------------------------
    print("\n[exp] ===== SUMMARY =====")
    print(f"{'method':>6} {'acc':>7} {'trainable':>11} {'C_DoRA(sum)':>12} "
          f"{'C_MAP(glob)':>12} {'mean s/d':>9}")
    for method, mres in results["methods"].items():
        Ls = mres["layers"].values()
        c_dora = sum(L["c_dora"] for L in Ls)
        c_map = sum(L["c_map"] for L in Ls)
        sd = np.mean([L["s"] / L["d"] for L in Ls])
        print(f"{method:>6} {mres['accuracy']:>7.4f} {mres['trainable_params']:>11,} "
              f"{c_dora:>12.3f} {c_map:>12.4f} {sd:>9.3f}")


if __name__ == "__main__":
    main()
