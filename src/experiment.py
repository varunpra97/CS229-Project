"""
Directional-update sweep: LoRA / DoRA / MAP on the GLUE tasks.

For each (task, method, seed) we fine-tune RoBERTa-base, then compare the
pretrained vs fine-tuned weights of the target modules through the paper's
hyperspherical lens:

  * per-column angles      theta_j   = angle(w0_j, what_j)          (DoRA / local view)
  * global angle           Theta_g   = angle(vec(W0), vec(What))    (MAP / global view)
  * directional complexity C_DoRA = kappa * sum_j (1 - cos theta_j)
                           C_MAP  = kappa * (1 - cos Theta_g)
  * effective sparsity     s = |{j : theta_j > tau}|

Each run is saved as its own pickle in `results/runs/{task}_{method}_seed{seed}.pkl`
so a long sweep is resumable (existing runs are skipped unless --overwrite) and
`scripts/aggregate.py` can pool everything afterward.

Examples
--------
  # the full paper protocol (6 tasks x 3 methods x 3 seeds, full data) on a GPU
  python -m src.experiment --tasks sst2 mrpc rte qnli qqp mnli \
      --methods lora dora map --seeds 0 1 2 --epochs 3 --batch-size 32

  # a fast local check (CPU/MPS): two small tasks, capped train set, one seed
  python -m src.experiment --tasks rte mrpc --seeds 0 --max-train 500 --epochs 1
"""

from __future__ import annotations

import argparse
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

from .data import load_glue, GLUE_TASKS
from .train import build_model, make_metric
from .map_adapter import build_map_model, map_layer_weights
from .run import snapshot_base_weights, merged_weights, align_names
from .angles import column_angles, global_angle

ALL_TASKS = ["sst2", "mrpc", "rte", "qnli", "qqp", "mnli"]
ALL_METHODS = ["lora", "dora", "map"]

# Fixed analysis/model hyperparameters (paper defaults).
MODEL_NAME = "roberta-base"
RANK = 8
LR = 2.0e-4
KAPPA = 10.0
TAU = 0.01


def device_info():
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if cuda:
        return "cuda", torch.cuda.get_device_name(0)
    if mps:
        return "mps", "Apple MPS"
    return "cpu", "CPU"


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
    base = snapshot_base_weights(model)
    merged = merged_weights(model)
    pairs = align_names(base, merged)
    return {mname: (base[bname], merged[mname]) for bname, mname in pairs}


def build(method: str, num_labels: int):
    if method == "map":
        return build_map_model(MODEL_NAME, num_labels, rank=RANK)
    return build_model(MODEL_NAME, num_labels, method, RANK)


def analyze(model, method: str, kappa: float, tau: float) -> dict:
    """Compute per-layer angles and complexity terms for a trained model."""
    layers = {}
    for name, (W0, Wf) in extract_layers(model, method).items():
        thetas = column_angles(W0, Wf)
        tg = global_angle(W0, Wf)
        d_in = int(W0.shape[1])
        layers[name] = dict(
            theta=thetas.astype(np.float32),
            theta_global=float(tg),
            layer_type=layer_type(name),
            d=d_in,
            s=int(np.sum(thetas > tau)),
            c_dora=float(kappa * np.sum(1.0 - np.cos(thetas))),
            c_map=float(kappa * (1.0 - np.cos(tg))),
        )
    return layers


def train_one_run(task, method, seed, tok, train_ds, eval_ds, num_labels, args, dev):
    """Fine-tune one (task, method, seed); return a result dict."""
    set_seed(seed)
    model = build(method, num_labels)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    targs = TrainingArguments(
        output_dir=os.path.join("out_exp", f"{task}_{method}_{seed}"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 64),
        learning_rate=LR,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.0,
        seed=seed,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
        fp16=(dev == "cuda"),                 # mixed precision only on CUDA
        dataloader_num_workers=args.num_workers,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=make_metric(task),
    )
    trainer.train()
    metrics = trainer.evaluate()
    acc = metrics.get("eval_accuracy", float("nan"))
    layers = analyze(model, method, args.kappa, args.tau)
    res = dict(task=task, method=method, seed=seed, accuracy=acc,
               trainable_params=n_train, layers=layers,
               cfg=dict(model_name=MODEL_NAME, rank=RANK, epochs=args.epochs,
                        lr=LR, kappa=args.kappa, tau=args.tau,
                        max_train=args.max_train, max_len=args.max_len))
    del model, trainer
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["sst2"], choices=ALL_TASKS)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap train examples per task (default: full data)")
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", default=os.path.join("results", "runs"))
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run and overwrite runs that already have a pickle")
    args = ap.parse_args()

    dev, dev_name = device_info()
    os.makedirs(args.out_dir, exist_ok=True)
    total = len(args.tasks) * len(args.methods) * len(args.seeds)
    print(f"[exp] device={dev} ({dev_name}) | fp16={dev=='cuda'} | "
          f"{len(args.tasks)} tasks x {len(args.methods)} methods x "
          f"{len(args.seeds)} seeds = {total} runs")
    print(f"[exp] tasks={args.tasks} methods={args.methods} seeds={args.seeds} "
          f"epochs={args.epochs} batch={args.batch_size} "
          f"max_train={args.max_train or 'full'}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    done = 0
    for task in args.tasks:
        # Load this task's data once; share the train/eval sets across methods/seeds.
        train_full, eval_ds, num_labels = load_glue(task, tok, max_len=args.max_len)
        train_ds = train_full
        if args.max_train is not None and args.max_train < len(train_full):
            train_ds = train_full.select(range(args.max_train))
        print(f"\n[exp] === task={task} | train={len(train_ds)} eval={len(eval_ds)} "
              f"num_labels={num_labels} ===")

        for seed in args.seeds:
            for method in args.methods:
                done += 1
                out = os.path.join(args.out_dir, f"{task}_{method}_seed{seed}.pkl")
                tag = f"[{done}/{total}] {task}/{method}/seed{seed}"
                if os.path.exists(out) and not args.overwrite:
                    print(f"[exp] {tag}: SKIP (exists)")
                    continue
                print(f"[exp] {tag}: training ...")
                res = train_one_run(task, method, seed, tok, train_ds, eval_ds,
                                    num_labels, args, dev)
                with open(out, "wb") as f:
                    pickle.dump(res, f)
                print(f"[exp] {tag}: acc={res['accuracy']:.4f} "
                      f"trainable={res['trainable_params']:,} -> {out}")

    print(f"\n[exp] sweep complete: {total} runs in {args.out_dir}")
    print("[exp] next: python scripts/aggregate.py")


if __name__ == "__main__":
    main()
