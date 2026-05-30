"""
Entry point for one (method, task, seed) run.

Flow:
  1. Snapshot pretrained target weights W0 (before any adaptation).
  2. Fine-tune with LoRA/DoRA.
  3. Merge the adapter -> recover What = W0 + dW.
  4. Extract per-layer angles + complexity estimators.
  5. Append a flat row per layer to results/results.csv.

Run:
  python -m src.run --config configs/smoke.yaml
  python -m src.run --config configs/sst2_dora.yaml --method lora --task mrpc
"""

from __future__ import annotations

import argparse
import copy
import csv
import os

import yaml
import numpy as np

from .train import train_one, TARGET_MODULES
from .angles import summarize_layer
from .bounds import Ad


def _is_target(name: str) -> bool:
    return any(t.split(".")[-1] in name or t in name for t in TARGET_MODULES)


def snapshot_base_weights(peft_model) -> dict[str, np.ndarray]:
    """Grab pretrained Linear weights of target modules BEFORE merging.

    In a peft model the original layer weight is preserved as `base_layer.weight`
    inside each LoRA-wrapped module, so we can read W0 directly even after
    training (the base weights are frozen).
    """
    base = {}
    for name, module in peft_model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            w = module.base_layer.weight.detach().cpu().numpy()
            base[name] = w.copy()
    return base


def merged_weights(peft_model) -> dict[str, np.ndarray]:
    """Merge adapter into base and read What = W0 + dW for the same modules.

    We merge a *copy* so the live model is untouched. merge_and_unload folds the
    adapter into the Linear weight in place on the copy.
    """
    merged = copy.deepcopy(peft_model).merge_and_unload()
    out = {}
    for name, module in merged.named_modules():
        # after unload, wrapped modules become plain nn.Linear at the same path
        if hasattr(module, "weight") and _is_target(name) and module.weight.dim() == 2:
            out[name] = module.weight.detach().cpu().numpy().copy()
    return out


def align_names(base: dict, merged: dict) -> list[tuple[str, str]]:
    """Match base-layer module paths to their merged counterparts by suffix.

    peft path:   roberta...attention.self.query  (with .base_layer in base dict)
    merged path: roberta...attention.self.query  (plain Linear)
    We strip a trailing '.base_layer' if present and match on the remainder.
    """
    pairs = []
    for bname in base:
        key = bname[:-len(".base_layer")] if bname.endswith(".base_layer") else bname
        if key in merged:
            pairs.append((bname, key))
    return pairs


def layer_type(name: str) -> str:
    """Classify a module path as 'attn' or 'mlp' for prediction P3."""
    if "attention" in name:
        return "attn"
    if "intermediate" in name or "output.dense" in name:
        # roberta FFN: intermediate.dense (up) + output.dense (down)
        return "mlp" if "intermediate" in name or "output.dense" in name else "attn"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method")            # CLI overrides for config
    ap.add_argument("--task")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k in ("method", "task", "seed"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    kappa = cfg.get("kappa", 10.0)
    tau = cfg.get("tau", 0.01)

    print(f"[run] method={cfg['method']} task={cfg['task']} seed={cfg.get('seed', 0)}")
    model, tok, metrics = train_one(cfg)
    acc = metrics.get("eval_accuracy", float("nan"))
    print(f"[run] eval_accuracy={acc:.4f}")

    base = snapshot_base_weights(model)
    merged = merged_weights(model)
    pairs = align_names(base, merged)
    print(f"[run] matched {len(pairs)} target layers")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "results.csv")
    write_header = not os.path.exists(out_path)
    fields = ["method", "task", "seed", "kappa", "layer", "layer_type",
              "eval_accuracy", "test_error", "n_cols", "s", "s_over_d",
              "theta_mean", "theta_var", "theta_max", "theta_global",
              "c_dora_asym", "c_map_asym", "c_dora_nonasym"]

    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for bname, mname in pairs:
            W0, What = base[bname], merged[mname]
            d = W0.shape[0]                       # ambient dim per column (d_out)
            summ = summarize_layer(W0, What, kappa, tau)
            ad = Ad(kappa, d)                     # non-asymptotic factor for this layer
            row = {
                "method": cfg["method"], "task": cfg["task"], "seed": cfg.get("seed", 0),
                "kappa": kappa, "layer": mname, "layer_type": layer_type(mname),
                "eval_accuracy": acc, "test_error": 1.0 - acc,
                "n_cols": summ["n_cols"], "s": summ["s"], "s_over_d": summ["s_over_d"],
                "theta_mean": summ["theta_mean"], "theta_var": summ["theta_var"],
                "theta_max": summ["theta_max"], "theta_global": summ["theta_global"],
                "c_dora_asym": summ["c_dora_asym"], "c_map_asym": summ["c_map_asym"],
                "c_dora_nonasym": summ["c_dora_asym"] * ad,
            }
            w.writerow(row)
    print(f"[run] wrote {len(pairs)} rows to {out_path}")


if __name__ == "__main__":
    main()
