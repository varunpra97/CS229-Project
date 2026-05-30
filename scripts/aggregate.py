"""
Aggregate a full GLUE sweep into the paper's tables and predictions.

Reads every `results/runs/*.pkl` written by `src.experiment` and produces:

  * results/report_<task>.pdf       -- the 4-page directional report per task
  * results/report_aggregate.pdf    -- cross-task summary:
        - accuracy table (tasks x methods, mean +/- std over seeds) + bars
        - P1: pooled Spearman correlation of directional complexity vs test error
        - P3: MLP/attention angular-variance ratio per task
  * results/aggregate_summary.csv   -- one row per (task, method, seed)

Run (after the sweep finishes):  python scripts/aggregate.py
"""

from __future__ import annotations

import csv
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, pearsonr

# Reuse the per-task plotting + styling from make_report.
sys.path.insert(0, os.path.dirname(__file__))
import make_report as MR  # noqa: E402

RUNS_DIR = os.path.join("results", "runs")
TASK_ORDER = ["sst2", "mrpc", "rte", "qnli", "qqp", "mnli"]
ORDER, COLORS, LABEL = MR.ORDER, MR.COLORS, MR.LABEL


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.pkl"))):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    if not runs:
        raise SystemExit(f"no runs found in {RUNS_DIR}; run `python -m src.experiment` first")
    return runs


def tasks_present(runs):
    seen = {r["task"] for r in runs}
    return [t for t in TASK_ORDER if t in seen] + sorted(seen - set(TASK_ORDER))


def methods_present(runs):
    seen = {r["method"] for r in runs}
    return [m for m in ORDER if m in seen]


def run_totals(run):
    """Aggregate per-run quantities used by the cross-task analyses."""
    Ls = list(run["layers"].values())
    c_dora = sum(L["c_dora"] for L in Ls)
    c_map = sum(L["c_map"] for L in Ls)
    sd = float(np.mean([L["s"] / L["d"] for L in Ls]))
    attn = np.concatenate([L["theta"] for L in Ls if L["layer_type"] == "attn"]) \
        if any(L["layer_type"] == "attn" for L in Ls) else np.array([])
    mlp = np.concatenate([L["theta"] for L in Ls if L["layer_type"] == "mlp"]) \
        if any(L["layer_type"] == "mlp" for L in Ls) else np.array([])
    return dict(c_dora=c_dora, c_map=c_map, s_over_d=sd,
                attn_var=float(np.var(attn)) if attn.size else float("nan"),
                mlp_var=float(np.var(mlp)) if mlp.size else float("nan"))


# --------------------------------------------------------------------------------
# Per-task reports (reuse make_report's 4-page layout)
# --------------------------------------------------------------------------------
def per_task_reports(runs):
    by_task_seed = defaultdict(dict)
    for r in runs:
        by_task_seed[(r["task"], r["seed"])][r["method"]] = r
    written = []
    for task in tasks_present(runs):
        # pick the lowest seed that has the most methods for the geometry plots
        seeds = sorted({s for (t, s) in by_task_seed if t == task})
        best = max(seeds, key=lambda s: len(by_task_seed[(task, s)]))
        method_runs = by_task_seed[(task, best)]
        res = {"cfg": {**next(iter(method_runs.values()))["cfg"], "task": task},
               "methods": {m: dict(accuracy=method_runs[m]["accuracy"],
                                   trainable_params=method_runs[m]["trainable_params"],
                                   layers=method_runs[m]["layers"])
                           for m in ORDER if m in method_runs}}
        out = os.path.join("results", f"report_{task}.pdf")
        MR.render_report(res, out,
                         title=f"Directional Updates: {task.upper()} (seed {best})")
        written.append(out)
        print(f"[agg] wrote {out} (seed {best}, methods={list(res['methods'])})")
    return written


# --------------------------------------------------------------------------------
# Aggregate CSV
# --------------------------------------------------------------------------------
def write_csv(runs):
    path = os.path.join("results", "aggregate_summary.csv")
    fields = ["task", "method", "seed", "accuracy", "test_error",
              "trainable_params", "c_dora_total", "c_map_total",
              "mean_s_over_d", "attn_var", "mlp_var", "mlp_attn_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r["task"], r["method"], r["seed"])):
            t = run_totals(r)
            ratio = t["mlp_var"] / t["attn_var"] if t["attn_var"] else float("nan")
            w.writerow(dict(task=r["task"], method=r["method"], seed=r["seed"],
                            accuracy=r["accuracy"], test_error=1.0 - r["accuracy"],
                            trainable_params=r["trainable_params"],
                            c_dora_total=t["c_dora"], c_map_total=t["c_map"],
                            mean_s_over_d=t["s_over_d"],
                            attn_var=t["attn_var"], mlp_var=t["mlp_var"],
                            mlp_attn_ratio=ratio))
    print(f"[agg] wrote {path}")
    return path


# --------------------------------------------------------------------------------
# Aggregate PDF
# --------------------------------------------------------------------------------
def page_accuracy(pdf, runs, tasks, methods):
    # mean +/- std accuracy over seeds, per (task, method)
    acc = defaultdict(list)
    for r in runs:
        acc[(r["task"], r["method"])].append(r["accuracy"])

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("GLUE accuracy by method (mean ± std over seeds)",
                 fontsize=15, fontweight="bold")

    ax_t = fig.add_axes([0.06, 0.55, 0.88, 0.33]); ax_t.axis("off")
    rows = []
    for t in tasks:
        row = [t.upper()]
        for m in methods:
            vals = acc.get((t, m), [])
            row.append(f"{np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}" if vals else "—")
        rows.append(row)
    col = ["Task"] + [LABEL[m] for m in methods]
    tbl = ax_t.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.9)
    for j in range(len(col)):
        tbl[0, j].set_facecolor("#222222"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j, m in enumerate(methods):
        for i in range(len(tasks)):
            tbl[i + 1, j + 1].set_text_props(color=COLORS[m])
    ax_t.set_title("Eval accuracy (%)", fontsize=11, pad=8)

    ax = fig.add_axes([0.10, 0.10, 0.82, 0.36])
    x = np.arange(len(tasks)); w = 0.8 / max(1, len(methods))
    for k, m in enumerate(methods):
        means = [np.mean(acc.get((t, m), [np.nan])) * 100 for t in tasks]
        stds = [np.std(acc.get((t, m), [0])) * 100 for t in tasks]
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, means, w, yerr=stds,
               capsize=3, label=LABEL[m], color=COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("eval accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def page_p1(pdf, runs, methods):
    """P1: directional complexity vs test error, pooled over (method,task,layer-type,seed)."""
    pts = []  # (c_dir, err, method)
    for r in runs:
        err = 1.0 - r["accuracy"]
        for lt in ("attn", "mlp"):
            c = sum(L["c_dora"] for L in r["layers"].values() if L["layer_type"] == lt)
            if c > 0:
                pts.append((c, err, r["method"]))
    if len(pts) < 3:
        return
    c = np.array([p[0] for p in pts]); err = np.array([p[1] for p in pts])
    rho_s, p_s = spearmanr(c, err)
    rho_p, p_p = pearsonr(np.log(c + 1e-12), err)

    fig, ax = plt.subplots(figsize=(11, 7))
    for m in methods:
        xs = [p[0] for p in pts if p[2] == m]
        ys = [p[1] for p in pts if p[2] == m]
        ax.scatter(xs, ys, s=28, alpha=0.7, color=COLORS[m], label=LABEL[m])
    ax.set_xscale("log")
    ax.set_xlabel("directional complexity  Ĉ_dir = κ Σⱼ(1−cosθⱼ)  per (run, layer-type)")
    ax.set_ylabel("test error (1 − accuracy)")
    ax.set_title("P1: directional complexity vs test error  (pooled, n=%d points)\n"
                 "Spearman ρ = %+.3f (p=%.2g)   |   Pearson(ln Ĉ) ρ = %+.3f (p=%.2g)"
                 % (len(pts), rho_s, p_s, rho_p, p_p), fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.text(0.5, 0.01, "Paper prediction P1: ρ_S > 0.5, p < 0.01 (their pooled test uses 108 points).",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    pdf.savefig(fig); plt.close(fig)


def page_p3(pdf, runs, tasks, methods):
    """P3: MLP/attention angular-variance ratio per task (averaged over methods, seeds)."""
    ratios = defaultdict(list)
    for r in runs:
        t = run_totals(r)
        if t["attn_var"] and not np.isnan(t["attn_var"]):
            ratios[r["task"]].append(t["mlp_var"] / t["attn_var"])

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("P3: MLP vs attention angular variance  Var(θⱼ)", fontsize=15, fontweight="bold")

    ax_t = fig.add_axes([0.15, 0.55, 0.70, 0.32]); ax_t.axis("off")
    rows = []
    for t in tasks:
        vals = ratios.get(t, [])
        rows.append([t.upper(),
                     f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if vals else "—",
                     "✓ > 1" if vals and np.mean(vals) > 1 else ("✗" if vals else "—")])
    tbl = ax_t.table(cellText=rows, colLabels=["Task", "MLP/Attn ratio", "P3 (pred > 1)"],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    for j in range(3):
        tbl[0, j].set_facecolor("#222222"); tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax = fig.add_axes([0.12, 0.10, 0.78, 0.36])
    x = np.arange(len(tasks))
    means = [np.mean(ratios.get(t, [np.nan])) for t in tasks]
    stds = [np.std(ratios.get(t, [0])) for t in tasks]
    ax.bar(x, means, 0.6, yerr=stds, capsize=4, color="#CCB974")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="ratio = 1 (no difference)")
    ax.set_xticks(x); ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("Var(θⱼ)  MLP / attention")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def main():
    runs = load_runs()
    tasks = tasks_present(runs)
    methods = methods_present(runs)
    print(f"[agg] {len(runs)} runs | tasks={tasks} methods={methods}")

    per_task_reports(runs)
    write_csv(runs)

    agg = os.path.join("results", "report_aggregate.pdf")
    with PdfPages(agg) as pdf:
        page_accuracy(pdf, runs, tasks, methods)
        page_p1(pdf, runs, methods)
        page_p3(pdf, runs, tasks, methods)
        pdf.infodict()["Title"] = "GLUE sweep aggregate: LoRA vs DoRA vs MAP"
    print(f"[agg] wrote {agg}")


if __name__ == "__main__":
    main()
