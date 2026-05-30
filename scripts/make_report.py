"""
Build the results PDF: directional-update comparison of LoRA vs DoRA vs MAP.

Reads results/directional.pkl (written by src/experiment.py) and produces
results/report.pdf with a metrics table and a set of plots that visualize how
the three methods distribute their directional change across the weight columns.

Run:  python scripts/make_report.py
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = "results"
PKL = os.path.join(RESULTS_DIR, "directional.pkl")
PDF = os.path.join(RESULTS_DIR, "report.pdf")

# Stable method order + colors.
ORDER = ["lora", "dora", "map"]
COLORS = {"lora": "#4C72B0", "dora": "#DD8452", "map": "#55A868"}
LABEL = {"lora": "LoRA", "dora": "DoRA", "map": "MAP"}


def load(path=PKL):
    with open(path, "rb") as f:
        return pickle.load(f)


def render_report(res, pdf_path, title="Directional Updates: LoRA vs DoRA vs MAP"):
    """Write the 4-page directional report for one task's `res` dict to pdf_path."""
    with PdfPages(pdf_path) as pdf:
        page_summary(pdf, res)
        page_distributions(pdf, res)
        page_sparsity(pdf, res)
        page_layers(pdf, res)
        pdf.infodict()["Title"] = title
    return pdf_path


def methods_in(res):
    return [m for m in ORDER if m in res["methods"]]


def pooled_thetas(mres):
    """All per-column angles across all layers (radians)."""
    return np.concatenate([L["theta"] for L in mres["layers"].values()])


def pooled_thetas_by_type(mres, ltype):
    arrs = [L["theta"] for L in mres["layers"].values() if L["layer_type"] == ltype]
    return np.concatenate(arrs) if arrs else np.array([])


# --------------------------------------------------------------------------------
# Page 1: title + metrics table + headline directional-complexity comparison
# --------------------------------------------------------------------------------
def page_summary(pdf, res):
    cfg = res["cfg"]
    methods = methods_in(res)
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Directional Updates: LoRA vs DoRA vs MAP on RoBERTa-base / "
                 f"{cfg['task'].upper()}", fontsize=15, fontweight="bold")

    # --- metrics table (top) ---
    ax_t = fig.add_axes([0.06, 0.60, 0.88, 0.26]); ax_t.axis("off")
    rows = []
    for m in methods:
        mr = res["methods"][m]
        Ls = list(mr["layers"].values())
        c_dora = sum(L["c_dora"] for L in Ls)
        c_map = sum(L["c_map"] for L in Ls)
        sd = np.mean([L["s"] / L["d"] for L in Ls])
        mean_theta = np.mean(pooled_thetas(mr))
        mean_tg = np.mean([L["theta_global"] for L in Ls])
        rows.append([LABEL[m], f"{mr['accuracy']*100:.2f}%",
                     f"{mr['trainable_params']:,}",
                     f"{c_dora:.2f}", f"{c_map:.3f}",
                     f"{mean_theta:.4f}", f"{mean_tg:.4f}", f"{sd:.3f}"])
    col = ["Method", "Eval acc", "Trainable\nparams", "C_DoRA\n(Σ local)",
           "C_MAP\n(global)", "mean θ_j\n(rad)", "mean Θ_g\n(rad)", "mean s/d"]
    tbl = ax_t.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    for j in range(len(col)):
        tbl[0, j].set_facecolor("#222222"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(methods):
        tbl[i + 1, 0].set_facecolor(COLORS[m]); tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")
    ax_t.set_title("Fine-tuning metrics and aggregate directional complexity",
                   fontsize=11, pad=8)

    # --- headline bar: C_DoRA (sum of local angles) vs C_MAP (single global) ---
    ax1 = fig.add_axes([0.09, 0.10, 0.38, 0.40])
    x = np.arange(len(methods)); w = 0.38
    cdora = [sum(L["c_dora"] for L in res["methods"][m]["layers"].values()) for m in methods]
    cmap = [sum(L["c_map"] for L in res["methods"][m]["layers"].values()) for m in methods]
    ax1.bar(x - w/2, cdora, w, label="C_DoRA = κ Σⱼ(1−cosθⱼ)", color="#4C72B0")
    ax1.bar(x + w/2, cmap, w, label="C_MAP = κ Σ_layers(1−cosΘ_g)", color="#55A868")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels([LABEL[m] for m in methods])
    ax1.set_ylabel("directional complexity (log scale)")
    ax1.set_title("Local (sum) vs global (single rotation)\ncomplexity, κ=%g" % res["cfg"]["kappa"],
                  fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # --- mean per-column angle with spread ---
    ax2 = fig.add_axes([0.57, 0.10, 0.38, 0.40])
    means = [np.mean(pooled_thetas(res["methods"][m])) for m in methods]
    stds = [np.std(pooled_thetas(res["methods"][m])) for m in methods]
    tg = [np.mean([L["theta_global"] for L in res["methods"][m]["layers"].values()]) for m in methods]
    ax2.bar(x, means, 0.5, yerr=stds, capsize=4,
            color=[COLORS[m] for m in methods], alpha=0.85, label="mean θⱼ ± std")
    ax2.scatter(x, tg, color="black", marker="D", zorder=5, label="mean Θ_global")
    ax2.set_xticks(x); ax2.set_xticklabels([LABEL[m] for m in methods])
    ax2.set_ylabel("angle (radians)")
    ax2.set_title("Per-column angle θⱼ vs global angle Θ_global", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    fig.text(0.06, 0.04,
             "Θ_global is the single rotation of the whole flattened matrix (MAP view); "
             "θⱼ are the per-column rotations (DoRA view).\n"
             "Exact identity (Thm 3.10):  1−cosΘ_global = mean_j(1−cosθⱼ)  →  DoRA sums what MAP averages.",
             fontsize=8, style="italic")
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------------
# Page 2: distribution of per-column angles
# --------------------------------------------------------------------------------
def page_distributions(pdf, res):
    methods = methods_in(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Distribution of per-column directional change θⱼ (pooled over all target layers)",
                 fontsize=13, fontweight="bold")

    # histogram (log-count) of angles
    ax = axes[0]
    for m in methods:
        th = pooled_thetas(res["methods"][m])
        ax.hist(th, bins=80, histtype="step", linewidth=1.8,
                color=COLORS[m], label=f"{LABEL[m]} (n={len(th):,})")
    ax.set_yscale("log")
    ax.set_xlabel("θⱼ (radians)"); ax.set_ylabel("count (log)")
    ax.set_title("Angle histogram"); ax.legend(); ax.grid(alpha=0.3)

    # violin of angle distributions
    ax = axes[1]
    data = [pooled_thetas(res["methods"][m]) for m in methods]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(COLORS[methods[i]]); b.set_alpha(0.7)
    ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels([LABEL[m] for m in methods])
    ax.set_ylabel("θⱼ (radians)")
    ax.set_title("Angle distribution (violin)"); ax.grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.01,
             "A long right tail = a few columns rotate a lot (sparse/localized update). "
             "A concentrated body near 0 = most columns barely move.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------------
# Page 3: sparsity s/d and the sum-vs-average identity
# --------------------------------------------------------------------------------
def page_sparsity(pdf, res):
    methods = methods_in(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Sparsity of directional updates and the DoRA↔MAP identity",
                 fontsize=13, fontweight="bold")

    # effective sparsity s/d per method (box over layers)
    ax = axes[0]
    data = [[L["s"] / L["d"] for L in res["methods"][m]["layers"].values()] for m in methods]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[LABEL[m] for m in methods])
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="s/d = 0.5 (P2 crossover)")
    ax.set_ylabel("effective sparsity  s/d  (fraction of columns with θⱼ > τ)")
    ax.set_title(f"Per-layer sparsity (τ={res['cfg']['tau']} rad)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # verify identity: per layer, sum_j(1-cos) vs d*(1-cosTheta_g)
    ax = axes[1]
    for m in methods:
        xs, ys = [], []
        for L in res["methods"][m]["layers"].values():
            local_sum = np.sum(1.0 - np.cos(L["theta"]))
            global_scaled = L["d"] * (1.0 - np.cos(L["theta_global"]))
            xs.append(local_sum); ys.append(global_scaled)
        ax.scatter(xs, ys, s=18, alpha=0.7, color=COLORS[m], label=LABEL[m])
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=1, label="y = x (identity)")
    ax.set_xlabel("Σⱼ(1−cosθⱼ)   [DoRA local sum]")
    ax.set_ylabel("d·(1−cosΘ_global)   [MAP global ×d]")
    ax.set_title("DoRA local sum vs MAP global×d  (Thm 3.10 relation,\n"
                 "approx: Θ_global uses raw — not unit — column flattening)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------------
# Page 4: attention vs MLP angular variance (P3) + layer-depth profile
# --------------------------------------------------------------------------------
def _layer_index(name):
    # roberta...layer.<i>...  -> i
    import re
    mt = re.search(r"layer\.(\d+)\.", name)
    return int(mt.group(1)) if mt else -1


def page_layers(pdf, res):
    methods = methods_in(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Where the rotation happens: attention vs MLP, and across depth",
                 fontsize=13, fontweight="bold")

    # P3: attn vs MLP angular variance
    ax = axes[0]
    x = np.arange(len(methods)); w = 0.38
    attn_var, mlp_var = [], []
    for m in methods:
        a = pooled_thetas_by_type(res["methods"][m], "attn")
        l = pooled_thetas_by_type(res["methods"][m], "mlp")
        attn_var.append(np.var(a) if a.size else 0.0)
        mlp_var.append(np.var(l) if l.size else 0.0)
    ax.bar(x - w/2, attn_var, w, label="attention", color="#8172B3")
    ax.bar(x + w/2, mlp_var, w, label="MLP", color="#CCB974")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in methods])
    ax.set_ylabel("Var(θⱼ)  (rad²)")
    ax.set_title("P3: angular variance, attention vs MLP")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i in range(len(methods)):
        ratio = mlp_var[i] / attn_var[i] if attn_var[i] > 0 else float("nan")
        ax.text(x[i], max(attn_var[i], mlp_var[i]), f"×{ratio:.2f}",
                ha="center", va="bottom", fontsize=8)

    # depth profile: mean theta per layer index
    ax = axes[1]
    for m in methods:
        by_depth = {}
        for name, L in res["methods"][m]["layers"].items():
            i = _layer_index(name)
            if i < 0:
                continue
            by_depth.setdefault(i, []).append(np.mean(L["theta"]))
        if not by_depth:
            continue
        idx = sorted(by_depth)
        ax.plot(idx, [np.mean(by_depth[i]) for i in idx], marker="o", ms=4,
                color=COLORS[m], label=LABEL[m])
    ax.set_xlabel("transformer layer index (0 = input side)")
    ax.set_ylabel("mean θⱼ (radians)")
    ax.set_title("Mean per-column angle across depth")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def main(pkl=PKL, pdf=PDF):
    if not os.path.exists(pkl):
        raise SystemExit(f"missing {pkl}; run `python -m src.experiment` first")
    render_report(load(pkl), pdf)
    print(f"[report] wrote {pdf} ({os.path.getsize(pdf)//1024} KB)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else PKL,
         sys.argv[2] if len(sys.argv) > 2 else PDF)
