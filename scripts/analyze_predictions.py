"""
Corrected / extended evaluation of the paper's predictions P1, P3, P4.

This is an ADDITION alongside scripts/aggregate.py (which is left intact). It fixes
two ways the original analysis diverged from the paper's own theory and fills in the
parts of the P4 protocol that were missing:

  P1  The bound is  err <~ R_hat + sqrt(C_dir / n)  (Section 5.2), so the quantity
      that should track test error is sqrt(C_dir / n), NOT raw C_dir. Pooling raw
      C_dir across tasks of very different n makes it track dataset size instead of
      generalization (and flips the sign). We report raw vs normalized, all-methods
      vs column-wise-only, and -- when train accuracy is recorded -- the partial
      correlation controlling for training error, exactly as P1 specifies.

  P3  The paper measures Var(theta_j) per weight matrix, averaged across layers/seeds,
      with success criterion ratio > 1.5 (Table 2). The original pooled all angles
      into one variance (mixing between-matrix means) and scored against 1.0. P3 is a
      column-wise-update statement, so we report it for DoRA/LoRA (MAP's per-column
      angles are degenerate by construction).

  P4  Beyond the A_d(kappa) table, we compute the actual McAllester RHS under the
      asymptotic (A_d=1) vs non-asymptotic A_d(kappa) KL on the empirical {theta_j},
      and the Catoni bound (Theorem 3.8) at its optimal lambda*, as the protocol asks.

Usage:
  python scripts/analyze_predictions.py --runs-dir gcp-results/runs        # v1 (original MAP)
  python scripts/analyze_predictions.py --runs-dir results/runs_v2         # v2 (instrumented + map_geo)
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bounds import Ad, overestimate_ratio, mcallester_bound  # noqa: E402

# Full-train sizes for the GLUE tasks (the n in sqrt(C_dir/n)). Falls back to a
# run's recorded n_train when present (instrumented runs store it).
GLUE_N = {"sst2": 67349, "mrpc": 3668, "rte": 2490, "qnli": 104743,
          "qqp": 363846, "mnli": 392702}
COLUMN_WISE = ("lora", "dora")          # methods whose per-column theta_j are meaningful for P3


def load_runs(runs_dir):
    runs = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(runs_dir, "*.pkl")))]
    if not runs:
        raise SystemExit(f"no runs in {runs_dir}")
    return runs


def n_of(run):
    return int(run.get("n_train") or GLUE_N.get(run["task"], 1))


def c_by_type(run, lt):
    return sum(L["c_dora"] for L in run["layers"].values() if L["layer_type"] == lt)


# --------------------------------------------------------------------------- P1
def p1(runs):
    print("=" * 78)
    print("P1  directional complexity vs test error   (target: Spearman rho > +0.5)")
    print("=" * 78)

    def points(quantity, methods=None):
        xs, ys = [], []
        for r in runs:
            if methods and r["method"] not in methods:
                continue
            err = 1.0 - r["accuracy"]
            n = n_of(r)
            for lt in ("attn", "mlp"):
                c = c_by_type(r, lt)
                if c <= 0:
                    continue
                xs.append(np.sqrt(c / n) if quantity == "sqrt_c_over_n" else c)
                ys.append(err)
        return np.array(xs), np.array(ys)

    for label, q in [("raw  C_dir", "raw"), ("sqrt(C_dir / n)  [the bound's quantity]", "sqrt_c_over_n")]:
        for mlabel, methods in [("all methods", None), ("DoRA/LoRA only", COLUMN_WISE)]:
            x, y = points(q, methods)
            if len(x) < 3:
                continue
            rs, ps = spearmanr(x, y)
            print(f"  {label:42s} | {mlabel:15s} n={len(x):3d}  Spearman={rs:+.3f} (p={ps:.2g})")
    print()

    # Partial correlation controlling for training error (needs recorded train acc).
    have_train = [r for r in runs if np.isfinite(r.get("train_accuracy", np.nan))]
    if len(have_train) >= 4:
        c, err, terr = [], [], []
        for r in have_train:
            n = n_of(r)
            for lt in ("attn", "mlp"):
                cc = c_by_type(r, lt)
                if cc <= 0:
                    continue
                c.append(np.sqrt(cc / n)); err.append(1 - r["accuracy"])
                terr.append(1 - r["train_accuracy"])
        c, err, terr = map(np.array, (c, err, terr))
        # partial Spearman(C, err | train_err) via residuals of rank regression
        def resid(a, b):
            from scipy.stats import rankdata
            ra, rb = rankdata(a), rankdata(b)
            beta = np.polyfit(rb, ra, 1)[0]
            return ra - (beta * rb + (ra.mean() - beta * rb.mean()))
        rc, re_ = resid(c, terr), resid(err, terr)
        pr, pp = pearsonr(rc, re_)
        print(f"  partial corr  sqrt(C/n) vs err | controlling train error : {pr:+.3f} (p={pp:.2g}, n={len(c)})")
    else:
        print("  partial-correlation P1 (control for train error): NEEDS instrumented runs "
              "(train_accuracy not in these pkls).")
    print("  NOTE: full P1 = 108 pts over 6 tasks; a 2-task slice cannot reach the target.\n")


# --------------------------------------------------------------------------- P3
def p3(runs):
    print("=" * 78)
    print("P3  MLP vs attention angular variance   (target: Var ratio MLP/Attn > 1.5)")
    print("=" * 78)
    by_task = defaultdict(list)
    for r in runs:
        if r["method"] not in COLUMN_WISE:
            continue
        av = [np.var(L["theta"]) for L in r["layers"].values() if L["layer_type"] == "attn"]
        mv = [np.var(L["theta"]) for L in r["layers"].values() if L["layer_type"] == "mlp"]
        if av and mv and np.mean(av) > 0:
            by_task[r["task"]].append(np.mean(mv) / np.mean(av))
    for t, v in sorted(by_task.items()):
        ok = "PASS" if np.mean(v) > 1.5 else ("dir. ok" if np.mean(v) > 1 else "FAIL")
        print(f"  {t:5s}: per-matrix MLP/Attn Var ratio = {np.mean(v):.2f} ± {np.std(v):.2f}   [{ok}]")
    print("  (computed on DoRA/LoRA only; MAP per-column angles are degenerate for P3)\n")


# --------------------------------------------------------------------------- P4
def p4(runs, kappa=10.0, d=768, delta=0.05):
    print("=" * 78)
    print(f"P4  non-asymptotic vMF tightness   (target: overestimate > 10% at kappa={kappa:g})")
    print("=" * 78)
    print(f"  {'kappa':>6} {'A_d(kappa)':>14} {'overestimate 1/A_d':>20}")
    for k in (1, 5, 10, 50, 100):
        print(f"  {k:>6} {Ad(k, d):>14.4e} {overestimate_ratio(k, d):>20.2f}")

    # Actual bound values on an empirical run's angles (largest column-wise run available).
    pick = max((r for r in runs if r["method"] in COLUMN_WISE),
               key=lambda r: sum(len(L["theta"]) for L in r["layers"].values()), default=None)
    if pick is not None:
        thetas = np.concatenate([L["theta"] for L in pick["layers"].values()])
        n = n_of(pick)
        Rhat = 1.0 - pick.get("train_accuracy", pick["accuracy"])   # empirical risk (train if recorded)
        s = float(np.sum(1.0 - np.cos(thetas)))
        kl_asym = kappa * s                  # A_d = 1
        kl_nonasym = kappa * Ad(kappa, d) * s
        b_asym = mcallester_bound(Rhat, kl_asym, n, delta)
        b_non = mcallester_bound(Rhat, kl_nonasym, n, delta)
        print(f"\n  empirical {pick['task']}/{pick['method']}/seed{pick['seed']}: "
              f"sum(1-cos) over {len(thetas):,} cols = {s:.1f}, n={n:,}, Rhat={Rhat:.3f}")
        print(f"    KL  asymptotic (A_d=1)      = {kl_asym:12.1f}   McAllester RHS = {b_asym:.4f}")
        print(f"    KL  non-asymptotic A_d(k)   = {kl_nonasym:12.1f}   McAllester RHS = {b_non:.4f}")
        print(f"    KL overestimate factor      = {kl_asym / max(kl_nonasym,1e-12):8.1f}x")
        # Catoni (Thm 3.8) at optimal lambda, vs McAllester, on the non-asymptotic KL.
        kl = kl_nonasym + np.log(1 / delta)
        lams = np.linspace(0.01, 50, 5000)
        catoni = (1.0 / (1.0 - np.exp(-lams))) * (Rhat + kl / (lams * n))
        j = int(np.argmin(catoni))
        print(f"    Catoni RHS at lambda*={lams[j]:.2f}     = {catoni[j]:.4f}   "
              f"(vs McAllester {b_non:.4f})")
        if min(b_asym, b_non, catoni[j]) > 1:
            print("    [all RHS > 1: numerically vacuous, as the paper's limitations note expects]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join("results", "runs"))
    a = ap.parse_args()
    runs = load_runs(a.runs_dir)
    tasks = sorted({r["task"] for r in runs}); methods = sorted({r["method"] for r in runs})
    print(f"\nloaded {len(runs)} runs from {a.runs_dir} | tasks={tasks} methods={methods}\n")
    p1(runs); p3(runs); p4(runs)


if __name__ == "__main__":
    main()
