"""
Evaluate the actual PAC-Bayes bound VALUE per run, closing the loop from measured
directional complexity to the generalization guarantee of Theorem 3.1 / 3.4.

For every results/runs/*.pkl we:
  1. Sum the directional KL across all 72 target layers, using the per-column angles
     theta_j that were measured between pretrained and fine-tuned weights:
        KL_dir = sum_layers sum_j  kappa * A_{d_out}(kappa) * (1 - cos theta_j)
     in both the ASYMPTOTIC form (A_d = 1, what the report plots show) and the
     NON-ASYMPTOTIC form (true A_d(kappa) from bounds.py) -- this is Prediction P4.
  2. Plug KL into McAllester's bound (Thm 3.1):
        bound = R_hat + sqrt( (KL + ln(2 sqrt(n)/delta)) / (2n) )
     and report whether it is NON-VACUOUS (< 1).

Caveats (printed in the output):
  * KL here is DIRECTIONAL ONLY; the magnitude term C_mag is not measured, so the
    reported KL is a lower bound on the true KL and the bound is correspondingly
    optimistic.
  * Training error R_hat is not logged by the current pipeline. We report the
    COMPLEXITY TERM (the sqrt) on its own -- which alone decides vacuity -- and also
    bound_proxy = test_error + complexity_term as a reference point.

Run:  python scripts/eval_bounds.py
"""
from __future__ import annotations

import csv
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bounds import Ad, mcallester_bound  # noqa: E402

RUNS_DIR = os.path.join("results", "runs")
DELTA = 0.05
# GLUE train-set sizes (n) -- the sample size in the bound.
N_TRAIN = {"sst2": 67349, "qnli": 104743, "mnli": 392702, "rte": 2490,
           "qqp": 363846, "mrpc": 3668}


def d_out_of(layer_name: str, layer_type: str, n_cols: int) -> int:
    """Ambient dimension of each column vector (the vMF sphere dim S^{d_out-1}).

    RoBERTa-base target modules:
      attention.self.{query,key,value}, attention.output.dense : (768,768) -> d_out 768
      intermediate.dense  : (3072,768)  -> n_cols 768,  d_out 3072
      FFN output.dense     : (768,3072) -> n_cols 3072, d_out 768
    """
    if layer_type == "attn":
        return 768
    # mlp: distinguish the two FFN matrices by their column count
    return 3072 if n_cols == 768 else 768


def kl_directional(run, kappa: float, nonasym: bool) -> float:
    """Total directional KL over all layers, asymptotic (A_d=1) or non-asymptotic."""
    total = 0.0
    for name, L in run["layers"].items():
        thetas = np.asarray(L["theta"], dtype=np.float64)
        one_minus_cos = np.sum(1.0 - np.cos(thetas))
        if nonasym:
            d_out = d_out_of(name, L["layer_type"], len(thetas))
            factor = kappa * Ad(kappa, d_out)
        else:
            factor = kappa
        total += factor * one_minus_cos
    return float(total)


def complexity_term(kl: float, n: int) -> float:
    """The sqrt( (KL + ln(2 sqrt(n)/delta)) / 2n ) part of McAllester's RHS."""
    return float(np.sqrt((kl + np.log(2 * np.sqrt(n) / DELTA)) / (2 * n)))


def main():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.pkl"))):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    if not runs:
        raise SystemExit(f"no runs in {RUNS_DIR}")

    kappa = runs[0]["cfg"].get("kappa", 10.0)
    out = os.path.join("results", "bound_values.csv")
    fields = ["task", "method", "seed", "n", "test_error",
              "KL_asym", "KL_nonasym", "term_asym", "term_nonasym",
              "bound_proxy_asym", "bound_proxy_nonasym",
              "nonvacuous_asym", "nonvacuous_nonasym", "overestimate_ratio"]
    rows = []
    for r in sorted(runs, key=lambda r: (r["task"], r["method"], r["seed"])):
        n = N_TRAIN[r["task"]]
        terr = 1.0 - r["accuracy"]
        kl_a = kl_directional(r, kappa, nonasym=False)
        kl_na = kl_directional(r, kappa, nonasym=True)
        term_a = complexity_term(kl_a, n)
        term_na = complexity_term(kl_na, n)
        bp_a = terr + term_a
        bp_na = terr + term_na
        rows.append(dict(
            task=r["task"], method=r["method"], seed=r["seed"], n=n,
            test_error=round(terr, 4),
            KL_asym=round(kl_a, 2), KL_nonasym=round(kl_na, 3),
            term_asym=round(term_a, 4), term_nonasym=round(term_na, 4),
            bound_proxy_asym=round(bp_a, 4), bound_proxy_nonasym=round(bp_na, 4),
            nonvacuous_asym=bool(bp_a < 1.0), nonvacuous_nonasym=bool(bp_na < 1.0),
            overestimate_ratio=round(kl_a / kl_na, 2) if kl_na > 0 else float("inf"),
        ))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- console summary ----
    print(f"kappa={kappa}, delta={DELTA}.  KL = DIRECTIONAL ONLY (C_mag omitted).")
    print(f"bound_proxy = test_error + complexity_term  (train error R_hat not logged).\n")
    hdr = f"{'task':5}{'meth':5}{'sd':>3}{'n':>8}{'KL_asym':>10}{'KL_na':>9}{'term_a':>8}{'term_na':>9}{'bnd_na':>8}{'nonvac':>8}"
    print(hdr); print("-" * len(hdr))
    # show seed-0 rows for readability, plus a per-(task,method) note
    for row in rows:
        if row["seed"] != 0:
            continue
        print(f"{row['task']:5}{row['method']:5}{row['seed']:>3}{row['n']:>8}"
              f"{row['KL_asym']:>10.1f}{row['KL_nonasym']:>9.2f}{row['term_asym']:>8.3f}"
              f"{row['term_nonasym']:>9.4f}{row['bound_proxy_nonasym']:>8.3f}"
              f"{'YES' if row['nonvacuous_nonasym'] else 'no':>8}")

    # P4: A_d(kappa) overestimate for d=768 and d=3072
    print(f"\nP4 -- asymptotic overestimates KL by factor 1/A_d(kappa):")
    for d in (768, 3072):
        a = Ad(kappa, d)
        print(f"  d={d:5}: A_d({kappa})={a:.4e}  ->  overestimate x{1.0/a:,.1f}")
    nv_na = sum(r["nonvacuous_nonasym"] for r in rows)
    nv_a = sum(r["nonvacuous_asym"] for r in rows)
    print(f"\nNon-vacuous (bound_proxy < 1): {nv_na}/{len(rows)} non-asym, {nv_a}/{len(rows)} asym.")
    print(f"[wrote {out}]")


if __name__ == "__main__":
    main()
