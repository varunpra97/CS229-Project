"""
Prediction P1: does the directional complexity C_dir predict test error?

Pools results/results.csv across runs and computes the Spearman rank correlation
between the complexity estimator and test error. With only a handful of runs the
result is a trend, not the 108-point permutation test of the paper -- report n
honestly in the writeup.

Usage:  python scripts/analyze_p1.py
"""

from __future__ import annotations

import sys
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np


def main(path: str = "results/results.csv"):
    df = pd.read_csv(path)
    # Aggregate to one (complexity, error) pair per run x layer_type, as in the paper.
    grp = (df.groupby(["method", "task", "seed", "layer_type"])
             .agg(c_dir=("c_dora_nonasym", "sum"),
                  test_error=("test_error", "mean"))
             .reset_index())
    print(f"pooled points: {len(grp)}")
    if len(grp) < 3:
        print("need >= 3 points for a correlation; run more configs first.")
        return
    rho_s, p_s = spearmanr(grp["c_dir"], grp["test_error"])
    rho_p, p_p = pearsonr(np.log(grp["c_dir"] + 1e-12), grp["test_error"])
    print(f"Spearman rho (C_dir vs error):      {rho_s:+.3f}  (p={p_s:.3g})")
    print(f"Pearson  rho (ln C_dir vs error):   {rho_p:+.3f}  (p={p_p:.3g})")
    print("\nPrediction P1 target: rho_s > 0.5, p < 0.01 (paper uses 108 points).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")
