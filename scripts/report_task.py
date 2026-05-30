"""
Build the directional-update report for a SINGLE GLUE task from its sweep runs.

Reads results/runs/<task>_<method>_seed*.pkl (all methods, all seeds) and writes
results/report_<task>.pdf. Accuracy in the summary table is the mean over seeds;
the geometry plots use one representative seed (the angles are stable across
seeds, and using one seed keeps the complexity magnitudes correct).

Usage:  python scripts/report_task.py <task>     # e.g. sst2, mnli, qqp, qnli, mrpc, rte
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import make_report as MR  # noqa: E402


def main(task: str):
    paths = sorted(glob.glob(os.path.join("results", "runs", f"{task}_*.pkl")))
    if not paths:
        raise SystemExit(f"no runs for task '{task}' in results/runs/")
    runs = []
    for p in paths:
        with open(p, "rb") as f:
            runs.append(pickle.load(f))

    by_seed = defaultdict(dict)        # seed -> method -> run
    acc_by_method = defaultdict(list)  # method -> [acc over seeds]
    for r in runs:
        by_seed[r["seed"]][r["method"]] = r
        acc_by_method[r["method"]].append(r["accuracy"])

    # representative seed = the one with the most methods present
    rep = max(by_seed, key=lambda s: len(by_seed[s]))
    method_runs = by_seed[rep]
    seeds = sorted(by_seed)

    methods = {m: dict(accuracy=float(np.mean(acc_by_method[m])),   # mean over seeds
                       trainable_params=method_runs[m]["trainable_params"],
                       layers=method_runs[m]["layers"])
               for m in MR.ORDER if m in method_runs}
    res = {"cfg": {**next(iter(method_runs.values()))["cfg"], "task": task},
           "methods": methods}

    acc_str = ", ".join(f"{MR.LABEL[m]} {np.mean(acc_by_method[m])*100:.2f}"
                        f"±{np.std(acc_by_method[m])*100:.2f}%" for m in methods)
    out = os.path.join("results", f"report_{task}.pdf")
    MR.render_report(res, out,
                     title=f"Directional Updates: {task.upper()}  (seeds {seeds})")
    print(f"[report_task] wrote {out}")
    print(f"[report_task] {task}: accuracy mean±std over seeds {seeds} -> {acc_str}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/report_task.py <task>")
    main(sys.argv[1])
