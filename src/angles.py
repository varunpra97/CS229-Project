"""
Angle extraction and directional complexity estimators.

This is the core of the experiment. After fine-tuning, we compare each column of
the *merged* fine-tuned weight (W0 + dW) against the corresponding pretrained
column W0, measuring the angle between them. These angles feed the PAC-Bayes
directional complexity terms from the paper:

    theta_j      = arccos( <w0_j, what_j> / (||w0_j|| ||what_j||) )   (per column)
    Theta_global = arccos( <vec(W0), vec(What)> / (||.|| ||.||) )     (flattened)

    C_DoRA_dir = kappa * sum_j (1 - cos theta_j)        (asymptotic form)
    C_MAP_dir  = kappa * (1 - cos Theta_global)

The non-asymptotic correction factor A_d(kappa) lives in bounds.py.
"""

from __future__ import annotations

import numpy as np


def column_angles(W0: np.ndarray, What: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-column angles (radians) between pretrained and fine-tuned weight columns.

    Convention: weight matrices are (d_out, d_in); columns are indexed by d_in,
    matching the paper's column-wise decomposition w_j in R^{d_out}.

    Returns an array of shape (d_in,).
    """
    assert W0.shape == What.shape, f"shape mismatch: {W0.shape} vs {What.shape}"
    # Normalize each column.
    n0 = np.linalg.norm(W0, axis=0)            # (d_in,)
    nh = np.linalg.norm(What, axis=0)
    dots = np.sum(W0 * What, axis=0)           # (d_in,)
    cos = dots / (n0 * nh + eps)
    cos = np.clip(cos, -1.0, 1.0)              # guard against fp drift
    return np.arccos(cos)


def global_angle(W0: np.ndarray, What: np.ndarray, eps: float = 1e-12) -> float:
    """Global angle (radians) between the flattened weight matrices (MAP view)."""
    v0 = W0.reshape(-1)
    vh = What.reshape(-1)
    cos = float(np.dot(v0, vh) / (np.linalg.norm(v0) * np.linalg.norm(vh) + eps))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


def c_dir_dora(thetas: np.ndarray, kappa: float, ad_factor: float = 1.0) -> float:
    """DoRA directional complexity: kappa * A_d(kappa) * sum_j (1 - cos theta_j).

    ad_factor = A_d(kappa) for the non-asymptotic bound (Theorem 3.4); pass 1.0
    for the asymptotic approximation (Section 5.5 / P4 compares the two).
    """
    return float(kappa * ad_factor * np.sum(1.0 - np.cos(thetas)))


def c_dir_map(theta_global: float, kappa: float, ak_factor: float = 1.0) -> float:
    """MAP directional complexity: kappa * A_k(kappa) * (1 - cos Theta_global)."""
    return float(kappa * ak_factor * (1.0 - np.cos(theta_global)))


def effective_sparsity(thetas: np.ndarray, tau: float = 0.01) -> int:
    """s = |{ j : theta_j > tau }|, with tau in radians (paper uses 0.01)."""
    return int(np.sum(thetas > tau))


def summarize_layer(W0: np.ndarray, What: np.ndarray, kappa: float,
                    tau: float = 0.01) -> dict:
    """Compute all per-layer geometric quantities for one weight matrix."""
    thetas = column_angles(W0, What)
    tg = global_angle(W0, What)
    s = effective_sparsity(thetas, tau)
    return {
        "n_cols": int(W0.shape[1]),
        "theta_mean": float(np.mean(thetas)),
        "theta_var": float(np.var(thetas)),        # used by P3 (attn vs MLP)
        "theta_max": float(np.max(thetas)),
        "theta_global": tg,
        "s": s,
        "s_over_d": float(s / W0.shape[1]),
        "c_dora_asym": c_dir_dora(thetas, kappa, ad_factor=1.0),
        "c_map_asym": c_dir_map(tg, kappa, ak_factor=1.0),
        # raw angles kept so bounds.py / P4 can recompute with A_d(kappa) factors
        "_thetas": thetas,
    }
