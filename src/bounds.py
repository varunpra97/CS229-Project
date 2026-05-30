"""
Non-asymptotic vMF quantities and PAC-Bayes bound values.

A_d(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)   (mean resultant length)

This is the correction the asymptotic analysis drops (it sets A_d(kappa) ~ 1).
Prediction P4 is just: show A_d(kappa) << 1 for d = 768 and moderate kappa, so
the non-asymptotic bound (Theorem 3.4) is materially tighter. All pure CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive  # exponentially scaled I_nu


def _log_iv(nu: float, kappa: float) -> float:
    """log I_nu(kappa) computed stably.

    For large order nu (here nu ~ d/2 ~ 384) and moderate kappa, I_nu(kappa) is
    astronomically small and ive(nu, kappa) underflows to exactly 0.0 in double
    precision. We fall back to the uniform asymptotic (Olver) expansion in log
    space, which is accurate precisely in the large-order regime where ive fails.
    """
    val = ive(nu, kappa)
    if val > 0.0:
        # ive(nu, k) = I_nu(k) * exp(-k)  =>  log I_nu = log(ive) + k
        return float(np.log(val) + kappa)
    # Large-order uniform asymptotic: with z = kappa/nu, eta = sqrt(1+z^2) + log(z/(1+sqrt(1+z^2)))
    #   I_nu(nu z) ~ exp(nu*eta) / (sqrt(2*pi*nu) * (1+z^2)^{1/4})
    z = kappa / nu
    t = np.sqrt(1.0 + z * z)
    eta = t + np.log(z / (1.0 + t))
    return float(nu * eta - 0.5 * np.log(2.0 * np.pi * nu) - 0.25 * np.log(1.0 + z * z))


def Ad(kappa: float, d: int) -> float:
    """Mean resultant length A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa).

    Computed as exp(log I_{d/2} - log I_{d/2-1}) to avoid underflow at large d.
    """
    if kappa <= 0:
        return 0.0
    log_ratio = _log_iv(d / 2.0, kappa) - _log_iv(d / 2.0 - 1.0, kappa)
    return float(np.exp(log_ratio))


def overestimate_ratio(kappa: float, d: int) -> float:
    """How much the asymptotic bound (A_d ~ 1) overestimates the true KL factor.

    Asymptotic uses factor 1; non-asymptotic uses A_d(kappa). Ratio = 1 / A_d.
    Large ratio => asymptotic is loose => P4 confirmed.
    """
    a = Ad(kappa, d)
    return float("inf") if a == 0.0 else 1.0 / a


def mcallester_bound(emp_risk: float, kl: float, n: int, delta: float = 0.05) -> float:
    """McAllester bound RHS (Theorem 3.1): R_hat + sqrt((KL + ln(2 sqrt(n)/delta)) / 2n)."""
    return emp_risk + np.sqrt((kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n))


if __name__ == "__main__":
    # Quick P4 sanity table for RoBERTa-base hidden dim d = 768.
    d = 768
    print(f"{'kappa':>8} {'A_d(kappa)':>14} {'overestimate':>14}")
    for k in (1, 5, 10, 50, 100):
        print(f"{k:>8} {Ad(k, d):>14.6e} {overestimate_ratio(k, d):>14.2f}")
