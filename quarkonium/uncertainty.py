"""Propagate fit-parameter covariance to predicted masses (linear error
propagation with a numerical Jacobian): σ²(M) = J Σ Jᵀ.

This is the *parametric* (statistical) uncertainty of a prediction given
the fitted data; the *model* uncertainty is the spread across potential
families (see refit_pdg.py / scan_exponent.py)."""
from __future__ import annotations

import numpy as np

from .fit import FAMILIES, predict


def predict_with_errors(family: str, x, cov, pair: str, labels, h: float = 2e-3, rel_step: float = 2e-3):
    """Return {label: (mass, sigma)} in GeV for the given family/parameters."""
    make = FAMILIES[family]["make"]
    x = np.asarray(x, float)
    cov = np.asarray(cov, float)
    central = predict(make(*x), pair, labels, h=h)
    J = np.zeros((len(labels), len(x)))
    for j in range(len(x)):
        dx = rel_step * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += dx
        xm[j] -= dx
        pp = predict(make(*xp), pair, labels, h=h)
        pm = predict(make(*xm), pair, labels, h=h)
        J[:, j] = [(pp[l] - pm[l]) / (2 * dx) for l in labels]
    var = np.einsum("ij,jk,ik->i", J, cov, J)
    return {l: (central[l], float(np.sqrt(max(v, 0.0)))) for l, v in zip(labels, var)}
