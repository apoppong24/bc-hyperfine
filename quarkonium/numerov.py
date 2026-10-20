"""Numerov integration of  U''(r) = -g(r) U(r)  on a uniform grid.

This is thesis eq. (35) with s(r) = 0:

    U_{i+1} (1 + h^2 g_{i+1}/12)
        = 2 U_i (1 - 5 h^2 g_i /12) - U_{i-1} (1 + h^2 g_{i-1}/12)

The recursion is a plain Python loop (it is inherently sequential); with
h = 1e-3 a 10 GeV^-1 domain is 10k points and runs in a few ms.
"""
from __future__ import annotations

import numpy as np


def numerov(g: np.ndarray, h: float, u0: float, u1: float) -> np.ndarray:
    """Integrate outward from (u0, u1) given g(r) sampled on the grid."""
    n = g.size
    u = np.empty(n)
    u[0], u[1] = u0, u1
    c = h * h / 12.0
    f = 1.0 + c * g                     # (1 + h^2 g /12) on every point
    two_minus = 12.0 - 10.0 * f         # 2(1 - 5h^2 g/12) = 12 - 10 f
    for i in range(1, n - 1):
        u[i + 1] = (two_minus[i] * u[i] - f[i - 1] * u[i - 1]) / f[i + 1]
        if abs(u[i + 1]) > 1e200:
            # Classically-forbidden blow-up for a non-eigen trial energy.
            # Freeze the tail at its current sign so node counting stays sane.
            u[i + 2:] = u[i + 1]
            break
    return u


def count_nodes(u: np.ndarray, skip: int = 1) -> int:
    """Number of sign changes of u, ignoring the first `skip` points
    (u[0] = 0 by construction) and the leading-zero region."""
    s = np.sign(u[skip:])
    s = s[s != 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))
