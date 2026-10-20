"""Bound-state solver for the radial Schrödinger equation.

    U''(r) = 2 mu [ V(r) + l(l+1)/(2 mu r^2) - E ] U(r)        (thesis eq. 23)

Eigenvalues are located automatically by the node-counting bisection the
thesis describes in Sec. 3.3 (but which the notebooks did by hand): for a
trial energy E, integrate outward; if the wavefunction has more than n_r
nodes before r_max the energy is too high, otherwise too low.  Bisection on
that predicate converges to E_{n_r, l}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .numerov import count_nodes, numerov
from .potentials import Potential


def reduced_mass(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)


@dataclass
class RadialState:
    """A solved radial eigenstate.  `u` is the reduced radial function
    U = rR, normalised so that  ∫ U² dr = 1  (thesis eq. 26)."""
    potential: Potential
    m1: float
    m2: float
    n: int                 # principal label 1,2,3… (n_r = n - 1 radial nodes)
    l: int
    E: float               # eigenvalue, GeV
    r: np.ndarray
    u: np.ndarray
    h: float

    # ----- derived quantities -------------------------------------------
    @property
    def mu(self) -> float:
        return reduced_mass(self.m1, self.m2)

    @property
    def mass(self) -> float:
        """Spin-averaged meson mass  M = m1 + m2 + E  (thesis eq. 40)."""
        return self.m1 + self.m2 + self.E

    @property
    def R(self) -> np.ndarray:
        """Radial function R = U/r (undefined at r=0; returned as 0 there)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(self.r > 0, self.u / self.r, 0.0)
        R[0] = self.u[1] / self.h if self.l == 0 else 0.0   # limit r -> 0
        return R

    def R_at_origin_sq(self) -> float:
        """|R_nl^{(l)}(0)|^2  (thesis eq. 27).

        Near the origin U ≈ c r^{l+1}, so R = U/r ≈ c r^l and
        d^l R / dr^l |_0 = l! c.  For l = 0 this is simply U'(0)^2, which is
        what the notebooks compute as 1/A."""
        c = self.u[1] / self.h ** (self.l + 1)
        return (math.factorial(self.l) * c) ** 2

    @property
    def nodes(self) -> int:
        """Radial nodes, ignoring the numerically-noisy tail."""
        big = np.abs(self.u) > 1e-6 * np.abs(self.u).max()
        return count_nodes(self.u[big])

    def label(self) -> str:
        return f"{self.n}{'SPDF'[self.l]}"

    # ----- relativistic size and decay constant ----------------------------
    def mean_p2(self) -> float:
        """<p²> in GeV² from the virial-free identity <p²>/2μ = E − <V_eff>."""
        veff = getattr(self, "_veff", None)
        if veff is None:
            veff = _effective_potential(self.potential, self.r, self.l, self.mu)
        mean_V = float(np.sum(self.u[1:] ** 2 * veff[1:]) * self.h)
        return 2.0 * self.mu * (self.E - mean_V)

    def v2(self) -> tuple[float, float]:
        """(<v₁²>, <v₂²>) = <p²>/m_i², the non-relativistic expansion parameters."""
        p2 = self.mean_p2()
        return p2 / self.m1**2, p2 / self.m2**2

    def decay_constant(self, alpha_s: float | None = None, qcd_correction: bool = True) -> float:
        """Pseudoscalar/vector decay constant from van Royen–Weisskopf,
            f² = 3|R(0)|²/(π M) · C,
        with the one-loop correction C = 1 − (α_s/π)[2 − (m₁−m₂)/(m₁+m₂) ln(m₁/m₂)]
        (Braaten & Fleming 1995; reduces to 1 − 16α_s/(3π)·… for equal masses
        only in the vector case—here we use the pseudoscalar/vector-averaged
        form appropriate to the spin-averaged wavefunction).  S waves only."""
        if self.l != 0:
            raise ValueError("decay constant needs an S-wave state")
        M = self.mass
        f2 = 3.0 * self.R_at_origin_sq() / (np.pi * M)
        if qcd_correction:
            a = self.potential.alpha_s_for(self.m1, self.m2) if alpha_s is None else alpha_s
            m1, m2 = max(self.m1, self.m2), min(self.m1, self.m2)
            C = 1.0 - (a / np.pi) * (2.0 - (m1 - m2) / (m1 + m2) * np.log(m1 / m2))
            f2 *= C
        return float(np.sqrt(max(f2, 0.0)))


def _effective_potential(V: Potential, r: np.ndarray, l: int, mu: float, extra_V=None) -> np.ndarray:
    """V_eff on the grid (thesis eq. 24).  r[0] = 0 is never used because
    U[0] = 0 multiplies it, so we silence the singularity there.
    `extra_V(r)` is an optional additional term (e.g. a smeared spin-spin
    contact interaction) added to V."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = V(r) + l * (l + 1) / (2.0 * mu * r**2)
        if extra_V is not None:
            v = v + extra_V(r)
    v[0] = 0.0
    return v


def _shoot(veff: np.ndarray, mu: float, E: float, h: float, l: int) -> np.ndarray:
    g = -2.0 * mu * (veff - E)                   # thesis eq. (36)
    # Boundary conditions (thesis eq. 37): U(0)=0, U(h)=h^{l+1}; the scale is
    # arbitrary and fixed later by normalisation.
    return numerov(g, h, 0.0, h ** (l + 1))


def solve(
    potential: Potential,
    m1: float,
    m2: float,
    n: int = 1,
    l: int = 0,
    r_max: float = 10.0,
    h: float = 1e-3,
    E_bracket: tuple[float, float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    extra_V=None,
) -> RadialState:
    """Find the bound state with n-1 radial nodes and angular momentum l.

    Parameters
    ----------
    potential : one of `quarkonium.potentials.*` (or your own `Potential`)
    m1, m2    : quark masses in GeV (use `potential.m_c`, `potential.m_b`)
    n, l      : spectroscopic labels (1S → n=1,l=0; 2P → n=2,l=1)
    r_max     : integration range in GeV^-1 – must lie well inside the
                classically forbidden region of the state wanted
    h         : grid spacing in GeV^-1
    E_bracket : optional (E_low, E_high); default spans the potential well
    extra_V   : optional callable r -> V_extra(r) added to the potential
                (used for the self-consistent smeared hyperfine term)
    """
    if n < 1 or l < 0:
        raise ValueError("need n >= 1 and l >= 0")
    n_r = n - 1
    mu = reduced_mass(m1, m2)
    r = np.arange(0.0, r_max + 0.5 * h, h)
    veff = _effective_potential(potential, r, l, mu, extra_V)

    if E_bracket is None:
        E_lo = float(np.min(veff[1:]))
        E_hi = float(veff[-1])
    else:
        E_lo, E_hi = E_bracket

    # Sanity: the bracket must straddle the eigenvalue.
    if count_nodes(_shoot(veff, mu, E_lo, h, l)) > n_r:
        raise ValueError(f"E_low={E_lo} is above the {n}{'SPDF'[l]} level; lower it.")
    if count_nodes(_shoot(veff, mu, E_hi, h, l)) <= n_r:
        raise ValueError(
            f"E_high={E_hi} is below the {n}{'SPDF'[l]} level (or r_max={r_max} "
            "is too small for this state to be bound on the grid); raise them.")

    for _ in range(max_iter):
        E = 0.5 * (E_lo + E_hi)
        u = _shoot(veff, mu, E, h, l)
        if count_nodes(u) > n_r:
            E_hi = E
        else:
            E_lo = E
        if E_hi - E_lo < tol:
            break

    E = 0.5 * (E_lo + E_hi)
    u = _shoot(veff, mu, E, h, l)

    # Truncate the exponentially growing tail (numerical, not physical) before
    # normalising: keep everything up to the last turning point of |u| that
    # lies in the forbidden region, i.e. where |u| is smallest beyond the
    # outermost node.
    tail = _tail_cut(u, r, veff, E)
    u = u.copy()
    u[tail:] = 0.0
    norm = np.sqrt(np.sum(u * u) * h)            # thesis eq. (26), as in the notebooks
    u /= norm
    st = RadialState(potential, m1, m2, n, l, E, r, u, h)
    st._veff = veff
    return st


def _tail_cut(u: np.ndarray, r: np.ndarray, veff: np.ndarray, E: float) -> int:
    """Index beyond which the integrated U is dominated by the growing
    exponential and should be dropped.  We look for the minimum of |U| in
    the classically forbidden region beyond the outermost classical turning
    point."""
    mask = veff > E
    mask[0] = False
    # start of the outermost forbidden block (last False->True transition)
    rises = np.where(mask[1:] & ~mask[:-1])[0] + 1
    if rises.size == 0:
        return u.size
    start = int(rises[-1])
    seg = np.abs(u[start:])
    k = int(np.argmin(seg))
    return start + k + 1
