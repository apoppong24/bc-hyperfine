"""Quark–antiquark interaction potentials used in the thesis.

All three are spin-independent central potentials with the parameters of
Eichten & Quigg, Phys. Rev. D 49, 5845 (1994).  Units are natural
(hbar = c = 1), energies in GeV, distances in GeV^-1.

Each `Potential` bundles the functional form with the quark masses that go
with it, because the fitted parameters are only meaningful together with
those masses (this is why the Cornell masses differ from the log/Martin ones).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Potential:
    name: str
    V: Callable[[Array], Array]       # V(r) in GeV, r in GeV^-1
    m_c: float                        # charm mass used with this fit, GeV
    m_b: float                        # bottom mass used with this fit, GeV
    alpha_s: float                    # strong coupling used for hyperfine splitting
    label: str = ""
    # Optional one-loop running: alpha_s(mu) with mu set by the quark masses.
    # scale = None (fixed), "reduced" (mu = 2 m1 m2/(m1+m2)) or "geometric"
    # (mu = sqrt(m1 m2)); lambda_qcd in GeV, n_f = 4.
    alpha_s_scale: str | None = None
    lambda_qcd: float = 0.3
    # Hyperfine treatment for S waves: "perturbative" (first-order, E&Q) or
    # "smeared" (a Gaussian-smeared contact term added to the Hamiltonian and
    # solved self-consistently, as in Barnes–Godfrey–Swanson 2005).
    hyperfine: str = "perturbative"
    smear_sigma: float = 1.0          # GeV; width of the smeared delta function

    def __call__(self, r: Array) -> Array:
        return self.V(r)

    def mass(self, flavour: str) -> float:
        return {"c": self.m_c, "b": self.m_b}[flavour]

    def alpha_s_for(self, m1: float, m2: float) -> float:
        """Coupling to use in the spin-dependent terms for a Q1 Q̄2 meson."""
        if self.alpha_s_scale is None:
            return self.alpha_s
        mu = 2.0 * m1 * m2 / (m1 + m2) if self.alpha_s_scale == "reduced" else np.sqrt(m1 * m2)
        return 12.0 * np.pi / (25.0 * np.log(mu**2 / self.lambda_qcd**2))      # n_f = 4

    def contact_term(self, m1: float, m2: float, spin_spin: float):
        """Smeared hyperfine potential  V_hf(r) = (32π α_s / 9 m1 m2) <s1·s2> δ_σ(r),
        δ_σ(r) = (σ/√π)³ exp(−σ² r²), for use as `extra_V` in the solver."""
        a = self.alpha_s_for(m1, m2)
        pref = 32.0 * np.pi * a / (9.0 * m1 * m2) * spin_spin * (self.smear_sigma / np.sqrt(np.pi)) ** 3
        sig2 = self.smear_sigma**2
        return lambda r: pref * np.exp(-sig2 * r**2)


def _cornell(r: Array, kappa: float = 0.52, a: float = 2.34) -> Array:
    """Coulomb + linear:  V = -kappa/r + r/a^2  (thesis eq. 7)."""
    return -kappa / r + r / a**2


def _logarithmic(r: Array) -> Array:
    """Quigg–Rosner:  V = -0.6635 + 0.733 ln(r)  (thesis eq. 9)."""
    return -0.6635 + 0.733 * np.log(r)


def _martin(r: Array) -> Array:
    """Martin power law:  V = -8.064 + 6.898 r^0.1  (thesis eq. 11)."""
    return -8.064 + 6.898 * r**0.1


CORNELL = Potential("cornell", _cornell, m_c=1.84, m_b=5.18, alpha_s=0.31, label="Cornell")
LOGARITHMIC = Potential("log", _logarithmic, m_c=1.5, m_b=4.906, alpha_s=0.37, label="Logarithmic")
MARTIN = Potential("martin", _martin, m_c=1.8, m_b=5.174, alpha_s=0.43, label="Power-law (Martin)")

POTENTIALS = {p.name: p for p in (MARTIN, LOGARITHMIC, CORNELL)}
POTENTIALS["power"] = MARTIN
POTENTIALS["powerlaw"] = MARTIN
