"""E1 and M1 radiative transition widths.

Eichten & Quigg, PRD 49, 5845 (1994), Sec. III:

    Γ_E1(i→fγ) = (4 α <e_Q>²/27) k³ (2J_f+1) S_if |<f|r|i>|²
    Γ_M1(i→fγ) = (16 α/3) μ² k³ (2J_f+1) |<f|j0(kr/2)|i>|²        (S waves)

    <e_Q> = (m_b e_c − m_c e_b̄)/(m_b + m_c),   μ = (m_b e_c − m_c e_b̄)/(4 m_c m_b)

For a generic Q1 Q̄2 meson the same forms hold with (m1, e1) = quark and
(m2, e2) = antiquark charge, which for cc̄ reduce to <e_Q> = e_c and
μ = e_c/(2 m_c).  The E1 statistical factor is written in general as
S_if = 9·max(L_i,L_f)·{L_f J_f S; J_i L_i 1}²  (6-j symbol), which gives
S = 1 for ³S₁↔³P_J and S = 3 for ¹P₁↔¹S₀ as in E&Q.

Photon energy from recoil: k = (M_i² − M_f²)/(2 M_i).
Widths are returned in keV.
"""
from __future__ import annotations

from math import factorial, sqrt

import numpy as np

from .solver import RadialState

ALPHA_EM = 1.0 / 137.036
GEV_TO_KEV = 1e6
QUARK_CHARGE = {"c": 2.0 / 3.0, "b": -1.0 / 3.0}


def effective_charges(q1: str, q2: str, m1: float, m2: float) -> tuple[float, float]:
    """(<e_Q>, μ) for the meson q1 q̄2.  Antiquark charge is -e_q2."""
    e1, e2bar = QUARK_CHARGE[q1], -QUARK_CHARGE[q2]
    eQ = (m2 * e1 - m1 * e2bar) / (m1 + m2)
    mu = (m2 * e1 - m1 * e2bar) / (4.0 * m1 * m2)
    return eQ, mu


# ------------------------------------------------------------- 6-j symbol
def _delta(a, b, c):
    return (factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c)
            / factorial(a + b + c + 1))


def wigner_6j(j1, j2, j3, j4, j5, j6) -> float:
    """Racah formula for integer arguments."""
    for tri in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)):
        a, b, c = tri
        if c < abs(a - b) or c > a + b:
            return 0.0
    pre = sqrt(_delta(j1, j2, j3) * _delta(j1, j5, j6) * _delta(j4, j2, j6) * _delta(j4, j5, j3))
    s = 0.0
    tmin = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    tmax = min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4)
    for t in range(tmin, tmax + 1):
        s += ((-1) ** t * factorial(t + 1)
              / (factorial(t - j1 - j2 - j3) * factorial(t - j1 - j5 - j6)
                 * factorial(t - j4 - j2 - j6) * factorial(t - j4 - j5 - j3)
                 * factorial(j1 + j2 + j4 + j5 - t) * factorial(j2 + j3 + j5 + j6 - t)
                 * factorial(j3 + j1 + j6 + j4 - t)))
    return pre * s


def e1_statistical_factor(Li, Ji, Lf, Jf, S) -> float:
    return 9.0 * max(Li, Lf) * wigner_6j(Lf, Jf, S, Ji, Li, 1) ** 2


# ------------------------------------------------------------- overlaps
def _overlap(si: RadialState, sf: RadialState, f: np.ndarray) -> float:
    """<f| f(r) |i> on a common grid (states may have different r_max)."""
    n = min(si.u.size, sf.u.size)
    assert abs(si.h - sf.h) < 1e-12, "states must share the grid spacing"
    return float(np.sum(sf.u[:n] * si.u[:n] * f[:n]) * si.h)


def photon_energy(Mi: float, Mf: float) -> float:
    return (Mi**2 - Mf**2) / (2.0 * Mi)


def e1_width(si: RadialState, Ji: int, S: int, sf: RadialState, Jf: int,
             Mi: float, Mf: float, q1: str, q2: str) -> float:
    """Γ_E1 in keV for (radial state si, J_i) -> (sf, J_f) + γ, same spin S."""
    if Mi <= Mf or abs(si.l - sf.l) != 1:
        return 0.0
    eQ, _ = effective_charges(q1, q2, si.m1, si.m2)
    k = photon_energy(Mi, Mf)
    r_me = _overlap(si, sf, si.r)
    Sif = e1_statistical_factor(si.l, Ji, sf.l, Jf, S)
    return (4.0 * ALPHA_EM * eQ**2 / 27.0) * k**3 * (2 * Jf + 1) * Sif * r_me**2 * GEV_TO_KEV


def m1_width(si: RadialState, sf: RadialState, Jf: int, Mi: float, Mf: float,
             q1: str, q2: str) -> float:
    """Γ_M1 in keV between S-wave levels (³S₁ ↔ ¹S₀)."""
    if Mi <= Mf or si.l != 0 or sf.l != 0:
        return 0.0
    _, mu = effective_charges(q1, q2, si.m1, si.m2)
    k = photon_energy(Mi, Mf)
    j0 = np.sinc(k * si.r / 2.0 / np.pi)          # np.sinc(x) = sin(πx)/(πx)
    ov = _overlap(si, sf, j0)
    return (16.0 * ALPHA_EM / 3.0) * mu**2 * k**3 * (2 * Jf + 1) * ov**2 * GEV_TO_KEV


# ------------------------------------------------------------- tables
def _parse(label: str):
    lab = label.rstrip("'")
    if len(lab) == 3:
        return int(lab[0]), None, "SPDF".index(lab[1]), int(lab[2])
    return int(lab[0]), (int(lab[1]) - 1) // 2, "SPDF".index(lab[2]), int(lab[3])


def radiative_table(states: dict[str, RadialState], levels, q1: str, q2: str):
    """All E1 (ΔL = 1) and M1 (S-wave, spin flip) widths between the given
    FineLevel objects.  For spin-mixed states the E1 rate is the sum over the
    S = 1 and S = 0 channels weighted by the squared composition amplitudes
    |a_i^S a_f^S|² (exact when only one channel contributes, e.g. P ↔ S; an
    incoherent approximation for D ↔ P when both ends are mixed)."""
    rows = []
    for fi in levels:
        ni, _, Li, Ji = _parse(fi.label)
        si = states[f"{ni}{'SPDF'[Li]}"]
        for ff in levels:
            if ff.mass >= fi.mass:
                continue
            nf, _, Lf, Jf = _parse(ff.label)
            sf = states[f"{nf}{'SPDF'[Lf]}"]
            if abs(Li - Lf) == 1:
                w = 0.0
                for S, idx in ((1, 0), (0, 1)):
                    weight = (fi.composition[idx] * ff.composition[idx]) ** 2
                    if weight > 1e-12:
                        w += weight * e1_width(si, Ji, S, sf, Jf, fi.mass, ff.mass, q1, q2)
                kind = "E1"
            elif Li == 0 and Lf == 0 and fi.composition != ff.composition:
                w = m1_width(si, sf, Jf, fi.mass, ff.mass, q1, q2)
                kind = "M1"
            else:
                continue
            if w > 0:
                rows.append(dict(initial=fi.label, final=ff.label, type=kind,
                                 k_MeV=1000 * photon_energy(fi.mass, ff.mass), width_keV=w))
    return rows
