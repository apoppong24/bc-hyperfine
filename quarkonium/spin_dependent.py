"""First-order spin-dependent mass shifts (fine and hyperfine structure).

Follows Eichten & Quigg, Phys. Rev. D 49, 5845 (1994), eqs. (2.15)–(2.25):

    Δ = T1 + T2 + T3 + T4
    T1 = <L·s_i>/(2 m_i²) T̃1 + <L·s_j>/(2 m_j²) T̃1      (scalar-exchange spin–orbit)
    T2 = (<L·s_i> + <L·s_j>)/(m_i m_j) T̃2                 (vector-exchange spin–orbit)
    T3 = <s_i·s_j>/(m_i m_j) T̃3                           (hyperfine)
    T4 = <S_ij>/(m_i m_j) T̃4                              (tensor)

with radial coefficients
    T̃1 = -<(1/r) dV/dr> + 2 T̃2,   T̃2 = (4 α_s/3) <r⁻³>,
    T̃3 = (32π α_s/9) |Ψ(0)|² = (8 α_s/9) |R(0)|²,   T̃4 = (α_s/3) <r⁻³>.

The potential V is the full spin-independent potential of the fit; the
Coulombic one-gluon-exchange vector part is added through α_s.  α_s per
potential is the value fixed by the charmonium 1S hyperfine splitting
(stored on each `Potential`).

For equal masses the states are pure ²ˢ⁺¹L_J.  For unequal masses the two
J = L states (³L_L and ¹L_L) mix; we build the 2×2 matrix in the LS basis
and diagonalise, which is equivalent to E&Q's jj-coupling matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import RadialState


# ---------------------------------------------------------------- radial
def expectation(state: RadialState, f: np.ndarray) -> float:
    """<f> = ∫ U² f dr on the grid, skipping r = 0 (U(0) = 0 anyway)."""
    u, h = state.u[1:], state.h
    return float(np.sum(u * u * f[1:]) * h)


def _dV_dr(state: RadialState) -> np.ndarray:
    r = state.r
    dr = 1e-4 * state.h
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (state.potential(r + dr) - state.potential(r - dr)) / (2 * dr)
    d[0] = 0.0
    return d


@dataclass
class RadialCoefficients:
    T1: float
    T2: float
    T3: float
    T4: float


def radial_coefficients(state: RadialState, alpha_s: float | None = None) -> RadialCoefficients:
    a = state.potential.alpha_s_for(state.m1, state.m2) if alpha_s is None else alpha_s
    r = state.r
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r3 = np.where(r > 0, r**-3.0, 0.0)
        dV_over_r = np.where(r > 0, _dV_dr(state) / r, 0.0)
    if state.l == 0:
        # <r⁻³> and <V'/r> diverge for S waves; they multiply <L·S> = <S_ij> = 0
        T2 = T4 = 0.0
        T1 = 0.0
    else:
        r3 = expectation(state, inv_r3)
        T2 = 4.0 * a / 3.0 * r3
        T4 = a / 3.0 * r3
        T1 = -expectation(state, dV_over_r) + 2.0 * T2
    # T̃3 ∝ |Ψ(0)|², which is non-zero only for S waves (contact term)
    T3 = 8.0 * a / 9.0 * state.R_at_origin_sq() if state.l == 0 else 0.0
    return RadialCoefficients(T1, T2, T3, T4)


# --------------------------------------------------------------- angular
def LS(L: int, S: int, J: int) -> float:
    return 0.5 * (J * (J + 1) - L * (L + 1) - S * (S + 1))


def tensor(L: int, S: int, J: int) -> float:
    """<S_12> for S_12 = 4[3(s1·n)(s2·n) - s1·s2]; zero for S = 0."""
    if S == 0 or L == 0:
        return 0.0
    if J == L + 1:
        return -2.0 * L / (2 * L + 3)
    if J == L:
        return 2.0
    return -2.0 * (L + 1) / (2 * L - 1)          # J = L - 1


def spin_spin(S: int) -> float:
    return 0.5 * S * (S + 1) - 0.75


# --------------------------------------------------------------- shifts
@dataclass
class FineLevel:
    label: str            # e.g. "3P2", "1P1", "1+" / "1+'" for mixed J=L states
    J: int
    shift: float          # GeV
    mass: float           # GeV, spin-averaged mass + shift
    mixing_angle: float | None = None   # radians, J=L unequal-mass case
    # amplitudes of the (S=1, S=0) components; pure states are (1,0) or (0,1)
    composition: tuple[float, float] = (1.0, 0.0)


def fine_structure(state: RadialState, alpha_s: float | None = None) -> list[FineLevel]:
    """All spin states built on one spin-independent radial state."""
    c = radial_coefficients(state, alpha_s)
    m1, m2, L = state.m1, state.m2, state.l
    M0 = state.mass
    # coefficients of L·s1, L·s2, s1·s2, S12
    a1 = c.T1 / (2 * m1**2) + c.T2 / (m1 * m2)
    a2 = c.T1 / (2 * m2**2) + c.T2 / (m1 * m2)
    b = c.T3 / (m1 * m2)
    d = c.T4 / (m1 * m2)
    n = state.n
    Lsym = "SPDFG"[L]
    out: list[FineLevel] = []

    if L == 0:
        for S, J in ((0, 0), (1, 1)):
            sh = b * spin_spin(S)
            out.append(FineLevel(f"{n}{2*S+1}{Lsym}{J}", J, sh, M0 + sh,
                                 composition=(1.0, 0.0) if S == 1 else (0.0, 1.0)))
        return out

    # S = 1, J = L ± 1 : pure states
    for J in (L - 1, L + 1):
        ls = LS(L, 1, J)
        sh = (a1 + a2) * 0.5 * ls + b * spin_spin(1) + d * tensor(L, 1, J)
        out.append(FineLevel(f"{n}3{Lsym}{J}", J, sh, M0 + sh))

    # J = L : 2×2 in the basis (³L_L, ¹L_L)
    H33 = (a1 + a2) * 0.5 * LS(L, 1, L) + b * spin_spin(1) + d * tensor(L, 1, L)
    H11 = b * spin_spin(0)
    H31 = (a1 - a2) * 0.5 * np.sqrt(L * (L + 1))
    if abs(H31) < 1e-12:
        out.append(FineLevel(f"{n}3{Lsym}{L}", L, H33, M0 + H33, composition=(1.0, 0.0)))
        out.append(FineLevel(f"{n}1{Lsym}{L}", L, H11, M0 + H11, composition=(0.0, 1.0)))
    else:
        H = np.array([[H33, H31], [H31, H11]])
        w, v = np.linalg.eigh(H)
        theta = float(np.arctan2(v[1, 1], v[0, 1]))   # mixing angle of the upper state
        # E&Q label the two mixed 1⁺ states n(1⁺) (lower) and n(1⁺') (upper);
        # columns of v are (triplet, singlet) amplitudes of each eigenstate
        out.append(FineLevel(f"{n}{Lsym}{L}", L, float(w[0]), M0 + float(w[0]), theta,
                             composition=(float(v[0, 0]), float(v[1, 0]))))
        out.append(FineLevel(f"{n}{Lsym}{L}'", L, float(w[1]), M0 + float(w[1]), theta,
                             composition=(float(v[0, 1]), float(v[1, 1]))))

    return sorted(out, key=lambda f: f.mass)


def full_spectrum(states: dict[str, RadialState], alpha_s: float | None = None) -> list[FineLevel]:
    """Fine-structure levels for every radial state, sorted by mass."""
    levels = []
    for st in states.values():
        levels += fine_structure(st, alpha_s)
    return sorted(levels, key=lambda f: f.mass)
