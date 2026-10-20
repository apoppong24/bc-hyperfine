"""Quantities derived from solved states: hyperfine splitting, spin
averages, singlet/triplet masses, level splittings and the B_c mass
formula used in the thesis."""
from __future__ import annotations

from .solver import RadialState


def hyperfine_splitting(state: RadialState, alpha_s: float | None = None) -> float:
    """M(³S₁) - M(¹S₀) = 8 α_s |R(0)|² / (9 m_i m_j)   (thesis eq. 41)."""
    if state.l != 0:
        raise ValueError("hyperfine formula (41) applies to S states only")
    a = state.potential.alpha_s_for(state.m1, state.m2) if alpha_s is None else alpha_s
    return 8.0 * a * state.R_at_origin_sq() / (9.0 * state.m1 * state.m2)


def spin_average_S(m_singlet: float, m_triplet: float) -> float:
    """M(S̄) = (M(¹S₀) + 3 M(³S₁)) / 4   (thesis eq. 38)."""
    return (m_singlet + 3.0 * m_triplet) / 4.0


def spin_average_P(m_p0: float, m_p1: float, m_p2: float) -> float:
    """M(χ̄) = (M(³P₀) + 3 M(³P₁) + 5 M(³P₂)) / 9   (thesis eq. 39)."""
    return (m_p0 + 3.0 * m_p1 + 5.0 * m_p2) / 9.0


def singlet_triplet(m_spin_avg: float, delta_hyp: float) -> tuple[float, float]:
    """Invert eq. (38)/(43): given the spin average and the hyperfine
    splitting return (M(¹S₀), M(³S₁))."""
    triplet = m_spin_avg + 0.25 * delta_hyp
    singlet = triplet - delta_hyp
    return singlet, triplet


def bc_mass(E_bc: float, E_bb: float, E_cc: float, M_bb: float, M_cc: float) -> float:
    """Thesis eq. (42):

        M_Bc = [E_Bc - (E_bb + E_cc)/2] + (M_bb + M_cc)/2

    NOTE: if M_bb and M_cc are the *model* masses (2m_b+E_bb, 2m_c+E_cc) this
    collapses algebraically to m_b + m_c + E_Bc, i.e. plain eq. (40), and the
    quark masses do not cancel.  The formula only removes quark-mass
    dependence when M_bb and M_cc are the *experimental* spin-averaged 1S
    masses.  Both uses are supported; pass whichever you intend.
    """
    return (E_bc - 0.5 * (E_bb + E_cc)) + 0.5 * (M_bb + M_cc)


def splittings(masses: dict[str, float]) -> dict[str, float]:
    """Level differences reported in thesis Tables 7, 8, 15."""
    m = masses
    out = {}
    for a, b in (("3S", "1S"), ("2S", "1S"), ("1P", "1S"), ("2P", "1S"), ("2P", "1P")):
        if a in m and b in m:
            out[f"{a}-{b}"] = m[a] - m[b]
    return out
