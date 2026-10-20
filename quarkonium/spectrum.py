"""Convenience: solve the five levels studied in the thesis for a given
flavour pair, with sensible r_max per level."""
from __future__ import annotations

from .potentials import Potential
from .solver import RadialState, solve

# (label, n, l)
LEVELS = [("1S", 1, 0), ("2S", 2, 0), ("3S", 3, 0), ("1P", 1, 1), ("2P", 2, 1)]

# Integration range per level.  Higher states are bigger; the log and Martin
# potentials are also softer than Cornell at large r so they need more room.
_R_MAX = {"1S": 10.0, "2S": 14.0, "3S": 18.0, "1P": 14.0, "2P": 18.0}


def spectrum(
    potential: Potential,
    q1: str,
    q2: str,
    levels=LEVELS,
    h: float = 1e-3,
    r_max: dict[str, float] | None = None,
) -> dict[str, RadialState]:
    """Return {label: RadialState} for e.g. spectrum(MARTIN, "c", "b")."""
    rm = dict(_R_MAX)
    if r_max:
        rm.update(r_max)
    m1, m2 = potential.mass(q1), potential.mass(q2)
    return {lab: solve(potential, m1, m2, n, l, r_max=rm[lab], h=h) for lab, n, l in levels}
