"""quarkonium – Numerov solver for heavy quark–antiquark bound states.

Refactored from the MSc project notebooks "B_c Meson Masses".

Quick start
-----------
>>> from quarkonium import MARTIN, solve, spectrum
>>> s = solve(MARTIN, MARTIN.m_b, MARTIN.m_b, n=1, l=0)   # bottomonium 1S
>>> round(s.E, 4), round(s.mass, 4)
(-0.8742, 9.4738)
>>> spectrum(MARTIN, "c", "b")                            # all five levels
"""
from .numerov import count_nodes, numerov
from .observables import (
    bc_mass,
    hyperfine_splitting,
    singlet_triplet,
    spin_average_P,
    spin_average_S,
    splittings,
)
from .potentials import CORNELL, LOGARITHMIC, MARTIN, POTENTIALS, Potential
from .solver import RadialState, reduced_mass, solve
from .spectrum import LEVELS, spectrum
from .spin_dependent import FineLevel, fine_structure, full_spectrum, radial_coefficients

__all__ = [
    "CORNELL", "LOGARITHMIC", "MARTIN", "POTENTIALS", "Potential",
    "RadialState", "solve", "spectrum", "LEVELS", "reduced_mass",
    "numerov", "count_nodes",
    "hyperfine_splitting", "spin_average_S", "spin_average_P",
    "singlet_triplet", "bc_mass", "splittings",
    "FineLevel", "fine_structure", "full_spectrum", "radial_coefficients",
]
