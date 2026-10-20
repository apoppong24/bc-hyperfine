import numpy as np
import pytest
from quarkonium import MARTIN, spectrum, full_spectrum
from quarkonium.radiative import (e1_statistical_factor, effective_charges, radiative_table, wigner_6j)


def test_6j_known_values():
    # {1 1 1; 1 1 1} = 1/6 ;  {0 1 1; J 1 1}^2 = 1/9 for J = 0,1,2
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6)
    for J in (0, 1, 2):
        assert wigner_6j(0, 1, 1, J, 1, 1) ** 2 == pytest.approx(1 / 9)


def test_eichten_quigg_statistical_factors():
    assert all(e1_statistical_factor(1, J, 0, 1, 1) == pytest.approx(1.0) for J in (0, 1, 2))
    assert e1_statistical_factor(1, 1, 0, 0, 0) == pytest.approx(3.0)
    assert e1_statistical_factor(0, 1, 1, 1, 1) == pytest.approx(1.0)   # symmetric


def test_effective_charges_reduce_for_quarkonium():
    eQ, mu = effective_charges("c", "c", 1.5, 1.5)
    assert eQ == pytest.approx(2 / 3) and mu == pytest.approx((2 / 3) / (2 * 1.5))


def test_charmonium_E1_widths_near_pdg():
    S = spectrum(MARTIN, "c", "c", levels=[("1S", 1, 0), ("1P", 1, 1)], r_max={"1S": 14, "1P": 14})
    rows = {(r["initial"], r["final"]): r["width_keV"] for r in radiative_table(S, full_spectrum(S), "c", "c")}
    # PDG: chi_c0,1,2 -> J/psi gamma = 151, 288, 374 keV ; model within ~25%
    assert rows[("13P0", "13S1")] == pytest.approx(151, rel=0.3)
    assert rows[("13P1", "13S1")] == pytest.approx(288, rel=0.3)
    assert rows[("13P2", "13S1")] == pytest.approx(374, rel=0.3)
    assert 0.5 < rows[("13S1", "11S0")] < 5      # M1, PDG 1.6 keV
