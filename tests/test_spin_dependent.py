import numpy as np
import pytest

from quarkonium import MARTIN, hyperfine_splitting, spectrum
from quarkonium.spin_dependent import LS, fine_structure, radial_coefficients, tensor


@pytest.fixture(scope="module")
def cc_1P():
    return spectrum(MARTIN, "c", "c", levels=[("1P", 1, 1)], r_max={"1P": 14})["1P"]


@pytest.fixture(scope="module")
def cb():
    return spectrum(MARTIN, "c", "b", levels=[("1S", 1, 0), ("1P", 1, 1)], r_max={"1S": 10, "1P": 14})


def test_angular_factors():
    assert LS(1, 1, 2) == 1 and LS(1, 1, 1) == -1 and LS(1, 1, 0) == -2
    assert tensor(1, 1, 2) == pytest.approx(-0.4)
    assert tensor(1, 1, 1) == 2 and tensor(1, 1, 0) == -4
    assert tensor(1, 0, 1) == 0


def test_lande_centroid_rule(cc_1P):
    """(1·M(³P₀) + 3·M(³P₁) + 5·M(³P₂))/9 must equal the spin-averaged mass."""
    m = {f.label: f.mass for f in fine_structure(cc_1P)}
    cen = (m["13P0"] + 3 * m["13P1"] + 5 * m["13P2"]) / 9
    assert cen == pytest.approx(cc_1P.mass, abs=1e-6)


def test_singlet_P_at_centroid_for_equal_masses(cc_1P):
    m = {f.label: f.mass for f in fine_structure(cc_1P)}
    assert m["11P1"] == pytest.approx(cc_1P.mass, abs=1e-9)
    assert m["13P0"] < m["13P1"] < m["13P2"]        # correct ordering


def test_S_wave_reduces_to_thesis_hyperfine(cb):
    levels = {f.label: f.mass for f in fine_structure(cb["1S"])}
    assert levels["13S1"] - levels["11S0"] == pytest.approx(hyperfine_splitting(cb["1S"]), rel=1e-9)
    # and spin average is preserved
    assert (levels["11S0"] + 3 * levels["13S1"]) / 4 == pytest.approx(cb["1S"].mass, abs=1e-9)


def test_unequal_mass_mixing_trace(cb):
    """The two mixed J=1 B_c states carry the trace of the LS-basis matrix,
    which equals E&Q's jj-basis trace: -(T1/4)(1/mc²+1/mb²) - T2/(mc mb) + 2T4/(mc mb)."""
    st = cb["1P"]
    c = radial_coefficients(st)
    mc, mb = st.m1, st.m2
    expected = -(c.T1 / 4) * (1 / mc**2 + 1 / mb**2) - c.T2 / (mc * mb) + 2 * c.T4 / (mc * mb)
    mixed = [f for f in fine_structure(st) if f.J == 1]
    assert len(mixed) == 2
    assert sum(f.shift for f in mixed) == pytest.approx(expected, rel=1e-9)
    assert all(f.mixing_angle is not None for f in mixed)


def test_bc_1P_near_eichten_quigg(cb):
    m = {f.label: f.mass for f in fine_structure(cb["1P"])}
    assert m["13P0"] == pytest.approx(6.700, abs=0.02)
    assert m["13P2"] == pytest.approx(6.747, abs=0.04)


def test_fit_parser_and_predict_labels():
    from quarkonium.fit import _parse, predict
    assert _parse("13P2") == (1, 1, 1, 2)
    assert _parse("11P1") == (1, 0, 1, 1)
    assert _parse("1P1'") == (1, None, 1, 1)
    pr = predict(MARTIN, "cb", ["11S0", "13S1", "1P1", "1P1'"], h=2e-3)
    assert pr["13S1"] > pr["11S0"] and pr["1P1'"] > pr["1P1"]
