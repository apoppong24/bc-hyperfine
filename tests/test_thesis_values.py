"""Regression tests: the refactored solver must reproduce the thesis
tables to the precision of the original hand-tuned trial energies
(≈1 MeV; a few levels were tuned more coarsely and get a looser tolerance).
Known thesis typos are tested against the notebook value instead.
"""
import numpy as np
import pytest

from quarkonium import (
    CORNELL, LOGARITHMIC, MARTIN, hyperfine_splitting, numerov,
    singlet_triplet, solve, spectrum,
)

# Tables 2–4 (GeV).  Tolerance per entry = precision of the hand search.
EIGEN = {
    ("martin", "bb"): {"1S": -0.8742, "2S": -0.2987, "3S": 0.0367, "1P": -0.45848, "2P": -0.07486},
    ("martin", "cc"): {"1S": -0.5035, "2S": 0.102, "3S": 0.4542, "1P": -0.06637, "2P": 0.33696},
    ("martin", "cb"): {"1S": -0.6445, "2S": -0.0502, "3S": 0.2968, "1P": -0.21518, "2P": 0.1805},
    ("log", "bb"): {"1S": -0.3672, "2S": 0.2235, "3S": 0.5478, "1P": 0.0704, "2P": 0.4441},
    ("log", "cc"): {"1S": 0.0672, "2S": 0.6563, "3S": 0.9807, "1P": 0.5047, "2P": 0.8797},
    ("log", "cb"): {"1S": -0.0883, "2S": 0.4997, "3S": 0.8296, "1P": 0.3484, "2P": 0.7221},
    # Table 4 prints cc 1S as 0.2657; the notebook (and Table 5's 3.9375) use 0.2575.
    ("cornell", "bb"): {"1S": -0.1705, "2S": 0.4215, "3S": 0.7658, "1P": 0.3279, "2P": 0.68198},
    ("cornell", "cc"): {"1S": 0.2575, "2S": 0.8482, "3S": 1.2717, "1P": 0.6859, "2P": 1.1287},
    ("cornell", "cb"): {"1S": 0.1116, "2S": 0.6813, "3S": 1.0686, "1P": 0.5468, "2P": 0.9491},
}
LOOSE = {("log", "bb", "2S"), ("log", "bb", "3S"), ("log", "cb", "3S"),
         ("martin", "cb", "3S"), ("log", "cc", "2P"), ("cornell", "bb", "3S")}
POT = {"martin": MARTIN, "log": LOGARITHMIC, "cornell": CORNELL}


@pytest.fixture(scope="module")
def spectra():
    return {k: spectrum(POT[k[0]], k[1][0], k[1][1]) for k in EIGEN}


@pytest.mark.parametrize("key", list(EIGEN))
def test_eigenvalues_match_thesis(spectra, key):
    for lab, ref in EIGEN[key].items():
        tol = 7e-3 if (key[0], key[1], lab) in LOOSE else 1e-3
        assert spectra[key][lab].E == pytest.approx(ref, abs=tol), (key, lab)


@pytest.mark.parametrize("key", list(EIGEN))
def test_node_counts(spectra, key):
    for lab, st in spectra[key].items():
        assert st.nodes == st.n - 1, (key, lab)


def test_normalisation(spectra):
    st = spectra[("martin", "bb")]["2S"]
    assert np.sum(st.u**2) * st.h == pytest.approx(1.0, rel=1e-6)


def test_R0_and_hyperfine_table_9_12(spectra):
    # Table 9 / Table 12, 1S bottomonium
    bb = spectra[("martin", "bb")]["1S"]
    assert bb.R_at_origin_sq() == pytest.approx(4.4242, abs=3e-3)
    assert hyperfine_splitting(bb) == pytest.approx(0.0632, abs=5e-4)
    cc = spectra[("cornell", "cc")]["1S"]
    assert cc.R_at_origin_sq() == pytest.approx(1.4574, abs=3e-3)
    assert hyperfine_splitting(cc) == pytest.approx(0.1186, abs=5e-4)


def test_bc_masses_table_15(spectra):
    cb = spectra[("martin", "cb")]
    for lab, (bc, bcs) in {"1S": (6.2767, 6.3471), "2S": (6.8944, 6.9336), "3S": (7.2494, 7.2780)}.items():
        s, t = singlet_triplet(cb[lab].mass, hyperfine_splitting(cb[lab]))
        assert s == pytest.approx(bc, abs=2e-3)
        assert t == pytest.approx(bcs, abs=2e-3)


def test_log_potential_level_spacing_is_mass_independent(spectra):
    """Exact property of V = C ln r: E_n(mu) - E_m(mu) is independent of mu."""
    bb, cc = spectra[("log", "bb")], spectra[("log", "cc")]
    assert bb["2S"].E - bb["1S"].E == pytest.approx(cc["2S"].E - cc["1S"].E, abs=1e-6)


def test_grid_convergence():
    e = [solve(MARTIN, MARTIN.m_b, MARTIN.m_b, 1, 0, r_max=10, h=h).E for h in (2e-3, 5e-4)]
    assert e[0] == pytest.approx(e[1], abs=1e-6)


def test_numerov_harmonic_oscillator():
    """Unit check of the integrator alone: u'' = -u  =>  u = sin(x)."""
    h = 1e-3
    x = np.arange(0, 2 * np.pi, h)
    u = numerov(np.ones_like(x), h, 0.0, np.sin(h))
    assert np.max(np.abs(u - np.sin(x))) < 1e-8


def test_bad_bracket_raises():
    with pytest.raises(ValueError):
        solve(MARTIN, 1.8, 1.8, n=3, l=0, r_max=2.0)   # 3S not bound on r<2
