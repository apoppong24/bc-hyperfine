"""Checks on the contact-interaction machinery."""
import numpy as np
import pytest

from quarkonium.hyperfine import (HYPERFINE, HyperfineModel, base_potential, dataset, profile, width_of)


def test_profiles_are_normalised():
    r = np.linspace(1e-4, 30, 300001)
    for presc, p in (("gauss", [1.0]), ("gi", [1.8, 1.55]), ("yukawa", [2.47])):
        f = profile(presc, np.array(p), 1.8, 5.2)
        assert abs(np.trapezoid(f(r) * 4 * np.pi * r**2, r) - 1.0) < 1e-3


def test_widths():
    assert abs(width_of("yukawa", np.array([2.0]), 1, 1) - np.sqrt(6) / 2.0) < 1e-9
    assert abs(width_of("gauss", np.array([2.0]), 1, 1) - np.sqrt(1.5) / 2.0) < 1e-3
    assert width_of("pert", np.array([]), 1, 1) == 0.0


def test_dataset_variants():
    assert len(dataset()) == 9
    assert len(dataset(exclude=("cc2S",))) == 8
    row = [r for r in dataset(cc1S_lattice=True) if r[1] == "1S" and r[0] == "cc"][0]
    assert abs(row[2] - 0.1201) < 1e-9


def test_narrow_gaussian_collapses_not_converges():
    """A delta function has no non-perturbative solution in three dimensions: as the
    Gaussian narrows the attractive singlet channel collapses, so the smeared
    prescriptions have no first-order limit.  The P-wave shift, by contrast, does go to zero."""
    base = base_potential(0.0)
    a = 0.3
    pert = HyperfineModel("pert", "fixed", base, 0.0, np.array([a]))
    ratios = []
    for sig in (4.0, 8.0, 16.0):
        g = HyperfineModel("gauss", "fixed", base, 0.0, np.array([a, sig]))
        ratios.append(g.splitting("cc", "1S") / pert.splitting("cc", "1S"))
    assert ratios[0] > 1.0 and ratios[1] > ratios[0] and ratios[2] > 10 * ratios[1]
    assert abs(HyperfineModel("gauss", "fixed", base, 0.0, np.array([a, 16.0])).splitting("cc", "1P")) < 1e-4
    assert pert.splitting("cc", "1P") == 0.0


def test_pwave_shift_sign_and_growth():
    """Smearing pushes the P-wave singlet below the centroid, more for wider profiles."""
    base = base_potential(0.0)
    s1 = HyperfineModel("gauss", "fixed", base, 0.0, np.array([0.3, 1.0])).splitting("cc", "1P")
    s2 = HyperfineModel("gauss", "fixed", base, 0.0, np.array([0.3, 2.0])).splitting("cc", "1P")
    assert s1 < s2 < 0
