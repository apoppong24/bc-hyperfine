"""The centroid machinery must agree with the (2J+1)-weighted average of the
J-resolved levels: for S waves over all four states, for L>0 over the triplet
(the singlet sits at the centroid because the contact term vanishes for L>0)."""
import numpy as np
import pytest

from quarkonium.fit import predict
from quarkonium.flexible import (CENTROIDS, centroid_dataset, fit_spline_centroid, knot_grid,
                                 make_spline_potential, predict_centroids, V0)
from quarkonium.potentials import MARTIN


def _P(K=6):
    knots = knot_grid(K); k1 = int(np.argmin(np.abs(np.log(knots))))
    s = MARTIN(knots) + 0.3 / knots; s += (V0 + 0.3) - s[k1]
    return make_spline_potential(knots, np.delete(s, k1), 0.3, 1.8, 5.18, 0.4)


def test_dataset_has_eleven_centroids():
    rows = centroid_dataset()
    assert len(rows) == 11
    assert sum(1 for r in rows if r[0] == "cc") == 4
    assert len(centroid_dataset(exclude=("D",))) == 9


@pytest.mark.parametrize("pair", ["cc", "bb", "cb"])
def test_centroid_equals_weighted_mean_of_fine_structure(pair):
    P = _P()
    cen = predict_centroids(P, pair, ["1S", "1P"], h=2e-3)
    s = predict(P, pair, ["11S0", "13S1"], h=2e-3)
    assert abs((s["11S0"] + 3 * s["13S1"]) / 4 - cen["1S"]) < 1e-6
    if pair == "cb":
        trip = predict(P, pair, ["13P0", "1P1", "1P1'", "13P2"], h=2e-3)
        # trace of the mixed J=1 pair is invariant: (1·P0 + 3·(P1 + P1') ... ) weights 1,3,3,5 over 12
        mean = (trip["13P0"] + 3 * trip["1P1"] + 3 * trip["1P1'"] + 5 * trip["13P2"]) / 12
    else:
        trip = predict(P, pair, ["13P0", "13P1", "13P2", "11P1"], h=2e-3)
        mean = (trip["13P0"] + 3 * trip["13P1"] + 5 * trip["13P2"]) / 9
        assert abs(trip["11P1"] - cen["1P"]) < 1e-6
    assert abs(mean - cen["1P"]) < 1e-6


def test_centroid_fit_layout_roundtrip(tmp_path):
    """Short fit (loose tolerance via few data) just to check the parameter layout and JSON."""
    from quarkonium.flexible import SplineFit
    f = fit_spline_centroid(K=6, lam=1.0, verbose=False, exclude=("D", "3S", "3P"))
    assert len(f.x) == 6 + 3 and f.cov.shape == (9, 9)
    assert f.x[5] == 0.0 and f.cov[5, 5] == 0.0          # kappa fixed
    f.to_json(tmp_path / "c.json"); g = SplineFit.from_json(tmp_path / "c.json")
    V, s = g.band(np.array([0.5, 1.0, 3.0]))
    assert s[1] < 1e-6 and s[0] > 0 and s[2] > 0
