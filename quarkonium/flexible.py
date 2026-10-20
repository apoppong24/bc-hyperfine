"""Model-independent (spline) quark–antiquark potential and its fit to the
charmonium + bottomonium spectrum.

    V(r) = -kappa/r + S(r)

S is a natural cubic spline in x = ln r through K knots log-spaced on
[r_lo, r_hi]; for r < r_lo, S is held at S(r_lo) (the Coulomb term dominates
there); for r > r_hi, S is continued linearly in r with the slope at r_hi
(string-like; the data do not reach there and the band shows it).

The knot at r = 1 GeV^-1 is fixed so that V(1) = V0 = -1.166 GeV (the Martin
value), removing the exact degeneracy between an additive constant and m_c + m_b.
Free parameters: K-1 knot values, kappa, m_c, m_b, Lambda (one-loop running
alpha_s for the spin-dependent terms, as in the B_c paper).

A smoothness prior  sqrt(lam) * D2 s  (second differences in x) is appended
to the residual vector, so the least-squares covariance already includes it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .fit import dataset, predict
from .potentials import Potential

V0 = -1.166          # GeV, value of V at r = 1 GeV^-1 (fixes the offset)


def knot_grid(K: int, r_lo: float = 0.15, r_hi: float = 8.0) -> np.ndarray:
    """K knots log-spaced on [r_lo, r_hi] with r = 1 inserted exactly."""
    x = np.linspace(np.log(r_lo), np.log(r_hi), K)
    # move the knot nearest to x=0 onto x=0 exactly
    k = int(np.argmin(np.abs(x)))
    x[k] = 0.0
    return np.exp(x)


def make_spline_potential(knots: np.ndarray, s_free: np.ndarray, kappa: float,
                          m_c: float, m_b: float, lam_qcd: float,
                          alpha_s_scale: str = "geometric") -> Potential:
    """Build the Potential from the free knot values (the r=1 knot is V0)."""
    k1 = int(np.argmin(np.abs(np.log(knots))))
    s = np.insert(np.asarray(s_free, float), k1, V0 + kappa)     # so that V(1) = V0 exactly
    x = np.log(knots)
    cs = CubicSpline(x, s, bc_type="natural")
    slope_hi = cs(x[-1], 1) / knots[-1]      # dS/dr at r_hi  (dS/dx / r)
    s_lo, s_hi = s[0], s[-1]
    r_lo, r_hi = knots[0], knots[-1]

    def S(r):
        r = np.asarray(r, float)
        out = np.empty_like(r)
        inside = (r >= r_lo) & (r <= r_hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[inside] = cs(np.log(r[inside]))
        out[r < r_lo] = s_lo
        out[r > r_hi] = s_hi + slope_hi * (r[r > r_hi] - r_hi)
        return out

    def V(r):
        r = np.asarray(r, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -kappa / r + S(r)

    P = Potential("spline", V, m_c, m_b, 0.0, f"spline K={len(knots)}",
                  alpha_s_scale=alpha_s_scale, lambda_qcd=lam_qcd)
    P.__dict__["_spline"] = (knots, s, cs, S)       # frozen dataclass: side-channel
    return P


def spline_basis(knots: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Matrix B[i, k] = dV(r_i)/ds_k for the FREE knot values (linear map), so
    that var V(r) = B C B^T with C the covariance of the free knots."""
    k1 = int(np.argmin(np.abs(np.log(knots))))
    K = len(knots)
    cols = []
    for k in range(K):
        if k == k1:
            continue
        unit = np.zeros(K); unit[k] = 1.0
        s_free = np.delete(unit, k1)
        P = make_spline_potential(knots, s_free, 0.0, 1.0, 1.0, 0.3)
        S = P.__dict__["_spline"][3]
        cols.append(S(r) - make_spline_potential(knots, np.zeros(K - 1), 0.0, 1.0, 1.0, 0.3).__dict__["_spline"][3](r))
    return np.array(cols).T


@dataclass
class SplineFit:
    K: int
    lam: float
    knots: np.ndarray
    names: list
    x: np.ndarray
    cov: np.ndarray
    chi2_data: float
    chi2_prior: float
    dof: int
    residuals: list = field(default_factory=list)
    seconds: float = 0.0

    def potential(self) -> Potential:
        K = self.K
        return make_spline_potential(self.knots, self.x[:K - 1], *self.x[K - 1:])

    def band(self, r: np.ndarray):
        """V(r) and its 1-sigma band from the fit covariance."""
        P = self.potential()
        K = self.K
        B = spline_basis(self.knots, r)                      # dV/ds_k
        # dV/dkappa = -1/r + (basis function of the fixed knot), since s[k1] = V0 + kappa
        k1 = int(np.argmin(np.abs(np.log(self.knots))))
        unit = np.zeros(K); unit[k1] = 1.0
        Pk = make_spline_potential(self.knots, np.zeros(K - 1), 1.0, 1.0, 1.0, 0.3)
        P0 = make_spline_potential(self.knots, np.zeros(K - 1), 0.0, 1.0, 1.0, 0.3)
        with np.errstate(divide="ignore"):
            dV_dkappa = (Pk.__dict__["_spline"][3](r) - P0.__dict__["_spline"][3](r)) - 1.0 / r
        J = np.column_stack([B, dV_dkappa])                  # masses/Lambda do not enter V
        C = self.cov[:K, :K]
        var = np.einsum("ij,jk,ik->i", J, C, J)
        return P(r), np.sqrt(np.maximum(var, 0.0))

    def to_json(self, path):
        d = dict(K=self.K, lam=self.lam, knots=self.knots.tolist(), names=self.names, x=self.x.tolist(),
                 cov=self.cov.tolist(), chi2_data=self.chi2_data, chi2_prior=self.chi2_prior, dof=self.dof,
                 residuals=[dict(pair=p, label=l, exp=e, model=m, pull=q) for p, l, e, m, q in self.residuals],
                 seconds=self.seconds)
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path):
        d = json.loads(Path(path).read_text())
        return SplineFit(d["K"], d["lam"], np.array(d["knots"]), d["names"], np.array(d["x"]), np.array(d["cov"]),
                         d["chi2_data"], d["chi2_prior"], d["dof"],
                         [(r["pair"], r["label"], r["exp"], r["model"], r["pull"]) for r in d["residuals"]], d["seconds"])


def fit_spline(K: int = 8, lam: float = 1.0, sigma_th: float = 10.0, h: float = 2e-3,
               start_from: Potential | None = None, verbose: bool = True, exclude=(),
               kappa_fixed: float | None = None) -> SplineFit:
    """kappa_fixed: hold the Coulomb coefficient at this value (the spectrum
    does not probe r < 0.2 GeV^-1, so kappa is otherwise driven by the
    hyperfine term alone)."""
    knots = knot_grid(K)
    x_knots = np.log(knots)
    k1 = int(np.argmin(np.abs(x_knots)))
    data = dataset(sigma_th, exclude)
    by_pair = {}
    for pair, lab, m, s in data:
        by_pair.setdefault(pair, []).append((lab, m, s))

    # starting values: sample the Martin potential (minus a Coulomb piece) at the knots
    if start_from is None:
        from .potentials import MARTIN
        start_from = MARTIN
    kappa0 = 0.4
    s0_full = start_from(knots) + kappa0 / knots
    s0_full += (V0 + kappa0) - s0_full[k1]           # anchor: V(1) = V0
    s0 = np.delete(s0_full, k1)
    x0 = np.concatenate([s0, [kappa0, 1.75, 5.15, 0.40]])
    names = [f"s({r:.2f})" for i, r in enumerate(knots) if i != k1] + ["kappa", "m_c", "m_b", "Lambda"]
    lo = np.concatenate([np.full(K - 1, -8.0), [0.0, 1.0, 4.0, 0.05]])
    hi = np.concatenate([np.full(K - 1, 8.0), [1.5, 2.5, 6.0, 0.9]])
    if kappa_fixed is not None:
        kappa0 = kappa_fixed
        s0_full = start_from(knots) + kappa0 / knots
        s0_full += (V0 + kappa0) - s0_full[k1]
        x0 = np.concatenate([np.delete(s0_full, k1), [kappa0, 1.75, 5.15, 0.40]])
        lo[K - 1] = kappa_fixed - 1e-9; hi[K - 1] = kappa_fixed + 1e-9

    # second-difference operator on the FULL knot vector, applied to free + fixed
    D2 = np.zeros((K - 2, K))
    for i in range(K - 2):
        D2[i, i:i + 3] = [1.0, -2.0, 1.0]
    dx = x_knots[1] - x_knots[0]
    D2 /= dx**2

    def residuals(x):
        s_free, kappa, m_c, m_b, lam_q = x[:K - 1], x[K - 1], x[K], x[K + 1], x[K + 2]
        P = make_spline_potential(knots, s_free, kappa, m_c, m_b, lam_q)
        res = []
        for pair, rows in by_pair.items():
            try:
                pred = predict(P, pair, [r[0] for r in rows], h=h)
            except ValueError:
                return np.full(len(data) + K - 2, 1e3)
            res += [(pred[lab] - m) / s for lab, m, s in rows]
        s_full = np.insert(s_free, k1, V0 + kappa)
        prior = np.sqrt(lam) * (D2 @ s_full)
        return np.concatenate([res, prior])

    t0 = time.time()
    r = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3,
                      verbose=2 if verbose else 0, max_nfev=400)
    secs = time.time() - t0
    J = r.jac
    cov = np.linalg.pinv(J.T @ J)
    nd = len(data)
    chi2_data = float(np.sum(r.fun[:nd] ** 2)); chi2_prior = float(np.sum(r.fun[nd:] ** 2))
    P = make_spline_potential(knots, r.x[:K - 1], *r.x[K - 1:])
    rows_out = []
    for pair, rows in by_pair.items():
        pred = predict(P, pair, [q[0] for q in rows], h=h)
        rows_out += [(pair, lab, m, pred[lab], (pred[lab] - m) / s) for lab, m, s in rows]
    return SplineFit(K, lam, knots, names, r.x, cov, chi2_data, chi2_prior, nd - len(r.x), rows_out, secs)


# ---------------------------------------------------------------------------
# Spin-averaged (centroid) fit: V(r) from the spin-independent spectrum only,
# so that the extraction does not depend on the perturbative spin-dependent
# treatment at all.
# ---------------------------------------------------------------------------
# Centroids M̄ = Σ(2J+1)M_J / Σ(2J+1) over the triplet (S states: (M_PS + 3 M_V)/4).
# Where a multiplet is incomplete the missing member is estimated from the
# pattern of the lower multiplets and an error assigned (noted per entry).
CENTROIDS = {
    "cc": {
        "1S": (3068.70, 0.3),      # η_c, J/ψ
        "1P": (3525.37, 0.3),      # χ_c0,1,2  (h_c at 3525.37: identical to 0.01 MeV)
        "2S": (3674.02, 0.5),      # η_c(2S), ψ(2S)
        "1D": (3822.51, 2.0),      # ψ(3770), ψ2(3823), ψ3(3842); ψ(3770) sits at DD̄ threshold
    },
    "bb": {
        "1S": (9444.97, 1.5),      # η_b, Υ(1S)
        "1P": (9899.87, 0.4),      # χ_b0,1,2(1P)
        "2S": (10017.3, 2.6),      # η_b(2S), Υ(2S)
        "1D": (10163.7, 3.0),      # Υ2(1D) only; |centroid − J=2| ≲ 2 MeV in any potential model
        "2P": (10260.24, 0.5),     # χ_b0,1,2(2P)
        "3S": (10342.1, 5.0),      # Υ(3S) − (3/4)Δ_hyp(3S), Δ_hyp(3S) = 17 ± 6 from the 1S/2S pattern
        "3P": (10517.2, 4.0),      # χ_b1,2(3P); χ_b0(3P) ≈ χ_b1 − 18 ± 8 from the 1P/2P pattern
    },
}


def centroid_dataset(sigma_th=10.0, exclude=()):
    rows = []
    for pair, table in CENTROIDS.items():
        for lab, (m, s) in table.items():
            if any(x in lab for x in exclude):
                continue
            rows.append((pair, lab, m / 1000.0, np.hypot(s, sigma_th) / 1000.0))
    return rows


def predict_centroids(P: Potential, pair: str, labels, h=2e-3):
    """Spin-averaged masses m1 + m2 + E_nl (GeV) for labels like '2P'."""
    from .fit import R_MAX
    from .solver import solve
    q = {"cc": ("c", "c"), "bb": ("b", "b"), "cb": ("c", "b")}[pair]
    m1, m2 = P.mass(q[0]), P.mass(q[1])
    out = {}
    for lab in labels:
        n, L = int(lab[0]), "SPDF".index(lab[1])
        st = solve(P, m1, m2, n, L, r_max=R_MAX[L][n], h=h)
        out[lab] = m1 + m2 + st.E
    return out


def fit_spline_centroid(K: int = 6, lam: float = 1.0, sigma_th: float = 10.0, kappa: float = 0.0,
                        h: float = 2e-3, verbose: bool = True, exclude=()) -> SplineFit:
    """Fit the spline potential to the spin-averaged centroids only.
    The parameter vector keeps the layout [s_free, kappa, m_c, m_b, Lambda] of
    fit_spline so that SplineFit.potential()/band() work unchanged; kappa and
    Lambda are held fixed (kappa: the centroid spectrum is blind to r < 0.2 GeV⁻¹;
    Lambda: no spin-dependent term enters)."""
    knots = knot_grid(K)
    x_knots = np.log(knots)
    k1 = int(np.argmin(np.abs(x_knots)))
    data = centroid_dataset(sigma_th, exclude)
    by_pair = {}
    for pair, lab, m, s in data:
        by_pair.setdefault(pair, []).append((lab, m, s))

    from .potentials import MARTIN
    s0_full = MARTIN(knots) + kappa / knots
    s0_full += (V0 + kappa) - s0_full[k1]
    LAM = 0.40                                   # dummy; no spin-dependent term enters
    x0 = np.concatenate([np.delete(s0_full, k1), [1.75, 5.15]])      # free: s_free, m_c, m_b
    lo = np.concatenate([np.full(K - 1, -8.0), [1.0, 4.0]])
    hi = np.concatenate([np.full(K - 1, 8.0), [2.5, 6.0]])
    names = [f"s({r:.2f})" for i, r in enumerate(knots) if i != k1] + ["kappa", "m_c", "m_b", "Lambda"]

    D2 = np.zeros((K - 2, K))
    for i in range(K - 2):
        D2[i, i:i + 3] = [1.0, -2.0, 1.0]
    D2 /= (x_knots[1] - x_knots[0]) ** 2

    def residuals(x):
        s_free, m_c, m_b = x[:K - 1], x[K - 1], x[K]
        P = make_spline_potential(knots, s_free, kappa, m_c, m_b, LAM)
        res = []
        for pair, rows in by_pair.items():
            try:
                pred = predict_centroids(P, pair, [r[0] for r in rows], h=h)
            except ValueError:
                return np.full(len(data) + K - 2, 1e3)
            res += [(pred[lab] - m) / s for lab, m, s in rows]
        prior = np.sqrt(lam) * (D2 @ np.insert(s_free, k1, V0 + kappa))
        return np.concatenate([res, prior])

    t0 = time.time()
    r = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3,
                      verbose=2 if verbose else 0, max_nfev=400)
    secs = time.time() - t0
    cov_free = np.linalg.pinv(r.jac.T @ r.jac)
    nd = len(data)
    chi2_data = float(np.sum(r.fun[:nd] ** 2)); chi2_prior = float(np.sum(r.fun[nd:] ** 2))
    # embed in the standard layout [s_free, kappa, m_c, m_b, Lambda] with zero rows for the fixed parameters
    x_full = np.concatenate([r.x[:K - 1], [kappa], r.x[K - 1:], [LAM]])
    free_idx = list(range(K - 1)) + [K, K + 1]
    cov = np.zeros((K + 3, K + 3))
    cov[np.ix_(free_idx, free_idx)] = cov_free
    P = make_spline_potential(knots, r.x[:K - 1], kappa, r.x[K - 1], r.x[K], LAM)
    rows_out = []
    for pair, rows in by_pair.items():
        pred = predict_centroids(P, pair, [q[0] for q in rows], h=h)
        rows_out += [(pair, lab, m, pred[lab], (pred[lab] - m) / s) for lab, m, s in rows]
    return SplineFit(K, lam, knots, names, x_full, cov, chi2_data, chi2_prior, nd - (K + 1), rows_out, secs)
