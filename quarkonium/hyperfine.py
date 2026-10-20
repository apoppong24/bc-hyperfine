"""Four prescriptions for the spin–spin (contact) interaction, tested against
the complete set of measured heavy-quarkonium hyperfine splittings.

The spin-averaged potential V(r) and the quark masses are NOT fitted here:
they are taken from the model-independent centroid extraction of the
companion paper (results/centroid_K6_lam1*.json), so that only the contact
prescription is being tested.

Hyperfine Hamiltonian (all prescriptions):

    H_SS = (32π α_s / 9 m1 m2)  s1·s2  δ_X(r)

with δ_X a normalised profile (∫ δ_X d³r = 1):

    pert    first-order perturbation theory, δ_X = δ³(r):  Δ = 8 α_s |R(0)|² / (9 m1 m2)
    gauss   Gaussian, (σ/√π)³ exp(−σ²r²), σ mass-independent
    gi      Gaussian with the Godfrey–Isgur mass-dependent width
              σ_ij² = σ0² [½ + ½ (4 m_i m_j/(m_i+m_j)²)⁴] + s² (2 m_i m_j/(m_i+m_j))²
    yukawa  m² e^{−m r} / (4π r), the short-range form of the lattice spin–spin
            potential (Koma & Koma 2007: Yukawa with the 0⁻⁺ glueball mass, 2.47 GeV)

For gauss / gi / yukawa the singlet (s1·s2 = −3/4) and triplet (+1/4) states
are solved non-perturbatively with the smeared term in the Hamiltonian, for
S and P waves alike.  The coupling is either fixed (one α_s for all systems)
or one-loop running, α_s(√(m1 m2)) with Λ fitted.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .fit import R_MAX
from .flexible import SplineFit
from .potentials import Potential
from .solver import solve

# ---------------------------------------------------------------- data
# (pair, label, value MeV, error MeV, source).  S-wave: M(triplet) − M(singlet).
# P-wave: M(h) − centroid of the χ triplet; vanishes for a pure contact term.
HYPERFINE = [
    ("cc", "1S", 112.81, 0.30, "J/psi − eta_c, PDG 2026"),
    ("cc", "2S", 48.30, 0.60, "psi(2S) − eta_c(2S), PDG 2026"),
    ("bb", "1S", 61.7, 2.0, "Upsilon(1S) − eta_b, PDG 2026"),
    ("bb", "2S", 24.4, 3.5, "Upsilon(2S) − eta_b(2S), PDG 2026 (Belle)"),
    ("cb", "1S", 64.5, 1.7, "B_c* − B_c, ATLAS 2026"),
    ("cb", "2S", 35.4, 2.4, "B_c*(2S) − B_c(2S), ATLAS 2026 + CMS 2019"),
    ("cc", "1P", 0.07, 0.15, "h_c − centroid(chi_c0,1,2), PDG 2026"),
    ("bb", "1P", -0.57, 0.80, "h_b(1P) − centroid(chi_b(1P)), PDG 2026"),
    ("bb", "2P", -0.44, 1.20, "h_b(2P) − centroid(chi_b(2P)), PDG 2026"),
]
# lattice values, not fitted: used for comparison only
LATTICE = {
    ("cc", "1S"): (120.3, 1.1, "HPQCD 2020, connected only; eta_c annihilation −7.3(1.2)"),
    ("bb", "1S"): (60.0, 6.4, "HPQCD 2014 lattice NRQCD"),
    ("cb", "1S"): (54.0, 3.0, "HPQCD 2012"),
}
LATTICE_RATIO_BB = (0.445, 0.028, "HPQCD 2014: Δ(2S)/Δ(1S)")

PAIRS = {"cc": ("c", "c"), "bb": ("b", "b"), "cb": ("c", "b")}
SPIN = {"singlet": -0.75, "triplet": 0.25}


def dataset(floor_S=3.0, floor_P=0.5, exclude=(), cc1S_lattice=False):
    """Rows (pair, label, value, sigma) in GeV with theory floors in quadrature.
    `exclude`: observables to drop, as 'cc2S' etc.  `cc1S_lattice`: replace the
    measured cc̄ 1S splitting by the lattice connected value (experiment plus the
    η_c annihilation shift of 7.3(1.2) MeV, HPQCD 2020), which is what a potential
    model without annihilation should be compared to."""
    rows = []
    for pair, lab, v, e, _ in HYPERFINE:
        if f"{pair}{lab}" in exclude:
            continue
        if cc1S_lattice and (pair, lab) == ("cc", "1S"):
            v, e = 120.1, 1.2
        fl = floor_P if lab[1] == "P" else floor_S
        rows.append((pair, lab, v / 1000.0, np.hypot(e, fl) / 1000.0))
    return rows


# ---------------------------------------------------------------- prescriptions
PRESCRIPTIONS = {
    #  name     extra parameter names, grid of starting values (coarse pre-scan), bounds
    "pert":   dict(names=[], grid=[[]], lo=[], hi=[]),
    "gauss":  dict(names=["sigma"], grid=[[0.7], [1.5], [3.0], [6.0]], lo=[0.2], hi=[12.0]),
    "gi":     dict(names=["sigma0", "s"], grid=[[1.8, 1.55], [0.5, 1.5], [3.0, 0.5], [3.0, 2.0], [6.0, 0.5]],
                   lo=[0.2, 0.0], hi=[12.0, 6.0]),
    "yukawa": dict(names=["m"], grid=[[1.5], [2.47], [4.0], [8.0]], lo=[0.3], hi=[20.0]),
}
COUPLING = {
    "fixed": dict(names=["alpha_s"], grid=[[0.25], [0.35], [0.5]], lo=[0.05], hi=[1.5]),
    "run":   dict(names=["Lambda"], grid=[[0.15], [0.25], [0.40]], lo=[0.05], hi=[0.95]),
}


def alpha_s_of(coupling: str, c: float, m1: float, m2: float) -> float:
    if coupling == "fixed":
        return c
    mu = np.sqrt(m1 * m2)
    return 12.0 * np.pi / (25.0 * np.log(mu**2 / c**2))


def profile(presc: str, p: np.ndarray, m1: float, m2: float):
    """Normalised smearing function δ_X(r) (GeV³) for the smeared prescriptions."""
    if presc == "gauss":
        sig = p[0]
    elif presc == "gi":
        s0, s = p
        mu2 = 2.0 * m1 * m2 / (m1 + m2)
        sig = np.sqrt(s0**2 * (0.5 + 0.5 * (4 * m1 * m2 / (m1 + m2)**2)**4) + s**2 * mu2**2)
    elif presc == "yukawa":
        m = p[0]
        return lambda r: m**2 * np.exp(-m * r) / (4.0 * np.pi * np.where(r > 0, r, np.inf))
    else:
        raise ValueError(presc)
    pref = (sig / np.sqrt(np.pi)) ** 3
    return lambda r: pref * np.exp(-(sig * r) ** 2)


def width_of(presc: str, p: np.ndarray, m1: float, m2: float) -> float:
    """A characteristic width (GeV⁻¹) of the profile, for reporting: the r.m.s. radius."""
    if presc == "pert":
        return 0.0
    if presc == "yukawa":
        return np.sqrt(6.0) / p[0]           # <r²> = 6/m² for the Yukawa profile
    f = profile(presc, p, m1, m2)
    r = np.linspace(0, 20, 20001)
    w = f(r) * 4 * np.pi * r**2
    return float(np.sqrt(np.trapezoid(w * r**2, r) / np.trapezoid(w, r)))


@dataclass
class HyperfineModel:
    presc: str
    coupling: str
    base: Potential            # spin-averaged potential with its masses
    kappa: float
    x: np.ndarray              # [coupling parameter, prescription parameters...]

    def alpha(self, m1, m2):
        return alpha_s_of(self.coupling, self.x[0], m1, m2)

    def splitting(self, pair: str, lab: str, h: float = 2e-3) -> float:
        """Hyperfine observable (GeV): S-wave triplet − singlet; P-wave singlet − triplet centroid."""
        q = PAIRS[pair]
        m1, m2 = self.base.mass(q[0]), self.base.mass(q[1])
        n, L = int(lab[0]), "SPDF".index(lab[1])
        a = self.alpha(m1, m2)
        if self.presc == "pert":
            if L > 0:
                return 0.0
            st = solve(self.base, m1, m2, n, 0, r_max=R_MAX[0][n], h=h)
            return 8.0 * a / (9.0 * m1 * m2) * st.R_at_origin_sq()
        f = profile(self.presc, self.x[1:], m1, m2)
        pref = 32.0 * np.pi * a / (9.0 * m1 * m2)
        E = {}
        for name, ss in SPIN.items():
            st = solve(self.base, m1, m2, n, L, r_max=R_MAX[L][n], h=h, extra_V=lambda r, k=pref * ss: k * f(r))
            E[name] = st.E
        if L == 0:
            return E["triplet"] - E["singlet"]
        # the triplet centroid carries the +1/4 shift; h − centroid = E_singlet − E_triplet
        return E["singlet"] - E["triplet"]

    def predict(self, labels=None, h: float = 2e-3) -> dict:
        labels = labels or [(p, l) for p, l, *_ in HYPERFINE]
        return {(p, l): self.splitting(p, l, h) for p, l in labels}


@dataclass
class HyperfineFit:
    presc: str
    coupling: str
    kappa: float
    names: list
    x: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    residuals: list = field(default_factory=list)   # (pair, label, exp, model, pull)
    extra: dict = field(default_factory=dict)       # predictions not in the fit
    seconds: float = 0.0
    floors: tuple = (3.0, 0.5)

    def to_json(self, path):
        d = dataclasses.asdict(self)
        d["x"] = self.x.tolist(); d["cov"] = self.cov.tolist()
        d["residuals"] = [dict(pair=p, label=l, exp=e, model=m, pull=q) for p, l, e, m, q in self.residuals]
        d["extra"] = {f"{p} {l}": v for (p, l), v in self.extra.items()}
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path):
        d = json.loads(Path(path).read_text())
        d["x"] = np.array(d["x"]); d["cov"] = np.array(d["cov"])
        d["residuals"] = [(r["pair"], r["label"], r["exp"], r["model"], r["pull"]) for r in d["residuals"]]
        d["extra"] = {tuple(k.split()): v for k, v in d["extra"].items()}
        d["floors"] = tuple(d["floors"])
        return HyperfineFit(**d)


def base_potential(kappa: float = 0.0, results_dir="results") -> Potential:
    """The centroid-extracted spline potential (companion paper) at the given κ."""
    tag = "" if kappa == 0.0 else f"__kappa{kappa:g}"
    return SplineFit.from_json(Path(results_dir) / f"centroid_K6_lam1{tag}.json").potential()


def fit_hyperfine(presc: str, coupling: str = "run", kappa: float = 0.0, floor_S=3.0, floor_P=0.5,
                  h: float = 2e-3, verbose=True, results_dir="results", exclude=(), cc1S_lattice=False) -> HyperfineFit:
    base = base_potential(kappa, results_dir)
    spec = PRESCRIPTIONS[presc]; cp = COUPLING[coupling]
    names = cp["names"] + spec["names"]
    lo = np.array(cp["lo"] + spec["lo"]); hi = np.array(cp["hi"] + spec["hi"])
    data = dataset(floor_S, floor_P, exclude, cc1S_lattice)
    labels = [(p, l) for p, l, *_ in data]

    def residuals(x):
        m = HyperfineModel(presc, coupling, base, kappa, x)
        try:
            pred = m.predict(labels, h)
        except ValueError:
            return np.full(len(data), 1e3)
        return np.array([(pred[(p, l)] - v) / s for p, l, v, s in data])

    t0 = time.time()
    # coarse pre-scan over a grid of starting points (the smeared prescriptions have
    # local minima in (coupling, width)), then a local least-squares fit from the best
    starts = [np.array(c + q) for c in cp["grid"] for q in spec["grid"]]
    chi = [float(np.sum(residuals(x) ** 2)) for x in starts]
    x0 = starts[int(np.argmin(chi))]
    if verbose:
        print("pre-scan chi2:", [round(c, 1) for c in chi], "-> start", x0)
    r = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3,
                      verbose=2 if verbose else 0, max_nfev=200)
    secs = time.time() - t0
    cov = np.linalg.pinv(r.jac.T @ r.jac)
    m = HyperfineModel(presc, coupling, base, kappa, r.x)
    pred = m.predict(labels, h)
    rows = [(p, l, v, pred[(p, l)], (pred[(p, l)] - v) / s) for p, l, v, s in data]
    extra = m.predict([("cb", "1P"), ("cc", "3S"), ("bb", "3S"), ("cb", "3S")] + [(p, l) for p, l, *_ in HYPERFINE if f"{p}{l}" in exclude], h)
    return HyperfineFit(presc, coupling, kappa, names, r.x, cov, float(np.sum(r.fun**2)), len(data) - len(r.x),
                        rows, extra, secs, (floor_S, floor_P))
