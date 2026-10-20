"""Refit potential parameters, quark masses and α_s to the PDG 2026
charmonium + bottomonium spectrum (J-resolved, including fine structure).

Model:  M(n ²ˢ⁺¹L_J) = m1 + m2 + E_nl + Δ_SD(α_s)

Free parameters per family
    Martin   : B, α_exp, m_c, m_b, α_s     (A fixed: degenerate with masses)
    Log      : C, m_c, m_b, α_s            (constant fixed, same reason)
    Cornell  : κ, a, m_c, m_b, α_s

χ² uses σ² = σ_exp² + σ_th², with σ_th a theory floor (default 10 MeV) so
the fit is not dominated by the keV-precision J/ψ and ψ(2S) masses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import pdg
from .potentials import Potential
from .solver import solve
from .spin_dependent import fine_structure

R_MAX = {0: {1: 10, 2: 14, 3: 18, 4: 22}, 1: {1: 14, 2: 18, 3: 22}, 2: {1: 18, 2: 22}}


# ---------------------------------------------------------------- families
def make_martin(B, alpha, m_c, m_b, alpha_s, A=-8.064):
    return Potential("martin", lambda r: A + B * r**alpha, m_c, m_b, alpha_s, "Power-law (Martin)")


def make_log(C, m_c, m_b, alpha_s, D=-0.6635):
    return Potential("log", lambda r: D + C * np.log(r), m_c, m_b, alpha_s, "Logarithmic")


def make_cornell(kappa, a, m_c, m_b, alpha_s):
    return Potential("cornell", lambda r: -kappa / r + r / a**2, m_c, m_b, alpha_s, "Cornell")


FAMILIES = {
    "martin": dict(make=make_martin, names=["B", "alpha", "m_c", "m_b", "alpha_s"],
                   x0=[6.898, 0.1, 1.8, 5.174, 0.43],
                   bounds=([1, 0.01, 1.0, 4.0, 0.1], [20, 0.6, 2.5, 6.0, 0.8])),
    "log": dict(make=make_log, names=["C", "m_c", "m_b", "alpha_s"],
                x0=[0.733, 1.5, 4.906, 0.37],
                bounds=([0.3, 1.0, 4.0, 0.1], [1.5, 2.5, 6.0, 0.8])),
    "cornell": dict(make=make_cornell, names=["kappa", "a", "m_c", "m_b", "alpha_s"],
                    x0=[0.52, 2.34, 1.84, 5.18, 0.31],
                    bounds=([0.1, 1.0, 1.0, 4.0, 0.1], [1.2, 4.0, 2.5, 6.0, 0.8])),
}


# ---------------------------------------------------------------- data
def _parse(label):
    """'13P2' -> (n=1, S=1, L=1, J=2); '11P1' -> (1,0,1,1)."""
    lab = label.rstrip("'")
    if len(lab) == 3:                       # mixed J=L state of B_c, e.g. "1P1"
        return int(lab[0]), None, "SPDF".index(lab[1]), int(lab[2])
    n, s2, L, J = int(lab[0]), int(lab[1]), "SPDF".index(lab[2]), int(lab[3])
    return n, (s2 - 1) // 2, L, J


def dataset(sigma_th=10.0, exclude=()):
    """Rows (pair, label, mass GeV, sigma GeV).  `exclude` is a list of
    substrings; a level whose label contains any of them is dropped
    (e.g. ["D"] removes the D waves, ["3P"] the 3P levels)."""
    rows = []
    for pair, table in (("cc", pdg.CHARMONIUM), ("bb", pdg.BOTTOMONIUM)):
        for lab, (m, s) in table.items():
            if any(x in lab for x in exclude):
                continue
            rows.append((pair, lab, m / 1000.0, np.hypot(s, sigma_th) / 1000.0))
    return rows


# ---------------------------------------------------------------- model
def predict(P: Potential, pair: str, labels, h=2e-3):
    """Masses (GeV) for the requested J-resolved labels of one flavour pair.

    With P.hyperfine == "smeared" the S-wave singlet and triplet are obtained
    by solving the radial equation with the smeared contact term included
    (self-consistently), instead of adding the first-order T3 shift."""
    q = {"cc": ("c", "c"), "bb": ("b", "b"), "cb": ("c", "b")}[pair]
    m1, m2 = P.mass(q[0]), P.mass(q[1])
    need = sorted({(_parse(l)[0], _parse(l)[2]) for l in labels})
    out = {}
    for n, L in need:
        if L == 0 and P.hyperfine == "smeared":
            for S, ss, lab in ((1, 0.25, f"{n}3S1"), (0, -0.75, f"{n}1S0")):
                st = solve(P, m1, m2, n, 0, r_max=R_MAX[0][n], h=h, extra_V=P.contact_term(m1, m2, ss))
                out[lab] = st.mass
            continue
        st = solve(P, m1, m2, n, L, r_max=R_MAX[L][n], h=h)
        for f in fine_structure(st):
            out[f.label] = f.mass
    return {l: out[l] for l in labels}


@dataclass
class FitResult:
    family: str
    names: list
    x: np.ndarray
    cov: np.ndarray | None
    chi2: float
    dof: int
    residuals: list = field(default_factory=list)   # (pair, label, exp, model, pull)
    seconds: float = 0.0

    @property
    def errors(self):
        return np.sqrt(np.diag(self.cov)) if self.cov is not None else np.full(len(self.x), np.nan)

    def potential(self):
        return FAMILIES[self.family]["make"](*self.x)

    def summary(self):
        lines = [f"{self.family}: χ²/dof = {self.chi2:.1f}/{self.dof}  ({self.seconds:.0f} s)"]
        for n, v, e in zip(self.names, self.x, self.errors):
            lines.append(f"   {n:8s} = {v:.4f} ± {e:.4f}")
        return "\n".join(lines)


def fit(family: str, sigma_th=10.0, h=2e-3, verbose=True, exclude=()) -> FitResult:
    spec = FAMILIES[family]
    data = dataset(sigma_th, exclude)
    by_pair = {}
    for pair, lab, m, s in data:
        by_pair.setdefault(pair, []).append((lab, m, s))

    def residuals(x):
        P = spec["make"](*x)
        res = []
        for pair, rows in by_pair.items():
            try:
                pred = predict(P, pair, [r[0] for r in rows], h=h)
            except ValueError:
                return np.full(len(data), 1e3)
            res += [(pred[lab] - m) / s for lab, m, s in rows]
        return np.array(res)

    t0 = time.time()
    r = least_squares(residuals, spec["x0"], bounds=spec["bounds"], x_scale="jac",
                      diff_step=1e-3, verbose=2 if verbose else 0)
    secs = time.time() - t0
    J = r.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None
    chi2 = float(np.sum(r.fun**2))
    dof = len(data) - len(r.x)
    P = spec["make"](*r.x)
    rows_out = []
    for pair, rows in by_pair.items():
        pred = predict(P, pair, [q[0] for q in rows], h=h)
        rows_out += [(pair, lab, m, pred[lab], (pred[lab] - m) / s) for lab, m, s in rows]
    return FitResult(family, spec["names"], r.x, cov, chi2, dof, rows_out, secs)


def load_fit(family: str, results_dir=None) -> Potential:
    """Rebuild the refitted Potential from results/fit_<family>.json."""
    import json
    from pathlib import Path
    d = Path(results_dir) if results_dir else Path(__file__).resolve().parent.parent / "results"
    r = json.loads((d / f"fit_{family}.json").read_text())
    return FAMILIES[family]["make"](*r["x"])


# ---------------------------------------------------------------- running alpha_s
def make_martin_running(B, alpha, m_c, m_b, lam, scale, A=-8.064):
    return Potential("martin_run", lambda r: A + B * r**alpha, m_c, m_b, 0.0,
                     f"Martin, running α_s ({scale})", alpha_s_scale=scale, lambda_qcd=lam)


def make_log_running(C, m_c, m_b, lam, scale, D=-0.6635):
    return Potential("log_run", lambda r: D + C * np.log(r), m_c, m_b, 0.0,
                     f"Log, running α_s ({scale})", alpha_s_scale=scale, lambda_qcd=lam)


for _scale in ("reduced", "geometric"):
    FAMILIES[f"martin_run_{_scale}"] = dict(
        make=(lambda *x, _s=_scale: make_martin_running(*x, scale=_s)),
        names=["B", "alpha", "m_c", "m_b", "Lambda"], x0=[7.19, 0.096, 1.63, 5.02, 0.35],
        bounds=([1, 0.01, 1.0, 4.0, 0.05], [20, 0.6, 2.5, 6.0, 0.9]))
    FAMILIES[f"log_run_{_scale}"] = dict(
        make=(lambda *x, _s=_scale: make_log_running(*x, scale=_s)),
        names=["C", "m_c", "m_b", "Lambda"], x0=[0.737, 1.51, 4.91, 0.35],
        bounds=([0.3, 1.0, 4.0, 0.05], [1.5, 2.5, 6.0, 0.9]))


# ---------------------------------------------------------------- self-consistent hyperfine
def _sc(make):
    """Wrap a family's `make` so the Potential uses the smeared hyperfine."""
    import dataclasses

    def make_sc(*x, _sigma=1.0, **kw):
        return dataclasses.replace(make(*x, **kw), hyperfine="smeared", smear_sigma=_sigma)
    return make_sc


for _base in ("martin", "log", "martin_run_geometric", "log_run_geometric"):
    FAMILIES[_base + "_sc"] = dict(FAMILIES[_base], make=_sc(FAMILIES[_base]["make"]))
for _sig in (0.8, 1.2):
    _m = FAMILIES["martin_run_geometric"]["make"]
    FAMILIES[f"martin_run_geometric_sc_s{_sig:g}"] = dict(
        FAMILIES["martin_run_geometric"],
        make=(lambda *x, _mm=_m, _s=_sig: _sc(_mm)(*x, _sigma=_s)))


# ---------------------------------------------------------------- exponent scan
def register_power_family(alpha: float, V0: float = -1.166) -> str:
    """Add a power-law family with the exponent held fixed,
        V = V0 + B (r^alpha - 1),
    i.e. the potential is pinned at r = 1 GeV^-1 (V0 = Martin's value there,
    -8.064 + 6.898) so that changing alpha does not shift the whole potential
    by GeVs that the bounded quark masses would then have to absorb.
    alpha = 0.1 reproduces Martin exactly; alpha -> 0 tends to the log
    potential with C = B*alpha.  Free: B, m_c, m_b, alpha_s."""
    key = f"power_{alpha:g}"
    if key in FAMILIES:
        return key
    B0 = 0.733 / alpha            # matches the log-potential slope at r = 1

    def make(B, m_c, m_b, alpha_s, _a=alpha, _V0=V0):
        return Potential(key, lambda r: _V0 + B * (r**_a - 1.0), m_c, m_b, alpha_s, f"Power-law α={_a:g}")

    FAMILIES[key] = dict(make=make, names=["B", "m_c", "m_b", "alpha_s"],
                         x0=[B0, 1.6, 5.0, 0.45],
                         bounds=([0.05, 1.0, 4.0, 0.1], [200, 2.5, 6.0, 0.8]))
    return key
