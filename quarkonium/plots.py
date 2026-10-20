"""Figures from the thesis, regenerated from solved spectra.

    potentials_plot()          Fig. 3-style overlay of the three potentials
    trial_energy_plot()        Fig. 8   accepted vs rejected trial energies
    wavefunction_plot()        Fig. 9   radial wavefunctions per potential
    splitting_plot()           Figs. 10-12  level splittings vs potential model
"""
from __future__ import annotations

import numpy as np

from .observables import splittings
from .potentials import CORNELL, LOGARITHMIC, MARTIN, Potential
from .solver import RadialState, _effective_potential, _shoot, reduced_mass


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def potentials_plot(path, r_min=0.1, r_max=5.0):
    plt = _plt()
    r = np.linspace(r_min, r_max, 2000)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for P in (LOGARITHMIC, MARTIN, CORNELL):
        ax.plot(r, P(r), label=P.label)
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("r [GeV$^{-1}$]"); ax.set_ylabel("V(r) [GeV]")
    ax.set_title("Quark–antiquark potentials (Eichten & Quigg parameters)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def trial_energy_plot(path, potential: Potential = LOGARITHMIC, m1=None, m2=None,
                      E_exact=None, dE=1e-3, r_max=5.0, h=1e-3):
    """Fig. 8: wavefunction for E_exact and for E_exact ± dE, showing the
    tail diverging in opposite directions on either side of the eigenvalue."""
    plt = _plt()
    m1 = potential.m_b if m1 is None else m1
    m2 = potential.m_b if m2 is None else m2
    mu = reduced_mass(m1, m2)
    r = np.arange(0.0, r_max + 0.5 * h, h)
    veff = _effective_potential(potential, r, 0, mu)
    if E_exact is None:
        from .solver import solve
        E_exact = solve(potential, m1, m2, 1, 0, r_max=2 * r_max, h=h).E
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for E, lab in ((E_exact + dE, "too high (discard)"), (E_exact - dE, "too low (discard)"),
                   (E_exact, "eigenvalue (accept)")):
        u = _shoot(veff, mu, E, h, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(r > 0, u / r, u[1] / h)
        R /= R[0]
        ax.plot(r, R, label=f"E = {E:.4f} GeV: {lab}")
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlabel("r [GeV$^{-1}$]"); ax.set_ylabel("R(r) / R(0)")
    ax.set_title(f"Shooting method, {potential.label} potential, 1S")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def wavefunction_plot(path, spectra: dict[str, dict[str, RadialState]], pname: str):
    """Fig. 9: spectra = {'bb': {...}, 'cc': {...}, 'cb': {...}} for one potential."""
    plt = _plt()
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    titles = {"bb": "Bottomonium", "cc": "Charmonium", "cb": "$B_c$"}
    for row, pair in enumerate(("bb", "cc", "cb")):
        for col, labs in enumerate((("1S", "2S", "3S"), ("1P", "2P"))):
            ax = axes[row, col]
            for lab in labs:
                st = spectra[pair][lab]
                ax.plot(st.r, st.R, label=lab)
            ax.axhline(0, color="k", lw=0.6, ls="--")
            ax.set_xlim(0, 12)
            ax.set_title(f"{titles[pair]} {'S' if col == 0 else 'P'} states, {pname}")
            ax.set_xlabel("r [GeV$^{-1}$]"); ax.set_ylabel("R(r)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def splitting_plot(path, by_potential: dict[str, dict[str, RadialState]], title: str,
                   pdg: dict[str, float] | None = None):
    """Figs. 10–12: level splittings for each potential model, with PDG
    values as horizontal lines.  by_potential = {'Power-law': spectrum, ...}."""
    plt = _plt()
    pdg = pdg or {}
    names = list(by_potential)
    sp = {p: splittings({lab: st.mass for lab, st in S.items()}) for p, S in by_potential.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    for ax, keys in ((a, ("2S-1S", "1P-1S", "2P-1P")), (b, ("3S-1S", "2P-1S"))):
        for k, marker, color in zip(keys, "sod", ("C0", "C1", "C2")):
            y = np.array([sp[p][k] for p in names])
            # error bar = deviation of each model from the mean over the three
            # potential models (a model-dependence estimate)
            err = np.abs(y - y.mean())
            ax.errorbar(x, y, yerr=err, fmt=marker, ms=7, mfc="none", color=color,
                        capsize=4, label=k)
            if k in pdg:
                ax.axhline(pdg[k], lw=1, color=color, label=f"PDG({k})")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_xlabel("Potential model"); ax.set_ylabel("Energy splitting [GeV]")
        ax.legend(fontsize=8)
    fig.suptitle(f"Level splittings of {title}  (bars: deviation from the three-model mean)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
