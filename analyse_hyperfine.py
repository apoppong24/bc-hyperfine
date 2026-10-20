#!/usr/bin/env python3
"""Compare the contact-interaction prescriptions.

Reads results/hf_*.json, writes results/hyperfine_analysis.{md,json} and
paper/figures/fig1_pulls.pdf, fig2_profiles.pdf, fig3_pwave_bound.pdf.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from quarkonium.hyperfine import (HYPERFINE, LATTICE, LATTICE_RATIO_BB, HyperfineFit, HyperfineModel,
                                  base_potential, profile, width_of)
from quarkonium.solver import solve

RES = Path("results"); FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
HBARC = 0.197327          # GeV fm
ORDER = ["pert", "gauss", "gi", "yukawa"]
NAME = {"pert": "first-order", "gauss": "Gaussian", "gi": "Godfrey–Isgur", "yukawa": "Yukawa"}
OBS = [(p, l) for p, l, *_ in HYPERFINE]
M_BC, M_BC2S = 6274.5, 6871.2


def load():
    fits = {}
    for p in sorted(glob.glob(str(RES / "hf_*.json"))):
        f = HyperfineFit.from_json(p)
        fits[Path(p).stem[3:]] = f          # e.g. pert_run_k0.2, gi_run_k0.4_nocc2S
    return fits


def masses(f):
    base = base_potential(f.kappa)
    return base.mass("c"), base.mass("b"), base


def summarise(f: HyperfineFit):
    m_c, m_b, base = masses(f)
    pred = {(p, l): m for p, l, _, m, _ in f.residuals}
    pred.update({k: v for k, v in f.extra.items() if k not in pred})     # excluded observables are predicted, not fitted
    d = dict(presc=f.presc, coupling=f.coupling, kappa=f.kappa, chi2=f.chi2, dof=f.dof,
             params={n: (float(x), float(e)) for n, x, e in zip(f.names, f.x, np.sqrt(f.cov.diagonal()))},
             pred={f"{p} {l}": 1000 * v for (p, l), v in pred.items()},
             pulls={f"{p} {l}": q for p, l, _, _, q in f.residuals},
             extra={f"{p} {l}": 1000 * v for (p, l), v in f.extra.items()})
    # widths (fm) for each system
    d["width_fm"] = {pair: HBARC * width_of(f.presc, f.x[1:], *mm) for pair, mm in (("cc", (m_c, m_c)), ("bb", (m_b, m_b)), ("cb", (m_c, m_b)))}
    d["alpha"] = {pair: float(HyperfineModel(f.presc, f.coupling, base, f.kappa, f.x).alpha(*mm))
                  for pair, mm in (("cc", (m_c, m_c)), ("bb", (m_b, m_b)), ("cb", (m_c, m_b)))}
    # parametric errors on the B_c splittings (finite-difference Jacobian × fit covariance)
    labs = [("cb", "1S"), ("cb", "2S")]
    J = np.zeros((2, len(f.x)))
    for j in range(len(f.x)):
        dx = 1e-2 * max(abs(f.x[j]), 0.1)
        xp, xm = f.x.copy(), f.x.copy(); xp[j] += dx; xm[j] -= dx
        pp = HyperfineModel(f.presc, f.coupling, base, f.kappa, xp).predict(labs)
        pm = HyperfineModel(f.presc, f.coupling, base, f.kappa, xm).predict(labs)
        J[:, j] = [(pp[l] - pm[l]) / (2 * dx) for l in labs]
    cv = J @ f.cov @ J.T * 1e6
    scale = np.sqrt(max(f.chi2 / max(f.dof, 1), 1.0))
    d["bc_err"] = dict(d1S=scale * float(np.sqrt(cv[0, 0])), d2S=scale * float(np.sqrt(cv[1, 1])),
                       d1_minus_d2=scale * float(np.sqrt(max(cv[0, 0] + cv[1, 1] - 2 * cv[0, 1], 0))), scale=float(scale))
    d["bc"] = dict(M_Bcstar=M_BC + 1000 * pred[("cb", "1S")], M_Bcstar2S=M_BC2S + 1000 * pred[("cb", "2S")],
                   d1_minus_d2=1000 * (pred[("cb", "1S")] - pred[("cb", "2S")]),
                   ratio_cc=pred[("cc", "2S")] / pred[("cc", "1S")], ratio_bb=pred[("bb", "2S")] / pred[("bb", "1S")],
                   ratio_cb=pred[("cb", "2S")] / pred[("cb", "1S")])
    return d


def pwave_bound(kappa=0.0):
    """How large a smearing width the h_c − centroid(χ_c) datum allows: for a Gaussian
    of width σ with α_s tuned to reproduce the cc̄ 1S splitting, the predicted cc̄ 1P
    singlet–centroid shift versus σ (and the r.m.s. radius in fm)."""
    base = base_potential(kappa)
    out = []
    for sig in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        a = 0.3
        for _ in range(6):                             # fixed-point iteration: α_s such that the cc̄ 1S datum is reproduced
            m = HyperfineModel("gauss", "fixed", base, kappa, np.array([a, sig]))
            d1 = m.splitting("cc", "1S")
            if abs(d1 - 0.11281) < 2e-5:
                break
            a *= 0.11281 / d1
        dP = m.splitting("cc", "1P"); d2 = m.splitting("cc", "2S")
        out.append(dict(sigma=sig, rms_fm=HBARC * np.sqrt(1.5) / sig, alpha=a, cc1S=1000 * d1, cc2S=1000 * d2, cc1P=1000 * dP))
    return out


def main():
    fits = load()
    print("fits:", list(fits))
    S = {k: summarise(f) for k, f in fits.items()}
    md = ["# Contact-interaction prescriptions vs the hyperfine data\n",
          "Spin-averaged potential and quark masses fixed from the centroid extraction (companion paper); "
          "theory floors 3 MeV (S) / 0.5 MeV (P) in quadrature.\n"]
    md.append("## Fit summary\n\n| fit | χ² | dof | parameters | α_s(cc̄) / α_s(bb̄) | width cc̄ / bb̄ [fm] | M(B_c*) | M(B_c*(2S)) | Δ1S−Δ2S |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(S, key=lambda k: (S[k]["kappa"], S[k]["coupling"], ORDER.index(S[k]["presc"]))):
        d = S[k]
        pars = ", ".join(f"{n}={v:.3f}±{e:.3f}" for n, (v, e) in d["params"].items())
        md.append(f"| {k} | {d['chi2']:.1f} | {d['dof']} | {pars} | {d['alpha']['cc']:.3f} / {d['alpha']['bb']:.3f} | "
                  f"{d['width_fm']['cc']:.3f} / {d['width_fm']['bb']:.3f} | {d['bc']['M_Bcstar']:.1f} ± {d['bc_err']['d1S']:.1f} | {d['bc']['M_Bcstar2S']:.1f} ± {d['bc_err']['d2S']:.1f} | {d['bc']['d1_minus_d2']:.1f} ± {d['bc_err']['d1_minus_d2']:.1f} |")
    md.append("\nExperiment: M(B_c*) − M(B_c) = 64.5 ± 1.8 (ATLAS 2026), Δ1S−Δ2S = 29.1 ± 1.7 (CMS 2019), 30.9 (LHCb 2019); "
              "lattice M(B_c*) − M(B_c) = 54 ± 3 (HPQCD 2012), 55 ± 3 (Mathur et al. 2018); bb̄ 1S lattice 57.5 ± 2.5 (HPQCD 2021), 60.0 ± 6.4 (HPQCD NRQCD 2014); "
              "cc̄ 1S lattice connected 120.3 ± 1.1 (HPQCD 2020).\n")
    md.append("## Predictions (MeV) for every observable\n")
    md.append("| fit | " + " | ".join(f"{p} {l}" for p, l in OBS) + " |")
    md.append("|---|" + "---|" * len(OBS))
    md.append("| **experiment** | " + " | ".join(f"{v:.1f}±{e:.1f}" for _, _, v, e, _ in HYPERFINE) + " |")
    for k in sorted(S, key=lambda k: (S[k]["kappa"], S[k]["coupling"], ORDER.index(S[k]["presc"]))):
        d = S[k]
        md.append(f"| {k} | " + " | ".join(f"{d['pred'][f'{p} {l}']:.1f}" + (f" ({d['pulls'][f'{p} {l}']:+.1f})" if f"{p} {l}" in d["pulls"] else " (pred.)") for p, l in OBS) + " |")
    md.append("\n## 2S/1S ratios\n\n| fit | cc̄ | bb̄ | cb̄ |\n|---|---|---|---|")
    md.append(f"| experiment | {48.30/112.81:.3f} | {24.4/61.7:.3f} ± {np.hypot(4.2/61.7, 24.4*2/61.7**2):.3f} | {35.4/64.5:.3f} |")
    md.append(f"| lattice (HPQCD 2014) | – | {LATTICE_RATIO_BB[0]} ± {LATTICE_RATIO_BB[1]} | – |")
    for k in sorted(S, key=lambda k: (S[k]["kappa"], S[k]["coupling"], ORDER.index(S[k]["presc"]))):
        d = S[k]["bc"]
        md.append(f"| {k} | {d['ratio_cc']:.3f} | {d['ratio_bb']:.3f} | {d['ratio_cb']:.3f} |")
    # κ scan: χ² of each (prescription, coupling) against the Coulomb strength of the base potential
    md.append("\n## χ² versus the Coulomb strength κ of the spin-averaged potential\n")
    kappas = sorted({d["kappa"] for d in S.values()})
    md.append("| prescription / coupling | " + " | ".join(f"κ={k:g}" for k in kappas) + " |\n|---|" + "---|" * len(kappas))
    kscan = {}
    for presc in ORDER:
        for coup in ("run", "fixed"):
            row = {k: S[f"{presc}_{coup}_k{k:g}"]["chi2"] for k in kappas if f"{presc}_{coup}_k{k:g}" in S}
            if row:
                kscan[f"{presc}_{coup}"] = row
                md.append(f"| {NAME[presc]} / {coup} | " + " | ".join(f"{row[k]:.1f}" if k in row else "–" for k in kappas) + " |")
    md.append("\n(The spin-averaged spectrum itself is blind to κ: its χ² is 12.0–12.2 for every value.)\n")
    # without the cc 2S
    md.append("\n## Without the charmonium 2S splitting\n\n| fit | χ² | dof | parameters | predicted cc̄ 2S | M(B_c*) | Δ1S−Δ2S |\n|---|---|---|---|---|---|---|")
    for k in sorted(k for k in S if "nocc2S" in k):
        d = S[k]
        pars = ", ".join(f"{n}={v:.3f}±{e:.3f}" for n, (v, e) in d["params"].items())
        md.append(f"| {k} | {d['chi2']:.1f} | {d['dof']} | {pars} | {d['extra'].get('cc 2S', float('nan')):.1f} | {d['bc']['M_Bcstar']:.1f} ± {d['bc_err']['d1S']:.1f} | {d['bc']['d1_minus_d2']:.1f} ± {d['bc_err']['d1_minus_d2']:.1f} |")
    md.append("")
    pb = {k: pwave_bound(k) for k in (0.0, 0.4)}
    for k, rows in pb.items():
        md.append(f"\n## P-wave bound on the smearing (Gaussian, α_s tuned to the cc̄ 1S splitting, κ={k:g})\n")
        md.append("| σ [GeV] | r.m.s. [fm] | α_s | cc̄ 1S | cc̄ 2S | h_c − centroid |\n|---|---|---|---|---|---|")
        for q in rows:
            md.append(f"| {q['sigma']} | {q['rms_fm']:.3f} | {q['alpha']:.3f} | {q['cc1S']:.1f} | {q['cc2S']:.1f} | {q['cc1P']:+.2f} |")
    md.append("\nMeasured h_c − centroid = +0.07 ± 0.15 MeV.\n")

    text = "\n".join(md)
    (RES / "hyperfine_analysis.md").write_text(text)
    (RES / "hyperfine_analysis.json").write_text(json.dumps(dict(fits=S, pwave_bound={str(k): v for k, v in pb.items()}, kscan=kscan, obs=[list(o) for o in HYPERFINE]), indent=1))
    print(text)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm", "legend.fontsize": 7.5})
    lab = {"cc 1S": r"$c\bar c\,1S$", "cc 2S": r"$c\bar c\,2S$", "bb 1S": r"$b\bar b\,1S$", "bb 2S": r"$b\bar b\,2S$",
           "cb 1S": r"$B_c\,1S$", "cb 2S": r"$B_c\,2S$", "cc 1P": r"$h_c$", "bb 1P": r"$h_b(1P)$", "bb 2P": r"$h_b(2P)$"}
    MAIN = [("pert_run_k0.2", "first-order, running"), ("pert_fixed_k0", "first-order, fixed"),
            ("gauss_fixed_k0.8", "Gaussian, fixed"), ("gi_run_k0.2", "Godfrey–Isgur, running"), ("yukawa_fixed_k0.8", "Yukawa, fixed")]
    MAIN = [(k, n) for k, n in MAIN if k in S]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    nb = len(MAIN)
    for i, (k, name) in enumerate(MAIN):
        pulls = [S[k]["pulls"][f"{p} {l}"] for p, l in OBS]
        ax.bar(np.arange(len(OBS)) + (i - (nb - 1) / 2) * 0.8 / nb, pulls, 0.8 / nb, label=f"{name} ($\\kappa={S[k]['kappa']:g}$, $\\chi^2={S[k]['chi2']:.0f}$)")
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-1, 1, color="0.9", zorder=0)
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([lab[f"{p} {l}"] for p, l in OBS], fontsize=8)
    ax.set_ylabel("pull (model − exp.)/$\\sigma$"); ax.legend(loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig1_pulls.pdf"); plt.close(fig)

    # profiles vs wave functions
    r = np.linspace(1e-3, 3.0, 600)
    base = base_potential(0.0)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for pair, ls in (("cc", "-"), ("bb", "--")):
        m = base.mass(pair[0])
        st = solve(base, m, m, 1, 0, r_max=10, h=2e-3)
        ax.plot(st.r, st.u**2 / st.u.max()**2, "k" + ls, lw=0.8, label=f"$|u_{{1S}}|^2$ {pair[0]}$\\bar {pair[0]}$ (norm.)")
    for presc, c in (("gauss", "C0"), ("gi", "C1"), ("yukawa", "C2")):
        k = f"{presc}_fixed_k0"
        if k not in fits:
            continue
        f = fits[k]
        for pair, ls in (("cc", "-"), ("bb", "--")):
            m = base.mass(pair[0]); pr = profile(presc, f.x[1:], m, m)
            w = pr(r) * 4 * np.pi * r**2
            ax.plot(r, w / w.max(), color=c, ls=ls, lw=1, label=(f"{NAME[presc]} ({pair[0]}$\\bar {pair[0]}$)" if pair == "cc" else None))
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.05); ax.set_xlabel(r"$r$ [GeV$^{-1}$]"); ax.set_ylabel("normalised $4\\pi r^2\\,\\delta_X(r)$")
    ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(FIG / "fig2_profiles.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for k, c in ((0.0, "C0"), (0.4, "C1")):
        ax.plot([q["rms_fm"] for q in pb[k]], [-q["cc1P"] for q in pb[k]], "o-", color=c, label=f"Gaussian contact term, $\\kappa={k:g}$")
    ax.axhspan(-0.07 - 0.15, -0.07 + 0.15, color="C3", alpha=0.3, label="measured $\\pm1\\sigma$")
    ax.set_yscale("log"); ax.set_xlim(0, 0.5); ax.set_xlabel("r.m.s. radius of the contact term [fm]"); ax.set_ylabel("[MeV]")
    ax.set_ylim(0.03, 60); ax.set_title(r"$M(\mathrm{centroid}\,\chi_{cJ})-M(h_c)$", fontsize=9); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fig3_pwave_bound.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for key, row in kscan.items():
        presc, coup = key.split("_")
        ks = sorted(row); ax.plot(ks, [row[k] for k in ks], ("o-" if coup == "run" else "s--"),
                                  label=f"{NAME[presc]}, {'running' if coup == 'run' else 'fixed'} $\\alpha_s$")
    ax.set_yscale("log"); ax.set_xlabel(r"Coulomb strength $\kappa$ of $V(r)$"); ax.set_ylabel(r"$\chi^2$ (9 hyperfine data)")
    ax.legend(fontsize=6, ncol=1); fig.tight_layout(); fig.savefig(FIG / "fig4_kappa_scan.pdf"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
