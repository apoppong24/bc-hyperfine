#!/usr/bin/env python3
"""Regenerate paper/tables/*.tex from results/hyperfine_analysis.json.  Run from the repo root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "results" / "hyperfine_analysis.json").read_text())
OUT = ROOT / "paper" / "tables"; OUT.mkdir(exist_ok=True)
S = A["fits"]
NAME = {"pert": "first-order", "gauss": "Gaussian", "gi": "Godfrey--Isgur", "yukawa": "Yukawa"}
COUP = {"run": "running", "fixed": "fixed"}
OBS = [f"{o[0]} {o[1]}" for o in A["obs"]]
LAB = {"cc 1S": r"$c\bar c\,1S$", "cc 2S": r"$c\bar c\,2S$", "bb 1S": r"$b\bar b\,1S$", "bb 2S": r"$b\bar b\,2S$",
       "cb 1S": r"$B_c\,1S$", "cb 2S": r"$B_c\,2S$", "cc 1P": r"$h_c$", "bb 1P": r"$h_b(1P)$", "bb 2P": r"$h_b(2P)$"}


def pname(n):
    return {"alpha_s": r"\alpha_s", "Lambda": r"\Lambda", "sigma": r"\sigma", "sigma0": r"\sigma_0", "s": "s", "m": "m"}[n]


# ---- Table 1: the data
SRC = {"cc 1S": r"$J/\psi-\eta_c$", "cc 2S": r"$\psi(2S)-\eta_c(2S)$", "bb 1S": r"$\Upsilon(1S)-\eta_b$",
       "bb 2S": r"$\Upsilon(2S)-\eta_b(2S)$ (Belle)", "cb 1S": r"$\Bcs-\Bc$, ATLAS", "cb 2S": r"$\Bcs(2S)-\Bc(2S)$, ATLAS$+$CMS",
       "cc 1P": r"$h_c-\bar M(\chi_{cJ})$", "bb 1P": r"$h_b(1P)-\bar M(\chi_{bJ}(1P))$", "bb 2P": r"$h_b(2P)-\bar M(\chi_{bJ}(2P))$"}
rows = []
for pair, lab, v, e, src in A["obs"]:
    rows.append(f"{LAB[f'{pair} {lab}']} & ${v:.2f}\\pm{e:.2f}$ & {SRC[f'{pair} {lab}']} \\\\")
(OUT / "data.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- Table 2: main comparison at the reference κ for each (prescription, coupling): best κ of the scan
MAIN = ["pert_run_k0.2", "pert_fixed_k0", "gauss_fixed_k0.8", "gauss_run_k0", "gi_run_k0.2", "gi_fixed_k0", "yukawa_fixed_k0.8", "yukawa_run_k0"]
rows = []
for k in MAIN:
    d = S[k]
    pars = ", ".join(f"${pname(n)}={v:.2f}\\pm{e:.2f}$" for n, (v, e) in d["params"].items())
    w = d["width_fm"]
    width = "--" if d["presc"] == "pert" else (f"{w['cc']:.3f}" if abs(w["cc"] - w["bb"]) < 1e-3 else f"{w['cc']:.3f} / {w['bb']:.3f}")
    rows.append(f"{NAME[d['presc']]} & {COUP[d['coupling']]} & {d['kappa']:g} & {d['chi2']:.1f}/{d['dof']} & {pars} & {width} \\\\")
(OUT / "main.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- Table 3: predictions for every observable, main fits
rows = ["experiment & " + " & ".join(f"${o[2]:.1f}\\pm{o[3]:.1f}$" if o[3] >= 0.1 else f"${o[2]:.1f}$" for o in A["obs"]) + r" \\", r"\midrule"]
for k in MAIN:
    d = S[k]
    rows.append(f"{NAME[d['presc']]}, {COUP[d['coupling']]} & " + " & ".join(f"{d['pred'][o]:.1f}" for o in OBS) + r" \\")
(OUT / "predictions.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- Table 4: κ scan
ks = A["kscan"]; kappas = sorted({float(k) for row in ks.values() for k in row})
rows = []
for key in ["pert_run", "pert_fixed", "gauss_fixed", "gauss_run", "gi_run", "gi_fixed", "yukawa_fixed", "yukawa_run"]:
    if key not in ks:
        continue
    p, c = key.split("_")
    rows.append(f"{NAME[p]} & {COUP[c]} & " + " & ".join(f"{ks[key][f'{k:g}']:.1f}" if f"{k:g}" in ks[key] else "--" for k in kappas) + r" \\")
(OUT / "kscan.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- Table 5: B_c
rows = []
for k in ["pert_run_k0.2_nocc2S", "pert_run_k0.4_nocc2S", "gauss_fixed_k0.8_nocc2S", "yukawa_fixed_k0.8_nocc2S", "gi_run_k0.2_nocc2S", "pert_run_k0.2_nocc2S_latt"]:
    d = S[k]; b = d["bc"]; e = d["bc_err"]
    latt = ", lattice $c\\bar c\\,1S$" if k.endswith("_latt") else ""
    rows.append(f"{NAME[d['presc']]}, {COUP[d['coupling']]}, $\\kappa={d['kappa']:g}${latt} & {d['chi2']:.1f}/{d['dof']} & "
                f"${b['M_Bcstar']-6274.5:.1f}\\pm{e['d1S']:.1f}$ & ${b['M_Bcstar2S']-6871.2:.1f}\\pm{e['d2S']:.1f}$ & "
                f"${b['d1_minus_d2']:.1f}\\pm{e['d1_minus_d2']:.1f}$ & ${b['M_Bcstar']:.1f}$ & {d['extra']['cc 2S']:.1f} & {d['extra']['cb 1P']:+.1f} \\\\")
(OUT / "bc.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---- Table 6: P-wave bound (κ = 0.4)
rows = []
for q in A["pwave_bound"]["0.4"]:
    rows.append(f"{q['sigma']:g} & {q['rms_fm']:.3f} & {q['alpha']:.2f} & {q['cc2S']:.1f} & ${q['cc1P']:+.2f}$ \\\\")
(OUT / "pwave.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
