#!/usr/bin/env python3
"""python fit_hyperfine.py <presc> <coupling> [--kappa 0] [--floor 3] [--floorP 0.5]
presc in pert|gauss|gi|yukawa, coupling in fixed|run.  Writes results/hf_<presc>_<coupling>_k<kappa>[_floor<F>].json"""
import argparse
from pathlib import Path
from quarkonium.hyperfine import fit_hyperfine

ap = argparse.ArgumentParser()
ap.add_argument("presc"); ap.add_argument("coupling")
ap.add_argument("--kappa", type=float, default=0.0); ap.add_argument("--floor", type=float, default=3.0); ap.add_argument("--floorP", type=float, default=0.5)
ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--lattice-cc1S", action="store_true")
a = ap.parse_args()
tag = (f"_floor{a.floor:g}" if a.floor != 3.0 else "") + ("_no" + "".join(a.exclude) if a.exclude else "") + ("_latt" if a.lattice_cc1S else "")
f = fit_hyperfine(a.presc, a.coupling, a.kappa, a.floor, a.floorP, verbose=False, exclude=tuple(a.exclude), cc1S_lattice=a.lattice_cc1S)
out = Path("results") / f"hf_{a.presc}_{a.coupling}_k{a.kappa:g}{tag}.json"
f.to_json(out)
print(f"{a.presc:7s} {a.coupling:5s} kappa={a.kappa:g}{tag}: chi2={f.chi2:.1f} dof={f.dof} ({f.seconds:.0f}s)")
for n, x, e in zip(f.names, f.x, f.cov.diagonal() ** 0.5):
    print(f"   {n:8s} = {x:7.4f} ± {e:.4f}")
for p, l, v, m, q in f.residuals:
    print(f"   {p} {l}: exp {1000*v:7.2f}  model {1000*m:7.2f}  pull {q:+.2f}")
print("   extra:", {f"{p} {l}": round(1000 * v, 1) for (p, l), v in f.extra.items()})
