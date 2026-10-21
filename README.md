# bc-hyperfine — the quarkonium hyperfine splittings as the arbiter of the contact interaction

[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--4678--0034-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0009-4678-0034)

Code, fits and manuscript for

> P. A. Oppong, *The heavy-quarkonium hyperfine splittings as the arbiter of
> the contact interaction* (2026).

Nine hyperfine observables are now measured — the S-wave splittings of
cc̄, bb̄ and B_c for 1S and 2S (the last two from ATLAS 2026 and
ATLAS+CMS), and the P-wave singlet–centroid offsets of h_c, h_b(1P),
h_b(2P). Four prescriptions for the spin–spin contact term are fitted to
them, with fixed or running coupling, on top of a spin-averaged potential
and quark masses taken *fixed* from the model-independent centroid
extraction of https://github.com/apoppong24/bc-potential-inverse, so that
only the contact prescription is tested:

| prescription | profile δ_X(r) | parameters |
|---|---|---|
| first-order | δ³(r), first-order perturbation theory | coupling only |
| Gaussian | (σ/√π)³ e^{−σ²r²}, mass-independent (Barnes–Godfrey–Swanson) | σ |
| Godfrey–Isgur | Gaussian with their mass-dependent width | σ₀, s |
| Yukawa | m² e^{−mr}/4πr (form of the lattice spin–spin potential, Koma & Koma) | m |

The smeared forms are solved non-perturbatively (singlet and triplet
separately, S and P waves) with the Numerov solver of `quarkonium/`.

Headline results (`results/hyperfine_analysis.md`):

* **h_c kills smearing.** The h_c sits at the χ_cJ centroid to 0.2 MeV; a
  Gaussian of σ = 1 GeV (the standard potential-model value) puts it 16 MeV
  below. The r.m.s. radius of any contact term must be < 0.07 fm; every fit
  pushes its width there on its own — and cannot go further, because a
  δ-function has no non-perturbative solution in 3D (the singlet collapses).
* **The cc̄/bb̄ ratio fixes the Coulomb strength κ, which the spectrum cannot
  see — jointly with the running.** With running α_s the data want
  κ = 0.2–0.4 ≈ (4/3)α_s; with fixed α_s they want κ = 0. Mass-independent
  smearing + running coupling fails at every κ (χ² > 80). Robust to the
  theory floor (3 or 5 MeV).
* **The cc̄ 2S splitting is the universal outlier** (predicted 55–63 vs
  48.3 MeV) — the size of the coupled-channel shift computed by Eichten,
  Lane & Quigg (2004) for exactly these states.
* **Without it, first-order perturbation theory with a running coupling and no
  smearing fits the other eight with χ² = 2.6/7 and one parameter**
  (Λ = 0.213 ± 0.011 GeV); the smeared forms reach χ²/dof = 1.1–2.1 with
  two or three, their excess coming from the B_c.
* **B_c is the arbiter.** Fitted to cc̄ + bb̄ only, the prescriptions
  differ only in what they predict for B_c: first-order + running gives
  M(B_c*) − M(B_c) = 62 ± 2 MeV (64.5 with the η_c-annihilation-corrected cc̄
  input), the smeared forms 50–58 MeV. ATLAS (64.5 ± 1.8) selects the
  former; lattice (54 ± 3 HPQCD 2012, 55 ± 3 Mathur et al. 2018) would select
  the latter.

Inputs were checked against the PDG 2026 listings and the arXiv abstracts of
the lattice papers on 2026-10-20; the η_b(2S) (Belle, omitted from the PDG
summary table) is used with its full symmetrised error and dropping it
changes nothing.

## Run

```bash
pip install -r requirements.txt
python fit_hyperfine.py pert run --kappa 0.2          # presc: pert|gauss|gi|yukawa; coupling: run|fixed  (~20 s / ~5 min)
python fit_hyperfine.py pert run --kappa 0.2 --exclude cc2S cb1S cb2S [--lattice-cc1S] [--floor 5] [--dmc 0.024]
bash run_all.sh                                       # the full grid (~2 h on one core)
python analyse_hyperfine.py                           # -> results/hyperfine_analysis.{md,json}, paper/figures/fig1-4
python paper/make_tables.py                           # -> paper/tables/*.tex
cd paper && latexmk -pdf main.tex
python -m pytest                                      # solver + centroid + hyperfine tests
```

All fits used in the paper are committed under `results/` (`hf_*.json`,
logs), together with the five spin-averaged potentials
(`centroid_K6_lam1*.json`, κ = 0, 0.2, 0.4, 0.6, 0.8) they are built on.

## Layout

```
quarkonium/hyperfine.py   data, the four prescriptions, non-perturbative singlet/triplet solution, fit
quarkonium/               Numerov solver, spline potential (flexible.py), spin-dependent shifts, PDG 2026 data
fit_hyperfine.py          CLI; analyse_hyperfine.py: comparison, κ scan, P-wave bound, B_c with errors, figures
paper/                    main.tex, refs.bib, make_tables.py, figures/, tables/
results/                  hf_*.json fits (+ logs), centroid_K6_lam1*.json base potentials, hyperfine_analysis.{md,json}
tests/
```

## Acknowledgement

Extends the author's MSc project at the University of Glasgow (supervisor
Christine Davies). An AI assistant (Claude, Anthropic) was used to implement
the fits and analysis scripts and to help draft the manuscript.

## License

MIT (see `LICENSE`).
