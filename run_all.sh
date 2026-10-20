#!/bin/bash
# all 16 fits (4 prescriptions x 2 couplings x 2 kappa) + floor variants for the reference kappa
cd "$(dirname "$0")"
for k in 0 0.4; do for c in run fixed; do for p in pert gauss gi yukawa; do
  python fit_hyperfine.py $p $c --kappa $k > results/log_hf_${p}_${c}_k${k}.log 2>&1
done; done; done
for fl in 1 5; do for p in pert gauss gi yukawa; do
  python fit_hyperfine.py $p run --kappa 0 --floor $fl > results/log_hf_${p}_run_k0_floor${fl}.log 2>&1
done; done
echo ALLDONE
