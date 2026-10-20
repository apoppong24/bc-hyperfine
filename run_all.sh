#!/bin/bash
# All fits used in the paper: 4 prescriptions x 2 couplings x kappa in {0, 0.4} (+ kappa scan for the
# informative combinations), no-cc2S / no-bb2S / lattice-cc1S / floor variants.  ~2.5 h on one core.
cd "$(dirname "$0")"
run() { python fit_hyperfine.py "$@" > "results/log_hf_$(echo "$@" | tr ' ' '_' | sed 's/--//g').log" 2>&1; }
for k in 0 0.4; do for c in run fixed; do for p in pert gauss gi yukawa; do run $p $c --kappa $k; done; done; done
for k in 0.2 0.6 0.8; do run pert run --kappa $k; run pert fixed --kappa $k; run gi run --kappa $k; run gauss fixed --kappa $k; run yukawa fixed --kappa $k; done
for k in 0.2 0.4; do run pert run --kappa $k --exclude cc2S; done
run pert run --kappa 0.2 --exclude cc2S bb2S
run pert run --kappa 0.2 --exclude cc2S --lattice-cc1S
run gi run --kappa 0.2 --exclude cc2S
run gauss fixed --kappa 0.8 --exclude cc2S
run yukawa fixed --kappa 0.8 --exclude cc2S
for fl in 1 5; do run pert run --kappa 0.2 --floor $fl; done
echo ALLDONE
