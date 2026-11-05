#!/usr/bin/env bash
# Main experiment: MAGIC depth sweeps (L = 4, ..., 64, seeds 0-2) for three initialisations
# at lambda_res = 2e-3 and 5e-3, followed by the combined figures and depth tables.
# Sweeps can also be run in parallel, see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
for lam in 2e-3 5e-3; do
  for init in small depth_scaled near_zero; do
    "$PY" scripts/run_sweep.py "configs/magic_lam${lam}_${init}.yaml"
  done
  "$PY" scripts/make_figures.py --out "outputs/figures_lam${lam}" \
    small="outputs/magic_lam${lam}_small" \
    depth_scaled="outputs/magic_lam${lam}_depth_scaled" \
    near_zero="outputs/magic_lam${lam}_near_zero"
done
