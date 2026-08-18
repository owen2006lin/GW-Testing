#!/usr/bin/env bash
# Reproduce every result and figure from a clean checkout.
# CPU only, no model downloads. Roughly 80 minutes on 2 cores.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] verifying the isometric construction"
python experiments/check_isometry.py

echo "[2/6] width confounder reproduction"
python experiments/run_width_sweep.py

echo "[3/6] Type-I error control"
python experiments/run_type1.py

echo "[4/6] main experiment"
python experiments/run_main.py

echo "[5/6] curvature sweep"
python experiments/run_curvature_sweep.py

echo "[6/6] figures"
python experiments/make_figures.py

echo "done -- results in results/, figures in figures/"
