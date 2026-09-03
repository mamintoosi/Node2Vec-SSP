#!/bin/bash
# ============================================================
# Comprehensive (p, q, d) Grid Search for Node2Vec
# 3×3×5 = 45 configurations × 6 courses = 270 runs
# Estimated time: ~30–40 minutes
# ============================================================
set -e

cd /data/git/mamintoosi/Deepwalk-SSP

echo "=============================================="
echo "  Node2Vec (p, q, d) Grid Search"
echo "  45 configs × 6 courses, seed=42"
echo "=============================================="
echo ""
echo "Start time: $(date)"
echo ""

/data/python-envs/pytorch/bin/python experiments/grid_search_pqd.py 2>&1 | tee results/reproduced/grid_search.log

echo ""
echo "End time: $(date)"
echo "Log saved to: results/reproduced/grid_search.log"
echo "Results saved to: results/reproduced/grid_search_pqd.json"
echo "Figures saved to: results/reproduced/figures/grid_heatmap_*.pdf"
