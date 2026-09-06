#!/bin/bash
# ============================================================
# Comprehensive (p, q, d) Grid Search for Node2Vec — seed=0
# 3×3×5 = 45 configurations × 6 courses = 270 runs
# Estimated time: ~30–40 minutes
# Output: results/reproduced_seed0/
# ============================================================
set -e

cd /data/git/mamintoosi/Deepwalk-SSP || exit 1

CPU_CORES="2,3"
COOLDOWN_TIME=10

run_with_limits() {
    local cmd="$1"
    local log_file="$2"
    
    echo "========================================="
    echo "Running: $cmd"
    echo "Log: $log_file"
    echo "Start time: $(date)"
    echo "========================================="
    
    taskset -c $CPU_CORES $cmd 2>&1 | tee "$log_file"
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code"
        exit $exit_code
    fi
    
    echo "Finished at: $(date)"
    if [ $COOLDOWN_TIME -gt 0 ]; then
        echo "Cooling down for $COOLDOWN_TIME seconds..."
        sleep $COOLDOWN_TIME
    fi
}

# Create output directory
mkdir -p results/reproduced_seed0/figures

# ============================================================
# Run grid search with seed=0
# ============================================================
echo "=============================================="
echo "  Node2Vec (p, q, d) Grid Search — seed=0"
echo "  45 configs × 6 courses"
echo "=============================================="
echo ""

run_with_limits \
    "/data/python-envs/pytorch/bin/python experiments/grid_search_pqd.py --seed 0" \
    "results/reproduced_seed0/grid_search.log"

echo ""
echo "========================================="
echo "GRID SEARCH (seed=0) COMPLETED!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Results: results/reproduced_seed0/grid_search_pqd.json"
echo "Figures: results/reproduced_seed0/figures/"
echo "Log:     results/reproduced_seed0/grid_search.log"
