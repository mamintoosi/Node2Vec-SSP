#!/bin/bash
# =============================================================================
# run_pareto_validation.sh - 20-seed validation for the Pareto candidate
#                            (p=1.0, q=0.5, d=2)
# =============================================================================
# Why: multi-seed evidence already exists for
#        (2.0, 1.0, 1) -> results/final_d1_primary/   (20 seeds, complete)
#        (1.0, 1.0, 2) -> results/final_d2/           (20 seeds, n11 artifacts)
#      but (1.0, 0.5, 2) exists ONLY at seed 42 (final_d2 n105 labels).
#      This script fills that one gap with the SAME protocol as the existing
#      (2,1,1) 20-seed stability run, so results are directly comparable.
#
# Writes ONLY into results/final_pareto_validation/ - no existing result is
# touched or overwritten. Resume support: if stability.json already exists
# there, the heavy 20-seed step is skipped and only the cheap ARI/aggregation
# steps run.
#
# Style aligned with run_section_balance.sh / run_stability_primary_d1.sh.
# CPU limit: OMP/MKL/OPENBLAS/NUMEXPR=2 thread vars + taskset -c 0,1
# Runtime:   ~5-10 minutes (full run); well under a minute in resume mode
# =============================================================================

# Repo location (fall back to the directory containing this script if the
# fixed path does not exist on this machine).
REPO_DEFAULT="/data/git/mamintoosi/Node2Vec-SSP"
if [ -d "$REPO_DEFAULT" ]; then
    cd "$REPO_DEFAULT" || exit 1
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/data/python-envs/pytorch/bin/python}"
CPU_CORES="0,1"

OUT_DIR="results/final_pareto_validation"
LOG_FILE="$OUT_DIR/pareto_validation.log"

echo "========================================="
echo "20-seed Validation @ p=1.0, q=0.5, d=2"
echo "Start time: $(date)"
echo "========================================="
echo "Repository: $(pwd)"
echo "Python:     $PYTHON_BIN"
echo "CPU limit:  taskset -c $CPU_CORES + OMP/MKL/OPENBLAS/NUMEXPR=2"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

mkdir -p "$OUT_DIR"

echo ""
echo "Running: taskset -c $CPU_CORES $PYTHON_BIN scripts/pareto_validation.py"
echo ""

# pipefail so the exit-code check below reflects the python run, not tee
set -o pipefail
taskset -c $CPU_CORES "$PYTHON_BIN" scripts/pareto_validation.py \
    2>&1 | tee "$LOG_FILE"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Command failed with exit code $exit_code"
    exit $exit_code
fi

echo ""
echo "========================================="
echo "DONE!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Output files (all inside $OUT_DIR):"
echo "  stability.json                      per course, seeds 0..19: sil/dbi/ch"
echo "  per_seed_metrics.json               + balance and section sizes per seed"
echo "  aggregate.json                      mean/std/min/max across seeds"
echo "  ari_stability.json                  all-pairs ARI over 20 KMeans seeds"
echo "  labels_course*_p1.0_q0.5_d2_seed*.json  per-seed cluster labels"
echo "  embeddings_course*_p1.0_q0.5_d2.npy     seed-42 embeddings"
echo "  run_metadata.json                   config, seeds, versions, timing"
echo "  $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Compare aggregate.json with results/final_d1_primary (2,1,1)"
echo "     and results/final_d2 (1,1,2) before fixing the final configuration."
echo "  2. Re-run this script any time to resume: existing stability.json is"
echo "     detected and the heavy step is skipped automatically."
