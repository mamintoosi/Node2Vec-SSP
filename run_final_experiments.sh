#!/bin/bash
# =============================================================================
# Master Script for Final d=1, d=2, and Grid Search Experiments
# =============================================================================
# This script provides options to run:
#   1. Complete d=1 experiments
#   2. Complete d=2 experiments
#   3. Node2Vec grid search
#   4. All experiments (d=1, d=2, and grid search)
#
# Usage:
#   ./run_final_experiments.sh [d1|d2|grid|all]
#
# Default: runs all experiments
#
# IMPORTANT: These are expensive experiments. Do NOT run them automatically.
# The user should execute this script manually with appropriate options.
# =============================================================================

set -euo pipefail

# ── Configuration ──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/python-envs/pytorch/bin/python}"
CPU_CORES="0,1"
COOLDOWN_TIME=5  # seconds between experiments

# Output directories
OUT_BASE="${REPO_ROOT}/results"
OUT_D1="${OUT_BASE}/final_d1"
OUT_D2="${OUT_BASE}/final_d2"
OUT_GRID="${OUT_BASE}/grid_search"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Functions ──
print_header() {
    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_start_time() {
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
}

print_end_time() {
    echo ""
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
}

print_cpu_limit() {
    echo ""
    echo "CPU limitation:"
    echo "  - taskset -c ${CPU_CORES}"
    echo "  - OMP_NUM_THREADS=2"
    echo "  - MKL_NUM_THREADS=2"
    echo "  - OPENBLAS_NUM_THREADS=2"
    echo "  - NUMEXPR_NUM_THREADS=2"
}

print_environment() {
    echo ""
    echo "Environment:"
    echo "  - Repository: ${REPO_ROOT}"
    echo "  - Python: ${PYTHON_BIN}"
    echo "  - Python version: $(${PYTHON_BIN} --version 2>&1)"
    echo "  - GPU enabled: $(${PYTHON_BIN} -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")' 2>/dev/null || echo 'N/A')"
}

print_experiment_info() {
    local d_value="$1"
    echo ""
    echo "Experiment configuration:"
    echo "  - Embedding dimension (d): ${d_value}"
    echo "  - Node2Vec p: 1.0"
    echo "  - Node2Vec q: 1.0"
    echo "  - Walk length: 10"
    echo "  - Number of walks: 80"
    echo "  - Context window: 5"
    echo "  - Word2Vec epochs: 30"
    echo "  - Number of clusters: 2"
    echo "  - Random seed: 42"
    echo "  - Output directory: ${OUT_BASE}/final_d${d_value}"
}

print_grid_info() {
    echo ""
    echo "Grid search configuration:"
    echo "  - p values: 0.5, 1.0, 2.0"
    echo "  - q values: 0.5, 1.0, 2.0"
    echo "  - d values: 1, 2, 3, 5, 10"
    echo "  - Total configurations: 45"
    echo "  - Courses: 1, 2, 3, 4, 5, 6"
    echo "  - Total runs: 270"
    echo "  - Random seed: 42"
    echo "  - Output directory: ${OUT_GRID}"
}

run_d1_experiment() {
    print_header "RUNNING d=1 EXPERIMENTS"
    
    mkdir -p "${OUT_D1}"
    mkdir -p "${OUT_D1}/figures"
    
    print_experiment_info 1
    print_start_time
    print_cpu_limit
    print_environment
    
    echo ""
    echo "Commands to be executed:"
    echo "  cd ${REPO_ROOT}"
    echo "  taskset -c ${CPU_CORES} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\"
    echo "    ${PYTHON_BIN} experiments/final_d1_d2_experiments.py"
    echo ""
    
    # Confirm before running
    read -p "Execute d=1 experiment? (this may take ~15-30 minutes) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "d=1 experiment skipped by user"
        return 0
    fi
    
    cd "${REPO_ROOT}"
    
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    
    taskset -c ${CPU_CORES} \
        ${PYTHON_BIN} experiments/final_d1_d2_experiments.py \
        2>&1 | tee "${OUT_D1}/experiment.log"
    
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        print_error "d=1 experiment failed with exit code ${exit_code}"
        return ${exit_code}
    fi
    
    print_end_time
    print_info "d=1 experiment completed successfully"
    echo ""
    echo "Output directory: ${OUT_D1}"
    echo "Log file: ${OUT_D1}/experiment.log"
}

run_d2_experiment() {
    print_header "RUNNING d=2 EXPERIMENTS"
    
    mkdir -p "${OUT_D2}"
    mkdir -p "${OUT_D2}/figures"
    
    print_experiment_info 2
    print_start_time
    print_cpu_limit
    print_environment
    
    echo ""
    echo "Commands to be executed:"
    echo "  cd ${REPO_ROOT}"
    echo "  taskset -c ${CPU_CORES} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\"
    echo "    ${PYTHON_BIN} experiments/final_d1_d2_experiments.py"
    echo ""
    
    # Confirm before running
    read -p "Execute d=2 experiment? (this may take ~15-30 minutes) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "d=2 experiment skipped by user"
        return 0
    fi
    
    cd "${REPO_ROOT}"
    
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    
    taskset -c ${CPU_CORES} \
        ${PYTHON_BIN} experiments/final_d1_d2_experiments.py \
        2>&1 | tee "${OUT_D2}/experiment.log"
    
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        print_error "d=2 experiment failed with exit code ${exit_code}"
        return ${exit_code}
    fi
    
    print_end_time
    print_info "d=2 experiment completed successfully"
    echo ""
    echo "Output directory: ${OUT_D2}"
    echo "Log file: ${OUT_D2}/experiment.log"
}

run_grid_search() {
    print_header "RUNNING NODE2VEC GRID SEARCH"
    
    mkdir -p "${OUT_GRID}"
    mkdir -p "${OUT_GRID}/figures"
    
    print_grid_info
    print_start_time
    print_cpu_limit
    print_environment
    
    echo ""
    echo "Commands to be executed:"
    echo "  cd ${REPO_ROOT}"
    echo "  taskset -c ${CPU_CORES} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\"
    echo "    ${PYTHON_BIN} experiments/grid_search_final.py"
    echo ""
    
    # Confirm before running
    read -p "Execute grid search? (this may take ~30-45 minutes) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Grid search skipped by user"
        return 0
    fi
    
    cd "${REPO_ROOT}"
    
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    
    taskset -c ${CPU_CORES} \
        ${PYTHON_BIN} experiments/grid_search_final.py \
        2>&1 | tee "${OUT_GRID}/grid_search.log"
    
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        print_error "Grid search failed with exit code ${exit_code}"
        return ${exit_code}
    fi
    
    print_end_time
    print_info "Grid search completed successfully"
    echo ""
    echo "Output directory: ${OUT_GRID}"
    echo "Log file: ${OUT_GRID}/grid_search.log"
    echo "Results: ${OUT_GRID}/grid_search_complete.json"
    echo "Selection: ${OUT_GRID}/selection_rationale.json"
}

run_all_experiments() {
    print_header "RUNNING ALL FINAL EXPERIMENTS"
    
    echo -e "${YELLOW}WARNING: This will run ALL experiments (d=1, d=2, and grid search).${NC}"
    echo -e "${YELLOW}Total estimated time: 60-90 minutes${NC}"
    echo ""
    
    # Confirm before running all
    read -p "Execute ALL experiments? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "All experiments skipped by user"
        return 0
    fi
    
    # Run in sequence
    run_d1_experiment
    if [ $? -ne 0 ]; then
        print_error "d=1 experiment failed, aborting remaining experiments"
        return 1
    fi
    
    sleep ${COOLDOWN_TIME}
    
    run_d2_experiment
    if [ $? -ne 0 ]; then
        print_error "d=2 experiment failed, aborting remaining experiments"
        return 1
    fi
    
    sleep ${COOLDOWN_TIME}
    
    run_grid_search
    if [ $? -ne 0 ]; then
        print_error "Grid search failed"
        return 1
    fi
}

# ── Main ──
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  d1      Run complete d=1 experiments"
    echo "  d2      Run complete d=2 experiments"
    echo "  grid    Run Node2Vec grid search"
    echo "  all     Run all experiments (d=1, d=2, and grid search)"
    echo "  help    Show this help message"
    echo ""
    echo "Default: If no option is provided, shows help."
    echo ""
    echo "Examples:"
    echo "  $0 d1     # Run d=1 experiments only"
    echo "  $0 all    # Run all experiments"
    echo ""
    echo "Note: Each experiment requires confirmation before execution."
    echo "      This is intentional to avoid accidental long-running computations."
}

# Check if Python environment exists
if [ ! -f "${PYTHON_BIN}" ] && ! command -v ${PYTHON_BIN} &> /dev/null; then
    print_warning "Python binary not found at ${PYTHON_BIN}"
    print_info "Please set PYTHON_BIN environment variable to your Python executable"
    echo "Example: export PYTHON_BIN=/path/to/python"
    echo ""
fi

# Main logic
case "${1:-help}" in
    d1)
        run_d1_experiment
        ;;
    d2)
        run_d2_experiment
        ;;
    grid)
        run_grid_search
        ;;
    all)
        run_all_experiments
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

exit 0
