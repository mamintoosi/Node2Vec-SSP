#!/bin/bash
# run_section_balance.sh - Section balance analysis for all 5 methods
# Re-runs clustering with seed=42 to capture section sizes (~1-2 min)

cd /data/git/mamintoosi/Deepwalk-SSP || exit 1

CPU_CORES="0,1"
COOLDOWN_TIME=10

echo "========================================="
echo "Section Balance Analysis"
echo "Start time: $(date)"
echo "========================================="

mkdir -p results/reproduced

taskset -c $CPU_CORES /data/python-envs/pytorch/bin/python experiments/section_balance.py \
    2>&1 | tee results/reproduced/section_balance.log

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
echo "Output files:"
echo "  results/reproduced/section_balance.json"
echo "  results/reproduced/section_balance_table.tex"
echo "  results/reproduced/section_balance_table_detail.tex"
echo ""
echo "Next steps:"
echo "  1. Open results/reproduced/section_balance_table.tex"
echo "  2. Paste the compact table into sn-article.tex replacing tab:section_balance"
echo "  3. Optionally add the detail table as supplementary"
