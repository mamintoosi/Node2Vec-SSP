#!/bin/bash
# =============================================================================
# run_stability_primary_d1.sh - 20-seed stability for the PRIMARY Node2Vec
#                               configuration p=2.0, q=1.0, d=1
# =============================================================================
# Why: results/final_d1/stability.json was produced with the neutral config
#      (p=1, q=1, d=1) because experiments/final_d1_d2_experiments.py hardcodes
#      P=1.0, Q=1.0. The manuscript's tab:stability claims the primary config.
#      This script fills that gap without touching any existing result.
#
# Style aligned with run_section_balance.sh.
# CPU limit: OMP/MKL/OPENBLAS/NUMEXPR=2 thread vars + taskset -c 0,1
# Runtime:   ~5-10 minutes
# =============================================================================

# Repo location (GitHub repo renamed Deepwalk-SSP -> Node2Vec-SSP; fall back to
# the directory containing this script if the fixed path does not exist).
REPO_DEFAULT="/data/git/mamintoosi/Node2Vec-SSP"
if [ -d "$REPO_DEFAULT" ]; then
    cd "$REPO_DEFAULT" || exit 1
else
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/data/python-envs/pytorch/bin/python}"
CPU_CORES="0,1"

OUT_DIR="results/final_d1_primary"
LOG_FILE="$OUT_DIR/stability_primary_d1.log"

echo "========================================="
echo "20-seed Stability @ p=2.0, q=1.0, d=1"
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

# ---------------------------------------------------------------------------
# Generate the runner (so experiments/final_d1_d2_experiments.py is NOT
# modified). It reuses the exact pipeline functions from that module:
# same data prep, graph, walks, Word2Vec settings, KMeans; only P=2.0, Q=1.0.
# Protocol matches the existing stability analysis exactly:
#   - stability.json : per course, seeds 0..19, walks+Word2Vec+KMeans per seed
#   - ari_stability  : embeddings at seed 42; KMeans over 20 seeds; all-pair ARI
# ---------------------------------------------------------------------------
cat > "$OUT_DIR/run_stability_p2q1d1.py" <<'PYEOF'
# -*- coding: utf-8 -*-
"""20-seed stability for the PRIMARY config (p=2.0, q=1.0, d=1).

Reuses the pipeline functions of experiments/final_d1_d2_experiments.py
without modifying that file. Writes only into results/final_d1_primary/.
Protocol matches the existing stability analysis exactly.
"""
import os, sys, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))
import final_d1_d2_experiments as F   # module constants P/Q are NOT used here

from sklearn.metrics import adjusted_rand_score

OUT_DIR = HERE
N_SEEDS = F.N_SEEDS          # 20
SEED_MAIN = F.SEED           # 42
P, Q = 2.0, 1.0              # PRIMARY configuration (only change vs the module)

# ---- data & graphs (identical to run_experiment in the module) ----
print(f"[1/4] Loading data & building graphs ...")
random.seed(SEED_MAIN)
np.random.seed(SEED_MAIN)
cd = {}
for i in F.COURSES:
    mat, student_labels = F.read_class(os.path.join(F.DATA, f"{i}.txt"))
    uc = F.find_univ(mat)
    mf = np.delete(mat, uc, axis=1)
    G = F.mk_graph(mf)
    cd[i] = {"m": mf, "G": G, "n": mf.shape[0], "labels": student_labels}
    print(f"  Course {i}: {mf.shape[0]} students, {mf.shape[1]} courses, {G.number_of_edges()} edges")

# ---- stability: everything re-drawn per seed (matches existing protocol) ----
print(f"[2/4] 20-seed stability at p={P}, q={Q}, d=1 (seeds 0..{N_SEEDS-1}) ...")
stab = {}
for i in F.COURSES:
    sres = []
    for seed in range(N_SEEDS):
        w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH, p=P, q=Q, seed=seed)
        emb, _ = F.w2v(w, vs=1, window=F.WINDOW, epochs=F.EPOCHS, seed=seed)
        l = F.km(emb, seed=seed)
        me = F.calc_m(emb, l)
        sres.append({"seed": seed, "sil": me["sil"], "dbi": me["dbi"], "ch": me["ch"]})

        labels_file = os.path.join(OUT_DIR, f"labels_course{i}_n11_seed{seed}.json")
        with open(labels_file, "w") as f:
            json.dump({
                "course": i, "method": "n11", "d": 1, "p": P, "q": Q, "seed": seed,
                "student_index": list(range(len(cd[i]["labels"]))),
                "student_labels": cd[i]["labels"],
                "cluster_labels": l.tolist()
            }, f, indent=2)
    sils = [r["sil"] for r in sres]
    stab[str(i)] = sres
    print(f"  Course {i}: Sil={np.mean(sils):.3f} +/- {np.std(sils):.3f} "
          f"(min={np.min(sils):.3f}, max={np.max(sils):.3f})")

F.ss(stab, "stability.json", OUT_DIR)

# ---- ARI stability: fixed seed-42 embeddings, KMeans over 20 seeds, all pairs ----
print("[3/4] ARI stability (KMeans on seed-42 embeddings, 20 seeds) ...")
ari = {}
for i in F.COURSES:
    w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH, p=P, q=Q, seed=SEED_MAIN)
    emb, idx = F.w2v(w, vs=1, window=F.WINDOW, epochs=F.EPOCHS, seed=SEED_MAIN)
    np.save(os.path.join(OUT_DIR, f"embeddings_course{i}_n11.npy"), emb)
    labs = [F.km(emb, seed=s) for s in range(N_SEEDS)]
    ari_vals = [adjusted_rand_score(labs[a], labs[b])
                for a in range(len(labs)) for b in range(a + 1, len(labs))]
    ari[str(i)] = {"mean": float(np.mean(ari_vals)), "std": float(np.std(ari_vals))}

    seed42_file = os.path.join(OUT_DIR, f"labels_course{i}_n11_seed{SEED_MAIN}.json")
    with open(seed42_file, "w") as f:
        json.dump({
            "course": i, "method": "n11", "d": 1, "p": P, "q": Q, "seed": SEED_MAIN,
            "student_index": list(range(len(cd[i]["labels"]))),
            "student_labels": cd[i]["labels"],
            "cluster_labels": labs[SEED_MAIN].tolist()
        }, f, indent=2)
    print(f"  Course {i}: ARI={np.mean(ari_vals):.3f} +/- {np.std(ari_vals):.3f}")

print("[4/4] Saving summaries ...")
F.ss(ari, "ari_stability.json", OUT_DIR)
print("Done.")
PYEOF

echo ""
echo "Running: taskset -c $CPU_CORES $PYTHON_BIN $OUT_DIR/run_stability_p2q1d1.py"
echo ""

# pipefail so the exit-code check below reflects the python run, not tee
set -o pipefail
taskset -c $CPU_CORES "$PYTHON_BIN" "$OUT_DIR/run_stability_p2q1d1.py" \
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
echo "Output files:"
echo "  $OUT_DIR/stability.json"
echo "  $OUT_DIR/ari_stability.json"
echo "  $OUT_DIR/labels_course*_n11_seed*.json"
echo "  $OUT_DIR/embeddings_course*_n11.npy"
echo "  $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check $OUT_DIR/stability.json and $OUT_DIR/ari_stability.json"
echo "  2. Use them for tab:stability with the primary configuration"
echo "     (p=2.0, q=1.0, d=1) as the manuscript caption claims."
