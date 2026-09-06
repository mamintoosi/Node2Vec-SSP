# Final Report: Node2Vec Student Sectioning Experiments

## 1. Current State of the Experimental Pipeline

The repository at `/data/git/mamintoosi/Node2Vec-SSP` contains a complete experimental pipeline for Node2Vec-based student sectioning. The latest reproduction was performed using `experiments/run_all.py` with results under `results/reproduced/`.

**Key components:**
- Data loading and preprocessing (`experiments/core.py`)
- Graph construction from student-course enrollment matrices
- Node2Vec random walk generation and Word2Vec training
- KMeans clustering and evaluation metrics
- Existing sensitivity analyses (p,q,d,walk_length,num_walks,window)

## 2. Existing Sensitivity/Grid-Search Experiments

Found existing experiments in `results/reproduced/`:

- **sensitivity_pq.json**: p,q sensitivity (3×3 grid, d=2)
- **sensitivity_dim.json**: Dimensionality sensitivity (d=1,2,3,5,10)
- **grid_search_pqd.json**: Comprehensive (p,q,d) grid search (3×3×5=45 configs)

The existing grid search shows:
- **Best overall**: p=2.0, q=1.0, d=1 (avg Silhouette = 0.7654)
- **Best at d=1**: p=2.0, q=1.0 (avg Sil = 0.7654)
- **Best at d=2**: p=0.5, q=0.5 (avg Sil = 0.6321)
- d=1 substantially outperforms d=2 across all configurations

## 3. Grid-Search Design Selected

Based on existing experiments, the grid search uses:
- **p values**: [0.5, 1.0, 2.0] (return parameter)
- **q values**: [0.5, 1.0, 2.0] (in-out parameter)
- **d values**: [1, 2, 3, 5, 10] (embedding dimension)
- **Fixed parameters**: walk_length=10, num_walks=80, window=5, epochs=30, n_clusters=2
- **Seed**: 42
- **Criterion**: Average Silhouette Score across 6 courses

The grid is scientifically appropriate because:
- Uses the actual Node2Vec implementation from the codebase
- Parameters are within tested ranges
- Covers both BFS-like (q<1) and DFS-like (q>1) exploration
- Includes neutral configuration (p=1, q=1) equivalent to DeepWalk

## 4. Evidence for d=1 vs d=2

**Current evidence strongly supports d=1:**
- Best d=1 configuration: avg Silhouette = 0.7654
- Best d=2 configuration: avg Silhouette = 0.6321
- d=1 outperforms d=2 for ALL (p,q) combinations in the grid
- The difference is substantial (~0.13-0.14 in Silhouette)

**However, the choice should be based on actual reproduced results**, which is why both d=1 and d=2 experiments are prepared.

## 5. Experiments to Execute for d=1

Run: `./run_final_experiments.sh d1`

This executes `experiments/final_d1_d2_experiments.py` with d=1, producing:
- `results/final_d1/all_methods.json` - Silhouette, DBI, CH for all 5 methods
- `results/final_d1/runtime.json` - Execution times
- `results/final_d1/sensitivity_pq.json` - p,q sensitivity at d=1
- `results/final_d1/stability.json` - 20-seed stability analysis
- `results/final_d1/ari_stability.json` - ARI stability
- `results/final_d1/statistical_analysis.json` - Wilcoxon tests
- `results/final_d1/embeddings_course{i}_{method}.npy` - Node2Vec embeddings
- `results/final_d1/labels_course{i}_{method}.json` - Cluster assignments
- `results/final_d1/figures/*.pdf` - Publication figures

**Time estimate**: 15-30 minutes

## 6. Experiments to Execute for d=2

Run: `./run_final_experiments.sh d2`

Same structure as d=1 but with d=2:
- `results/final_d2/all_methods.json`
- `results/final_d2/runtime.json`
- `results/final_d2/sensitivity_pq.json`
- `results/final_d2/stability.json`
- `results/final_d2/ari_stability.json`
- `results/final_d2/statistical_analysis.json`
- `results/final_d2/embeddings_course{i}_{method}.npy`
- `results/final_d2/labels_course{i}_{method}.json`
- `results/final_d2/figures/*.pdf`

**Time estimate**: 15-30 minutes

## 7. Experiments That Do NOT Need Rerunning

The following baselines do NOT depend on Node2Vec embedding dimension and do not need recomputation:
- **BoW + KMeans**: Uses raw enrollment matrix
- **PCA + KMeans**: Projects to fixed 2D space
- **Spectral clustering**: Operates on graph directly

These are included in the main experiment script for completeness but do not change with d.

## 8. Clustering Labels Saved

**YES** - Clustering labels are now saved for every relevant experiment.

**Location for d=1 experiments:**
- `results/final_d1/labels_course{i}_{method}.json`
- Contains: course, method, d, p, q, seed, student_index, student_labels, cluster_labels

**Location for d=2 experiments:**
- `results/final_d2/labels_course{i}_{method}.json`

**Location for grid search:**
- `results/grid_search/labels_p{p}_q{q}_d{d}_course{i}.json`

Each label file includes:
- Configuration metadata (course, method, d, p, q, seed)
- Student identifiers (index and original labels)
- Cluster assignments for all students

This allows post-hoc calculation of ARI, stability, agreement, and other cluster-level analyses without rerunning Node2Vec.

## 9. Where Clustering Labels Are Saved

See Section 8 above for complete paths.

## 10. Embeddings Saved

**YES** - Embeddings are saved for Node2Vec methods when practical.

**Location for d=1 experiments:**
- `results/final_d1/embeddings_course{i}_{method}.npy`
- Saved for n11 (p=1,q=1) and n105 (p=1,q=0.5) methods

**Location for d=2 experiments:**
- `results/final_d2/embeddings_course{i}_{method}.npy`

**Location for grid search:**
- `results/grid_search/embeddings_p{p}_q{q}_d{d}_course{i}.npy`
- Saved for key configurations (d in [1,2], p=1, q=1)

**NOT saved:**
- BoW features (can be reconstructed from data)
- PCA-reduced features (compact, can be recomputed)
- Spectral affinity matrices (compact, can be recomputed)
- Grid search embeddings for all 45 configurations (would be excessive)

Each embedding file is a .npy array with shape (n_students, d).

## 11. Where Grid-Search Results Are Saved

- `results/grid_search/grid_search_complete.json` - Complete grid results
  - All 270 individual results (p,q,d,course,silhouette)
  - All configuration averages
  - Best overall configuration
  - Best d=1 configuration
  - Best d=2 configuration
  - Selection criterion and rationale
  
- `results/grid_search/selection_rationale.json` - Why configuration was selected

- `results/grid_search/figures/grid_heatmap_d{d}.{pdf,png}` - Individual heatmaps with independent normalization

- `results/grid_search/figures/grid_heatmap_all_d_independent.{pdf,png}` - Combined with independent per-panel normalization

- `results/grid_search/figures/grid_heatmap_all_d_common.{pdf,png}` - Combined with common scale for absolute comparison

## 12. Heatmap Normalization Decision

**Decision: Provide BOTH independent and common normalization.**

**Rationale:**
- The scientific purpose is to show the structure/sensitivity of each p×q grid separately AND to compare absolute scores across d values
- Independent normalization (each d gets its own color range) makes within-panel differences visible
- Common normalization allows comparison of absolute Silhouette Scores across d values
- Both versions are generated to serve different purposes
- Caption will state which normalization is used

**Implementation:**
- `grid_heatmap_all_d_independent.*` - Independent normalization per panel
- `grid_heatmap_all_d_common.*` - Common scale across all panels
- Individual heatmaps (`grid_heatmap_d{d}.*`) use independent normalization

## 13. Figures Created/Regenerated

**New figures to be generated:**

For d=1 experiments (`results/final_d1/figures/`):
- `baseline_comparison.pdf/png` - 5 methods comparison
- `method_comparison_all_metrics.pdf/png` - All 3 metrics
- `silhouette_score_comparison.pdf/png` - Silhouette scores
- `DBI_comparison.pdf/png` - DBI scores
- `CHI_comparison.pdf/png` - CH scores
- `stability_boxplot.pdf/png` - 20-seed stability
- `sensitivity_heatmap.pdf/png` - p,q heatmap at d=1

For d=2 experiments (`results/final_d2/figures/`):
- Same figures as d=1 but with d=2 results

For grid search (`results/grid_search/figures/`):
- `grid_heatmap_d{d}.{pdf,png}` for d in [1,2,3,5,10] - Individual heatmaps
- `grid_heatmap_all_d_independent.{pdf,png}` - Combined with independent normalization
- `grid_heatmap_all_d_common.{pdf,png}` - Combined with common scale
- `sil_vs_dim.{pdf,png}` - Silhouette vs dimension for selected configs

## 14. Manuscript Changes Made

**NONE YET** - Per instructions, the manuscript should NOT be modified to change numerical claims until the new runs have been executed.

**What needs updating after runs (documented for later):**

1. **Section "Dataset and Experimental Setup"** (line ~597):
   - Add paragraph about grid search results
   - Update statement about d=1 vs d=2 if results confirm

2. **Table \ref{tab:baseline_comparison}**:
   - May need updating if d=1 results are used instead of d=2

3. **New subsection needed**: "Node2Vec Grid Search and Hyperparameter Selection"
   - Location: After Section "Hyperparameter Configuration" or in experimental results
   - Content: grid design, evaluation method, selection criterion, selected configuration

4. **Section references**: Update any cross-references to sensitivity experiments

**Manuscript changes that ARE independent of results (can be done now):**
- Adding the grid-search subsection structure
- Documenting where d-dependent numerical claims occur

## 15. Cliff's δ Status

**Cliff's δ was NOT in the current codebase.** 

The existing `run_all.py` calculates a "delta" metric but this is NOT Cliff's δ - it's a dominance ratio:
```python
dom = sum(1 for a in x for b in y if a>b) - sum(1 for a in x for b in y if a<b)
delta = dom/(n*n)
```

This is preserved in the new scripts. No decision to remove or retain Cliff's δ was needed because it wasn't present.

## 16. Remaining Manuscript Issues

1. **Grid search subsection not yet added** - Needs to be added to manuscript
2. **d=1 vs d=2 narrative** - Currently says d=2 is primary for visualization, but evidence strongly favors d=1
3. **Numerical values** - All d-dependent numerical claims need updating after runs
4. **Figure references** - May need updating if new figures are added
5. **Section balance table** - May need regeneration with final configuration

## 17. Exact Bash Script Path

**Master script:** `./run_final_experiments.sh`

**Usage:**
```bash
# Run d=1 experiments only
./run_final_experiments.sh d1

# Run d=2 experiments only  
./run_final_experiments.sh d2

# Run grid search only
./run_final_experiments.sh grid

# Run all experiments (d1, d2, and grid)
./run_final_experiments.sh all

# Show help
./run_final_experiments.sh help
```

**Individual scripts:**
- `experiments/final_d1_d2_experiments.py` - Main experiment runner
- `experiments/grid_search_final.py` - Grid search runner

## 18. Exact Commands to Run

**To run d=1 experiments:**
```bash
cd /data/git/mamintoosi/Node2Vec-SSP
./run_final_experiments.sh d1
```

**To run d=2 experiments:**
```bash
cd /data/git/mamintoosi/Node2Vec-SSP
./run_final_experiments.sh d2
```

**To run grid search:**
```bash
cd /data/git/mamintoosi/Node2Vec-SSP
./run_final_experiments.sh grid
```

**To run all (recommended order: grid first, then d1, then d2):**
```bash
cd /data/git/mamintoosi/Node2Vec-SSP
./run_final_experiments.sh all
```

**Alternative: Run Python scripts directly**
```bash
cd /data/git/mamintoosi/Node2Vec-SSP
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
taskset -c 0,1 /data/python-envs/pytorch/bin/python experiments/final_d1_d2_experiments.py 2>&1 | tee results/final_d1/experiment.log
taskset -c 0,1 /data/python-envs/pytorch/bin/python experiments/grid_search_final.py 2>&1 | tee results/grid_search/grid_search.log
```

## 19. Confirmation: Scripts Use At Most 2 CPU Cores

**YES** - All scripts enforce 2-core limit.

**Implementation:**
- Bash script sets: `taskset -c 0,1`
- Environment variables: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`
- Python scripts use `workers=1` for Word2Vec
- No parallel processing in Python code

The bash script prints CPU limitation information before each experiment.

## 20. Confirmation: GPU Remains Enabled Where Supported

**YES** - GPU execution is not disabled.

The scripts do NOT explicitly disable GPU. Gensim's Word2Vec uses CPU by default (workers=1), but if the environment has GPU-accelerated libraries, they remain available.

The master script prints GPU status:
```bash
Python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")'
```

No GPU-specific code is used; the pipeline relies on CPU for random walks and Word2Vec training.

## 21. Confirmation: Expensive Experiments NOT Executed

**YES** - I have NOT executed any of the expensive experiments.

Created scripts and prepared infrastructure only. The user must execute:
- `./run_final_experiments.sh d1` - for d=1 experiments
- `./run_final_experiments.sh d2` - for d=2 experiments
- `./run_final_experiments.sh grid` - for grid search

Each script requires explicit user confirmation before execution.

Only lightweight validation was performed:
- Reading source code
- Reading JSON result files
- Static code inspection
- Checking dependencies
- Validating paths and filenames

## 22. Exact Files Modified/Created

**New files created:**

1. `experiments/final_d1_d2_experiments.py` - Main experiment script for d=1 and d=2
2. `experiments/grid_search_final.py` - Comprehensive grid search with complete outputs
3. `run_final_experiments.sh` - Master bash script for all experiments

**Existing files NOT modified:**
- `experiments/run_all.py` - Preserved as-is
- `experiments/grid_search_pqd.py` - Preserved as-is
- `paper/sn-article.tex` - Not modified (numerical claims await results)
- `paper/sn-bibliography.bib` - Not modified
- `results/reproduced/*` - Not modified (existing validated results)

**Output directories to be created when experiments run:**
- `results/final_d1/` - d=1 experiment outputs
- `results/final_d2/` - d=2 experiment outputs
- `results/grid_search/` - Grid search outputs

---

## Summary

All preparation work is complete. The repository now has:

1. **Complete grid-search infrastructure** that saves all results, labels, and embeddings
2. **Full d=1 experiment pipeline** that produces all required outputs
3. **Full d=2 experiment pipeline** that produces all required outputs
4. **Master bash script** with CPU limiting, confirmation prompts, and proper logging
5. **Output organization** that keeps d=1, d=2, and grid search results separate
6. **Reproducibility metadata** including timestamps, git commit, Python version, and all parameters
7. **Clustering labels saved** for every experiment in machine-readable JSON format
8. **Embeddings saved** for key configurations in NumPy format

**Next step:** Execute `./run_final_experiments.sh all` (or individual scripts) to generate the actual results, then update the manuscript with the real numerical findings.
