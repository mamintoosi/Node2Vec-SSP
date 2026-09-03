# Reproduction Report — Seed=42 Full Run

**Date:** 2026-09-03  
**Script:** `experiments/run_all.py`  
**Seed:** 42 (fixed for Node2Vec walks, Word2Vec init, KMeans)  
**Python:** `/data/python-envs/pytorch/bin/python`  
**Duration:** ~25 minutes  

---

## 1. Bug Fixed During Run

**File:** `experiments/run_all.py`, function `fig_graph_density()`  
**Issue:** `KeyError: 'density'` — the graph stats JSON uses key `"d"` for density, but the figure function referenced `"density"`.  
**Fix:** Changed all `["density"]` to `["d"]` in the function.  
**Also created:** `experiments/fix_two_figs.py` to regenerate the two missing figures from saved JSONs without re-running the full pipeline.

---

## 2. Methods Evaluated (5 methods × 6 courses = 30 runs)

| Method Key | Label | p | q |
|-----------|-------|---|---|
| `bow` | BoW+KMeans | — | — |
| `pca` | PCA+KMeans | — | — |
| `spec` | Spectral | — | — |
| `n11` | Node2Vec (p=1, q=1) | 1.0 | 1.0 |
| `n105` | Node2Vec (p=1, q=0.5) | 1.0 | 0.5 |

---

## 3. Main Results (Table 1 — Silhouette Scores)

| Course | BoW | PCA | Spectral | N2V (1,1) | N2V (1,0.5) | Winner |
|--------|-----|-----|----------|-----------|-------------|--------|
| 1 | 0.090 | 0.376 | 0.083 | 0.599 | 0.594 | N2V (1,1) |
| 2 | 0.231 | 0.603 | 0.205 | 0.614 | 0.608 | N2V (1,1) |
| 3 | 0.154 | 0.425 | 0.135 | 0.605 | 0.569 | N2V (1,1) |
| 4 | 0.128 | 0.428 | 0.128 | 0.628 | 0.597 | N2V (1,1) |
| 5 | 0.137 | 0.437 | 0.074 | 0.745 | 0.746 | N2V (1,0.5) |
| 6 | 0.202 | 0.524 | 0.143 | 0.583 | 0.602 | N2V (1,0.5) |
| **Avg** | **0.157** | **0.466** | **0.128** | **0.629** | **0.619** | **N2V (1,1)** |

**Win rate:** Node2Vec(1,1) wins 4/6 courses, Node2Vec(1,0.5) wins 2/6 courses vs PCA.  
**Node2Vec vs BoW:** Wins all 6 courses (100%).  
**Node2Vec vs Spectral:** Wins all 6 courses (100%).

---

## 4. Statistical Comparisons

| Comparison | Mean diff | Cliff's δ | Effect size r |
|-----------|-----------|-----------|---------------|
| N2V(1,1) vs BoW | +0.472 | 1.000 | 0.899 |
| N2V(1,1) vs PCA | +0.163 | 0.889 | 0.899 |
| N2V(1,1) vs Spectral | +0.501 | 1.000 | 0.899 |
| N2V(1,0.5) vs BoW | +0.462 | 1.000 | 0.899 |
| N2V(1,0.5) vs PCA | +0.154 | 0.778 | 0.899 |
| N2V(1,1) vs N2V(1,0.5) | +0.010 | 0.278 | 0.471 |

All graph-vs-baseline comparisons show **large effect sizes** (r = 0.899).

---

## 5. Figures Generated (11 figures, PDF + PNG each)

All saved to `results/reproduced/figures/`:

| # | Figure | Description |
|---|--------|-------------|
| 1 | `baseline_comparison` | Bar chart: 5 methods × 6 courses (Silhouette) |
| 2 | `method_comparison_all_metrics` | Grouped bars: Silhouette, DBI, CHI for all methods |
| 3 | `silhouette_score_comparison_all_files` | Silhouette comparison across courses |
| 4 | `DBI_comparison_all_files` | Davies-Bouldin Index comparison |
| 5 | `CHI_comparison_all_files` | Calinski-Harabasz Index comparison |
| 6 | `repro_silhouette_vs_d` | Silhouette vs embedding dimension |
| 7 | `sensitivity_heatmap` | (p,q) parameter sensitivity heatmap |
| 8 | `stability_boxplot` | Node2Vec stability across 20 seeds |
| 9 | `graph_density_comparison` | Graph density before/after preprocessing |
| 10 | `runtime_comparison` | Runtime comparison across methods |

**DeepWalk-SSP is absent from all figures.**

---

## 6. JSON Files Saved

All in `results/reproduced/`:

- `all_methods.json` — Main 5×6 comparison
- `runtime.json` — Runtime per method per course
- `sensitivity_pq.json` — (p,q) grid sweep
- `sensitivity_dim.json` — Embedding dimension sweep
- `sensitivity_wl.json` — Walk length sweep
- `sensitivity_nw.json` — Number of walks sweep
- `sensitivity_ws.json` — Window size sweep
- `stability.json` — 20-seed stability per course
- `ari_stability.json` — ARI for KMeans across seeds
- `statistical_analysis.json` — Wilcoxon + effect sizes
- `graph_stats.json` — Graph structure statistics

---

## 7. Note on Manuscript Numbers

The manuscript (`paper/sn-article.tex`) currently contains numbers from a **previous run** (seed=0, different random streams). The reproduced numbers above differ slightly but show the same qualitative conclusions. The manuscript tables and text should be updated to match these reproduced results before final submission.
