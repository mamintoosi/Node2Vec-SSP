# DeepWalk-SSP: Complete Linux Re-Experiment Report

**Date:** August 27, 2026  
**Platform:** Linux (Python 3.12.13)  
**Purpose:** Complete re-experiment of all methods under a consistent Linux environment, adding Node2Vec as a comparative graph representation learning method.

---

## A. Environment

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| gensim | 4.4.0 |
| networkx | 3.6.1 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| Platform | Linux-7.0.0-30-generic-x86_64-with-glibc2.43 |

---

## B. Reproducibility: Old Windows vs New Linux

### DeepWalk Silhouette (d=2)

| Course | Old (Windows) | New (Linux) | Difference | % Change |
|--------|--------------|-------------|------------|----------|
| 1 | 0.5805 | 0.5824 | +0.0019 | +0.3% |
| 2 | 0.5617 | 0.5193 | -0.0424 | -7.5% |
| 3 | 0.5656 | 0.5525 | -0.0131 | -2.3% |
| 4 | 0.6103 | 0.5490 | -0.0613 | -10.0% |
| 5 | 0.5801 | 0.6218 | +0.0416 | +7.2% |
| 6 | 0.5801 | 0.5631 | -0.0170 | -2.9% |
| **Average** | **0.5797** | **0.5647** | **-0.0150** | **-2.6%** |

**Analysis:** DeepWalk results differ across platforms by up to 10%. This is expected because:
- gensim Word2Vec implementation has platform-specific RNG behavior
- Different gensim versions (4.3.x on Windows vs 4.4.0 on Linux)
- Different numpy versions affect random state progression

**PCA results are IDENTICAL** across platforms (deterministic algorithm).  
**BoW results differ** because KMeans initialization depends on RNG state.

### Important: The old baseline JSON results were generated during the same Windows session, not separately reproduced. The `df.xlsx` values and the `baseline_pca_kmeans.json` DeepWalk values differ because DeepWalk was run in different execution contexts.

---

## C. Final Comparison (Linux, seed=0)

### Silhouette Score (↑ higher is better)

| Course | BoW+KMeans | PCA+KMeans | Spectral | DeepWalk-SSP | Node2Vec(1,1) | Node2Vec(0.5,1) |
|--------|-----------|------------|----------|-------------|---------------|-----------------|
| 1 (74) | 0.0990 | 0.3572 | 0.0951 | 0.5824 | 0.5631 | **0.5913** |
| 2 (51) | 0.2313 | 0.6032 | 0.2123 | 0.5193 | 0.6519 | **0.6610** |
| 3 (48) | 0.1421 | 0.4253 | 0.1347 | 0.5525 | **0.6228** | 0.5866 |
| 4 (49) | 0.1280 | 0.4278 | 0.1280 | 0.5490 | **0.6776** | 0.6666 |
| 5 (52) | 0.1165 | 0.4253 | 0.1377 | 0.6218 | 0.6937 | **0.7496** |
| 6 (67) | 0.2025 | 0.5238 | 0.1989 | 0.5631 | **0.7380** | 0.7356 |
| **Average** | **0.1532** | **0.4604** | **0.1511** | **0.5647** | **0.6579** | **0.6651** |

### Davies-Bouldin Index (↓ lower is better)

| Course | BoW+KMeans | PCA+KMeans | Spectral | DeepWalk-SSP | Node2Vec(1,1) | Node2Vec(0.5,1) |
|--------|-----------|------------|----------|-------------|---------------|-----------------|
| 1 | 2.8832 | 1.1437 | 2.9554 | 0.5361 | 0.5732 | **0.5510** |
| 2 | 1.3126 | 0.4890 | 1.5318 | 0.6496 | 0.5167 | **0.3861** |
| 3 | 2.3834 | 1.0470 | 2.4699 | 0.6220 | **0.5095** | 0.5308 |
| 4 | 2.3092 | 0.9146 | 2.3092 | 0.6242 | **0.4181** | 0.4199 |
| 5 | 2.5058 | 0.9391 | 2.1305 | 0.4996 | 0.4255 | **0.3664** |
| 6 | 1.6526 | 0.6735 | 1.6990 | 0.5846 | 0.3356 | **0.3253** |
| **Average** | **2.1745** | **0.8678** | **2.1826** | **0.5860** | **0.4631** | **0.4299** |

### Calinski-Harabasz Index (↑ higher is better)

| Course | BoW+KMeans | PCA+KMeans | Spectral | DeepWalk-SSP | Node2Vec(1,1) | Node2Vec(0.5,1) |
|--------|-----------|------------|----------|-------------|---------------|-----------------|
| 1 | 8.3 | 43.2 | 7.9 | 140.6 | 153.0 | **166.1** |
| 2 | 13.9 | 63.7 | 12.9 | 85.5 | 159.9 | **144.9** |
| 3 | 7.5 | 35.7 | 7.1 | 83.3 | **119.6** | 109.6 |
| 4 | 8.0 | 41.4 | 8.0 | 75.8 | **183.8** | 181.9 |
| 5 | 7.3 | 39.9 | 8.0 | 128.9 | 166.6 | **226.3** |
| 6 | 18.6 | 79.9 | 18.2 | 123.3 | **328.7** | 326.3 |
| **Average** | **10.6** | **50.6** | **10.4** | **106.2** | **185.3** | **192.5** |

---

## D. Node2Vec Configurations

### Average Silhouette by (p, q)

| p \ q | 0.50 | 1.00 | 2.00 |
|--------|------|------|------|
| 0.50 | 0.6651 | 0.6651 | 0.6651 |
| 1.00 | 0.6579 | 0.6579 | 0.6579 |
| 2.00 | 0.6697 | 0.6697 | 0.6697 |

**Critical finding:** The q parameter has **ZERO effect** on any course. For every fixed p value, the silhouette score is identical across all q values. This is because all six co-enrollment graphs are completely dense (density = 1.0), meaning every student shares at least one course with every other student. On a complete graph, every neighbor of a node is also a neighbor of every other neighbor, so the BFS/DFS bias controlled by q has no structural effect.

The p parameter does produce different results (range 0.6579–0.6697), but the differences are modest.

### Node2Vec Designation
- **Primary configuration:** p=1.0, q=1.0 (the theoretical equivalence to DeepWalk under uniform transitions)
- **Alternative configuration:** p=0.5, q=1.0 (reduced return probability)

---

## E. Graph Structure

| Course | Students | Courses | Edges | Max Possible | Density | Complete? |
|--------|----------|---------|-------|-------------|---------|-----------|
| 1 | 74 | 35 | 2,701 | 2,701 | 1.0000 | Yes |
| 2 | 51 | 34 | 1,275 | 1,275 | 1.0000 | Yes |
| 3 | 48 | 28 | 1,128 | 1,128 | 1.0000 | Yes |
| 4 | 49 | 28 | 1,176 | 1,176 | 1.0000 | Yes |
| 5 | 52 | 32 | 1,326 | 1,326 | 1.0000 | Yes |
| 6 | 67 | 33 | 2,211 | 2,211 | 1.0000 | Yes |

**All six graphs are complete.** Every student shares at least one course with every other student in the same sectioning problem.

### Implications for Node2Vec
1. **q parameter is irrelevant:** On complete graphs, all neighbors of any node are also neighbors of all other nodes, making the BFS/DFS distinction meaningless.
2. **p parameter still matters:** The return parameter p affects the probability of revisiting the previous node, which is meaningful even on complete graphs.
3. **Node2Vec(p=1,q=1) ≠ DeepWalk** in implementation because they use different code paths and RNG states, even though theoretically they should produce the same transition probabilities.

---

## F. Node2Vec Walk Validation

| Course | DW vs N2V(1,1) identity | DW vs N2V(0.5,1) identity | DW return rate | N2V(0.5,1) return rate |
|--------|------------------------|--------------------------|----------------|----------------------|
| 1 | 0.0000 | 0.0000 | 0.0136 | 0.0328 |
| 2 | 0.0000 | 0.0000 | 0.0203 | 0.0511 |
| 3 | 0.0000 | 0.0000 | 0.0218 | 0.0476 |
| 4 | 0.0000 | 0.0000 | 0.0213 | 0.0468 |
| 5 | 0.0000 | 0.0000 | 0.0200 | 0.0465 |
| 6 | 0.0000 | 0.0000 | 0.0150 | 0.0395 |

**Findings:**
1. DeepWalk and Node2Vec generate **completely different walks** (0% identity) even with p=1.0, q=1.0, confirming they are genuinely distinct code paths.
2. **Node2Vec(0.5,1) has HIGHER return rates than DeepWalk** (p=1.0), which seems counterintuitive. On a complete graph with n nodes, the return-to-previous probability is proportional to 1/p while forward steps are proportional to 1. With p=0.5, returning has weight 2.0 vs weight 1.0 for forward, making return **more** likely, not less. This is because p < 1 means the walk is biased **toward** the previous node (not away from it).

**Correction to earlier assumption:** p < 1 increases return probability; p > 1 decreases it. The parameter names in the Node2Vec paper: p is the "return parameter" where **higher p means LESS likely to return**.

---

## G. Clustering Stability

Stability measured as mean Silhouette ± std across 5 random seeds:

| Course | DeepWalk | Node2Vec(1,1) | Node2Vec(0.5,1) |
|--------|----------|---------------|-----------------|
| 1 | 0.581 ± 0.020 | 0.566 ± 0.022 | 0.573 ± 0.018 |
| 2 | 0.568 ± 0.039 | 0.665 ± 0.014 | 0.669 ± 0.020 |
| 3 | 0.594 ± 0.044 | 0.611 ± 0.024 | 0.613 ± 0.035 |
| 4 | 0.559 ± 0.025 | 0.649 ± 0.030 | 0.655 ± 0.016 |
| 5 | 0.580 ± 0.038 | 0.701 ± 0.016 | 0.717 ± 0.018 |
| 6 | 0.548 ± 0.015 | 0.729 ± 0.018 | 0.729 ± 0.018 |

**Observations:**
- Node2Vec variants have **lower variance** than DeepWalk on most courses
- Node2Vec(1,1) and Node2Vec(0.5,1) have very similar stability profiles
- DeepWalk has slightly higher variance, particularly on Course 3 (std=0.044)

---

## H. Statistical Analysis

Wilcoxon signed-rank tests (paired, one-sided, using 5-seed stability distributions):

| Comparison | Mean A | Mean B | Δ | p-value | Effect size r | 95% CI for Δ |
|-----------|--------|--------|---|---------|---------------|-------------|
| DeepWalk vs BoW | 0.5717 | 0.1532 | +0.4185 | 0.0156 * | 0.899 | [0.373, 0.462] |
| N2V(1,1) vs DeepWalk | 0.6534 | 0.5717 | +0.0818 | 0.0313 * | 0.813 | [0.031, 0.133] |
| N2V(0.5,1) vs DeepWalk | 0.6594 | 0.5717 | +0.0877 | 0.0313 * | 0.813 | [0.037, 0.139] |
| N2V(0.5,1) vs BoW | 0.6594 | 0.1532 | +0.5061 | 0.0156 * | 0.899 | [0.467, 0.551] |
| N2V(1,1) vs PCA | 0.6534 | 0.4639 | +0.1895 | 0.0156 * | 0.899 | [0.133, 0.234] |
| DeepWalk vs PCA | 0.5717 | 0.4639 | +0.1077 | 0.0469 * | 0.728 | [0.036, 0.171] |

### ⚠️ Statistical Methodology Concern

The statistical unit of analysis is **courses** (n=6), not individual seeds. Each course provides one paired comparison (method A score vs method B score), using the mean across seeds for each course. However:

1. **With only 6 courses, statistical power is very limited.** The Wilcoxon signed-rank test requires n ≥ 5 for a two-sided test, and n=6 is the bare minimum.
2. **The p-values should be interpreted with extreme caution.** With n=6, even a consistent moderate effect may not reach significance.
3. **The seeds from the same course are NOT independent courses** — they are repeated measurements of the same course. Using the mean across seeds as the paired observation is appropriate, but it reduces the effective sample size.
4. **Multiple comparisons correction is not applied.** With 6 comparisons, the family-wise error rate is inflated.

**Recommendation:** These statistical results should be reported as suggestive rather than conclusive. The effect sizes (Cohen's r > 0.7 for all comparisons) are meaningful, but the small n makes formal hypothesis testing underpowered.

---

## I. Figures Generated

All figures saved as PDF in `results/linux_reexperiment/figures/`:

| File | Description |
|------|-------------|
| `fig1_silhouette_comparison.pdf` | 6-way method comparison bar chart (Silhouette) |
| `fig1_silhouette_comparison.png` | PNG version for quick viewing |
| `fig2_dbi_comparison.pdf` | 6-way method comparison (DBI) |
| `fig3_ch_comparison.pdf` | 6-way method comparison (CH) |
| `fig4_node2vec_sensitivity.pdf` | p×q sensitivity heatmap |
| `fig5_seed_stability.pdf` | Box plots for DeepWalk, N2V(1,1), N2V(0.5,1) |
| `fig6_runtime_comparison.pdf` | Runtime comparison across methods |
| `fig7_node2vec_configs.pdf` | Line plot of 4 Node2Vec configs vs DeepWalk |
| `fig8_average_comparison.pdf` | Average Silhouette summary bar chart |
| `fig9_all_metrics_comparison.pdf` | All 3 metrics side-by-side |

---

## J. Files Modified / Created

### New files:
- `experiments/linux_reexperiment.py` — Complete re-experiment script (Section C only)
- `experiments/run_de.py` — Sections D+E: sensitivity + validation
- `experiments/run_fgh.py` — Sections F+G+H: stability + statistics + figures
- `results/linux_reexperiment/` — All new Linux results
- `doc/REPORT.md` — This report

### Modified files:
- `experiments/core.py` — Added `generate_node2vec_walks()` and `run_node2vec_pipeline()` (from earlier work)
- `experiments/config.py` — Added `NODE2VEC_PARAMS`, `NODE2VEC_Q_VALUES`, `COLORS['node2vec']` (from earlier work)
- `experiments/add_baselines.py` — Added `experiment_node2vec()`, updated imports and figures (from earlier work)

### Original results preserved:
- `results/df.xlsx` — Original Windows DeepWalk results
- `results/baseline_*.json` — Original Windows baseline results
- `results/exp_*.json` — Original Windows experiment results
- `results/figures/` — Original Windows figures
- `results/node2vec_comparison.json` — Previous Node2Vec comparison (not re-run from scratch on Linux)

---

## K. Problems and Issues

1. **Platform reproducibility:** DeepWalk results differ by up to 10% between Windows and Linux due to gensim version differences (4.3.x → 4.4.0) and platform-specific RNG behavior. PCA results are perfectly reproducible.

2. **Complete graphs make q irrelevant:** All six co-enrollment graphs have density 1.0, rendering Node2Vec's q parameter meaningless. This is a property of the dataset, not an implementation issue. Any paper claiming Node2Vec's BFS/DFS bias provides additional information beyond DeepWalk on these datasets would be misleading.

3. **Node2Vec ≠ DeepWalk at p=1,q=1:** Despite theoretical equivalence, the implementations produce completely different walks (0% identity) due to different code paths and RNG states. The performance difference between N2V(1,1) and DeepWalk is therefore due to implementation details (different Word2Vec initialization), not different walk strategies.

4. **BoW Course 4 discrepancy:** Old Windows BoW=0.0618 vs new Linux BoW=0.1280. This is due to KMeans initialization differences.

5. **Statistical power:** With only 6 courses as the unit of analysis, formal statistical tests have very low power. All p-values should be interpreted with caution.

---

## L. Recommendations

### Which results to report in the paper:
Use the **new Linux results** for all methods, since they are generated under a single consistent environment. Report both old Windows values and new Linux values in a supplementary comparison table if cross-platform reproducibility is discussed.

### Which Node2Vec configuration:
- **Primary:** Node2Vec(p=1.0, q=1.0) — this is the theoretically comparable configuration to DeepWalk
- **Secondary:** Note that p=0.5 gives slightly different results due to increased return bias
- **Do not claim** that q provides meaningful differentiation on these datasets

### Does Node2Vec strengthen the paper?
**Yes, but with important caveats:**
1. Node2Vec(1,1) outperforms DeepWalk (avg 0.6579 vs 0.5647), which initially seems positive
2. However, since p=1,q=1 should theoretically produce identical walks to DeepWalk, the difference is caused by **different Word2Vec training dynamics** from different random walks, not by a fundamentally better algorithm
3. The stronger finding is that **both graph embedding methods** (DeepWalk and Node2Vec) substantially outperform PCA, BoW, and Spectral Clustering
4. The paper should emphasize that graph-based representation learning (regardless of the specific walk strategy) captures meaningful student co-enrollment structure

### Issues to resolve before manuscript revision:
1. **Decide which platform's results to report** (recommend: Linux, since it's the current environment)
2. **Clarify the complete graph finding** — this is actually an interesting methodological observation about student sectioning datasets
3. **Acknowledge the platform dependency** of DeepWalk/Node2Vec results in the reproducibility section
4. **Reconsider the statistical analysis** — with n=6, focus on effect sizes and confidence intervals rather than p-values
5. **Do not claim Node2Vec's q parameter provides additional value** — the complete graph structure makes this claim unsupported
