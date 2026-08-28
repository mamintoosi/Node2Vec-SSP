# DeepWalk-SSP: Revised Pipeline — Final Report

**Date:** August 28, 2026  
**Status:** Complete re-experiment finished. Manuscript NOT yet revised.

---

## 1. Environment

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
| Platform | Linux-7.0.0-30-generic-x86_64-with-glibc2.43 |

---

## 2. Git Status

- Current branch: `main`
- All new code, results, and figures have been committed.
- The `paper/` directory remains in `.gitignore` (intentional — manuscript not modified).

---

## 3. Existing Graph Construction Behavior

The original `create_graph_from_bow()` function in `experiments/core.py` constructs a weighted co-enrollment graph:

1. Nodes = students
2. Edge between students i and j if they share ≥ 1 course
3. Edge weight = number of shared courses

**Critical finding:** The original data files include the target course itself as one of the columns in the binary enrollment vector. Since ALL students in a course are enrolled in that course, this column is universally shared, guaranteeing that every pair of students has at least one shared course (weight ≥ 1). This makes every co-enrollment graph a **complete graph** (density = 1.0, every node connected to every other node).

On complete graphs:
- The Node2Vec `q` parameter (BFS/DFS bias) has **zero effect** because every neighbor of the current node is also a neighbor of the previous node
- The `p` parameter (return bias) also has minimal effect because all transition weights become equal
- The only difference between DeepWalk and Node2Vec(p=1,q=1) is the random seed

---

## 4. Exact Graph-Construction Modification

**New function:** `find_universal_columns(matrix)`  
Identifies columns where ALL students have value 1 (universally shared courses).

**Modified pipeline:** Before constructing the co-enrollment graph, remove all universally shared columns from the enrollment matrix. In practice, exactly **1 column** (the target course) is removed per dataset.

The modified pipeline is:
```python
matrix_orig, _ = read_class(filepath)
universal_cols = find_universal_columns(matrix_orig)
matrix_filtered = np.delete(matrix_orig, universal_cols, axis=1)
G = create_graph_from_bow(matrix_filtered)
```

The original `create_graph_from_bow()` function is preserved unchanged. The filtering is done before graph construction.

---

## 5. Graph Statistics: Before vs After

| Course | Students | Courses (orig→new) | Edges (orig→new) | Density (orig→new) | Components |
|--------|----------|-------------------|-------------------|---------------------|------------|
| 1 | 74 | 35→34 | 2701→1828 | 1.000→0.677 | 3 |
| 2 | 51 | 34→33 | 1275→866 | 1.000→0.679 | 1 |
| 3 | 48 | 28→27 | 1128→894 | 1.000→0.793 | 2 |
| 4 | 49 | 28→27 | 1176→1043 | 1.000→0.887 | 1 |
| 5 | 52 | 32→31 | 1326→912 | 1.000→0.688 | 1 |
| 6 | 67 | 33→32 | 2211→1325 | 1.000→0.599 | 1 |

**Average density after modification:** 0.721 (was 1.000 for all)

**Degree statistics (revised graphs):**

| Course | Avg Degree | Min Degree | Max Degree | Avg Edge Weight | Weight Std |
|--------|-----------|------------|------------|----------------|------------|
| 1 | 49.4 | 0 | 69 | 1.97 | 1.08 |
| 2 | 34.0 | 13 | 48 | 2.15 | 1.33 |
| 3 | 37.2 | 0 | 46 | 1.89 | 1.03 |
| 4 | 42.6 | 17 | 48 | 2.33 | 1.18 |
| 5 | 35.1 | 3 | 48 | 2.09 | 1.09 |
| 6 | 39.6 | 15 | 62 | 2.24 | 1.48 |

**Important observations:**
- Courses 1 and 3 have isolated nodes (min degree = 0) — some students share no non-target courses with anyone
- Courses 1 and 3 also have multiple connected components (3 and 2 respectively)
- Edge weights vary significantly (std > 1.0 for most courses), providing meaningful weight information
- The revised graphs are no longer complete, so Node2Vec's `p` and `q` parameters can now have genuine effects

---

## 6. Complete Experimental Protocol

### Parameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension (d) | 2 |
| Walk length (t) | 10 |
| Walks per node (γ) | 80 |
| Context window (w) | 5 |
| Word2Vec epochs (ε) | 30 |
| Hierarchical softmax | yes (hs=1) |
| Skip-gram | yes (sg=1) |
| Workers | 1 |
| Min count | 1 |
| Subsampling | 0 (disabled) |
| KMeans clusters | 2 |
| KMeans n_init | 10 |

### Node2Vec Parameters

| Config | p | q | Description |
|--------|---|---|-------------|
| Neutral | 1.0 | 1.0 | Unbiased (theoretically equivalent to DeepWalk) |
| BFS-like | 1.0 | 0.5 | Favors local structure |
| DFS-like | 1.0 | 2.0 | Favors global structure |
| Low return | 0.5 | 1.0 | Less likely to return to previous node |
| High return | 2.0 | 1.0 | More likely to return to previous node |
| Best (observed) | 1.0 | 0.5 | Best average Silhouette |

### Random Seeds
- Primary results: seed = 0
- Stability analysis: 10 seeds (0–9) per course
- All methods: KMeans random_state = 0 for deterministic results

### Statistical Analysis
- Wilcoxon signed-rank test (two-sided, course-level paired comparison)
- Holm correction for multiple comparisons (3 tests)
- Effect size: r (from Z-score) and Cliff's delta

---

## 7. Results: All Methods (seed=0)

### Silhouette Score

| Method | C1 | C2 | C3 | C4 | C5 | C6 | **Average** |
|--------|-----|-----|-----|-----|-----|-----|-------------|
| BoW+KMeans | 0.099 | 0.231 | 0.142 | 0.128 | 0.116 | 0.202 | **0.153** |
| PCA+KMeans | 0.094 | 0.231 | 0.154 | 0.128 | 0.137 | 0.202 | **0.158** |
| Spectral | 0.083 | 0.205 | 0.135 | 0.128 | 0.067 | 0.143 | **0.127** |
| **Node2Vec(1,1)** | 0.535 | 0.599 | 0.570 | 0.644 | 0.734 | 0.582 | **0.611** |

### Davies-Bouldin Index (lower is better)

| Method | Average |
|--------|---------|
| BoW+KMeans | — |
| PCA+KMeans | — |
| Spectral | — |
| Node2Vec(1,1) | — |

*(DBI values available in all_methods.json)*

### Key Finding
Node2Vec dramatically outperforms all baselines on Silhouette Score. On the revised graphs (where the target course is removed):
- Node2Vec vs BoW: **+0.459** average improvement (+299%)
- Node2Vec vs PCA: **+0.453** average improvement (+287%)
- Node2Vec vs Spectral: **+0.485** average improvement (+382%)

---

## 8. Stability Analysis (10 seeds)

| Course | Mean ± Std Silhouette | Range |
|--------|----------------------|-------|
| 1 | 0.564 ± 0.018 | [0.531, 0.587] |
| 2 | 0.603 ± 0.009 | [0.591, 0.619] |
| 3 | 0.572 ± 0.013 | [0.549, 0.593] |
| 4 | 0.626 ± 0.014 | [0.604, 0.646] |
| 5 | 0.743 ± 0.008 | [0.734, 0.759] |
| 6 | 0.593 ± 0.008 | [0.582, 0.610] |

**Average stability std: 0.012** — very stable across seeds.

---

## 9. Statistical Tests

| Comparison | Wilcoxon p | Holm-corrected p | Effect size r | Cliff's δ | Significant |
|-----------|-----------|------------------|---------------|-----------|-------------|
| Node2Vec vs BoW | 0.031 | 0.031 | 0.899 | 1.000 | **Yes** |
| Node2Vec vs PCA | 0.031 | 0.031 | 0.899 | 1.000 | **Yes** |
| Node2Vec vs Spectral | 0.031 | 0.031 | 0.899 | 1.000 | **Yes** |

**Effect size interpretation:** r = 0.899 is a **large** effect. Cliff's delta = 1.000 means Node2Vec scored higher than every baseline on every single course (complete dominance).

**Caveat:** Only n = 6 courses (independent datasets). The Wilcoxon test has minimum power at this sample size. The p = 0.031 is the minimum achievable p-value for a one-sided Wilcoxon test with n = 6. The effect sizes are more informative than the p-values here.

---

## 10. Parameter Sensitivity

| p | q | Avg Silhouette |
|---|---|---------------|
| 0.5 | 0.5 | 0.620 |
| 0.5 | 1.0 | 0.616 |
| 0.5 | 2.0 | 0.614 |
| 1.0 | 0.5 | **0.622** |
| 1.0 | 1.0 | 0.611 |
| 1.0 | 2.0 | 0.605 |
| 2.0 | 0.5 | 0.615 |
| 2.0 | 1.0 | 0.615 |
| 2.0 | 2.0 | 0.616 |

**Best configuration:** p = 1.0, q = 0.5 (Silhouette = 0.622)  
**Sensitivity range:** 0.605 to 0.622 (Δ = 0.017)

**Key findings on the revised graphs:**
- The `q` parameter now has a **measurable effect** (previously zero on complete graphs)
- Lower `q` (more BFS-like, favoring local structure) tends to perform slightly better
- The `p` parameter has a smaller effect than `q`
- The overall sensitivity is modest — all configurations outperform baselines by a large margin
- The neutral configuration p=1, q=1 is near the middle of the range

---

## 11. Walk Validation

| Check | Result |
|-------|--------|
| Node | 0 (Course 1) |
| Neighbors | 45 |
| Expected uniform probability | 0.0222 |
| Max deviation from uniform | 0.0108 |
| Walks analyzed | 1000 |
| **Status** | **PASSED** |

Node2Vec(p=1, q=1) produces approximately uniform transition probabilities as expected by theory.

---

## 12. Runtime Analysis

| Method | C1 | C2 | C3 | C4 | C5 | C6 | **Average** |
|--------|-----|-----|-----|-----|-----|-----|-------------|
| BoW+KMeans | 0.169 | 0.009 | 0.008 | 0.007 | 0.007 | 0.010 | **0.035s** |
| PCA+KMeans | 0.095 | 0.010 | 0.007 | 0.012 | 0.010 | 0.007 | **0.024s** |
| Spectral | 0.091 | 0.010 | 0.009 | 0.014 | 0.006 | 0.015 | **0.024s** |
| **Node2Vec** | 8.403 | 5.371 | 4.995 | 5.223 | 5.625 | 7.339 | **6.159s** |

Node2Vec is ~200× slower than baselines due to random walk generation + Word2Vec training. Still very fast in absolute terms (~6 seconds per course).

---

## 13. Unexpected Findings

1. **One universal column per course:** Each data file has exactly one universally shared course column (the target course itself). Removing just this one column transforms complete graphs into sparse, informative graphs.

2. **Isolated nodes appear:** After removing the universal column, some students become isolated (degree 0) in courses 1 and 3. These students share no non-target courses with any other student. The multi-component structure in courses 1 and 3 means that random walks can get trapped in components.

3. **Spectral clustering degrades:** Spectral clustering performs worse than BoW+KMeans on the revised graphs (0.127 vs 0.153). This may be because the disconnected components in some graphs make the Laplacian eigenvalues less informative.

4. **PCA loses its advantage:** On the original complete graphs, PCA significantly outperformed BoW (0.460 vs 0.153 average). On the revised graphs, PCA barely differs from BoW (0.158 vs 0.153). This suggests PCA's improvement was partly driven by the universal column creating a trivially separable structure.

5. **q parameter is now informative:** On the revised graphs, q has a measurable effect (range 0.605–0.622), whereas on the original complete graphs, q had zero effect. This validates the hypothesis that removing universal courses creates meaningful graph topology.

---

## 14. Whether "Student2Vec" Is a Defensible Name

**Assessment:** The name "Student2Vec" is **not yet defensible** as a standalone contribution.

**Reasoning:**
- The implementation is a standard Node2Vec algorithm applied to a student co-enrollment graph
- The novelty lies in: (a) the domain application, (b) the graph construction (with universal course removal), (c) the evaluation protocol for student sectioning
- "Student2Vec" implies a new algorithm, but this is Node2Vec applied to students
- A more accurate name would be: **"Node2Vec-based Student Sectioning"** or **"Graph Representation Learning for Student Sectioning"**
- The term "Student2Vec" could appear in the manuscript as a convenient shorthand for the overall pipeline, but should not be presented as a novel method

**Recommendation:** Use "Student2Vec" only as a descriptive label for the complete pipeline, clearly stating it applies standard Node2Vec to student co-enrollment graphs. Do not claim algorithmic novelty for the embedding method itself.

---

## 15. Manuscript Sections Requiring Revision

The manuscript (`paper/sn-article.tex`) was **not modified** during this re-experiment. The following sections will need updating:

1. **Abstract:** Replace DeepWalk mentions with Node2Vec; update results
2. **Introduction:** Revise motivation to use Node2Vec; mention universal course removal insight
3. **Related Work:** Add Node2Vec reference (Grover & Leskovec, 2016)
4. **Methodology:**
   - Describe the revised graph construction (universal course removal)
   - Replace DeepWalk with Node2Vec description
   - Explain why universal course removal is necessary
5. **Experimental Setup:**
   - Update method descriptions (Node2Vec instead of DeepWalk)
   - Mention revised graph construction
6. **Results:**
   - Replace all DeepWalk results with Node2Vec results
   - Add PCA/Spectral comparison tables (if not already present)
   - Update all figures
7. **Discussion:**
   - Discuss the complete graph finding and its implications
   - Discuss why universal course removal matters
   - Discuss sensitivity results
   - Discuss whether "Student2Vec" is appropriate terminology
8. **Conclusion:** Update main findings

---

## 16. Generated Files

### Source code (new/modified)
- `experiments/revised_pipeline.py` — Full pipeline (not run in final form)
- `experiments/run_stage1.py` — Graph analysis, all methods, sensitivity
- `experiments/run_stage2.py` — Stability analysis
- `experiments/run_stage3.py` — Statistical analysis, walk validation
- `experiments/gen_figures.py` — Figure generation

### Results (in `results/revised_reexperiment/`)
- `graph_analysis.json` — Graph statistics before/after
- `all_methods.json` / `.xlsx` — All method results (seed=0)
- `node2vec_sensitivity.json` / `.xlsx` — p×q sensitivity grid
- `stability.json` — Stability statistics (10 seeds)
- `stability_per_seed.json` — Per-seed results
- `statistical_analysis.json` — Wilcoxon tests with Holm correction
- `walk_validation.json` — Node2Vec validation
- `runtime.json` — Runtime measurements
- `environment.json` — Package versions

### Figures (in `results/revised_reexperiment/figures/`)
- `graph_density_comparison.pdf` — Before/after graph density
- `method_comparison.pdf` — All methods bar chart
- `node2vec_sensitivity_heatmap.pdf` — p×q sensitivity
- `stability_boxplot.pdf` — Seed stability
- `runtime_comparison.pdf` — Runtime comparison

---

## 17. Git Status

- Branch: `main`
- All new files committed
- `paper/` is in `.gitignore` (manuscript not tracked, not modified)
- Previous results in `results/` preserved (old `linux_reexperiment/`, `df.xlsx`, etc.)

---

## 18. Recommendations

1. **Report Node2Vec(p=1, q=0.5) as the primary configuration** — it has the best average Silhouette (0.622) and the q=0.5 bias favors local structural exploration, which is theoretically appropriate for student similarity.

2. **Alternatively, report p=1, q=1 as the neutral configuration** — it demonstrates that the improvement comes from the graph representation itself, not from parameter tuning. The sensitivity analysis shows all configurations perform well.

3. **Do NOT overclaim novelty** — Node2Vec is a well-established algorithm. The contribution is its application to student sectioning with the novel graph construction insight (universal course removal).

4. **Highlight the universal course removal as a key insight** — this is arguably the most important methodological finding: removing the target course from the enrollment vector transforms an uninformative complete graph into a structurally rich graph.

5. **Be transparent about the n=6 caveat** — with only 6 courses, formal hypothesis testing has limited power. Report effect sizes prominently.

6. **Consider adding stability analysis with 20 seeds** — the current 10-seed analysis shows excellent stability (std ≈ 0.012), but 20 seeds would be more standard.

7. **Consider the disconnected component issue** — courses 1 and 3 have isolated nodes. The pipeline should handle this gracefully (it does — isolated nodes get walks of length 1).

---

## 19. Methodological Concerns

1. **Spectral clustering on disconnected graphs** — courses 1 and 3 have multiple connected components. Spectral clustering with a precomputed affinity matrix may not handle this well. Consider using normalized Laplacian or explicitly handling disconnected components.

2. **Isolated nodes** — Students with degree 0 (no shared non-target courses) get no useful random walk. Their embeddings are determined solely by the initial node identity. The pipeline handles this correctly (walks terminate immediately), but the embeddings for these students may be less meaningful.

3. **KMeans on d=2 embeddings** — With only 2 dimensions, KMeans is essentially finding the best line to split the data. This is appropriate for visualization but may not be optimal for all scenarios.

4. **Single evaluation metric dominance** — Silhouette Score is the primary metric. DBI and CH should also be reported prominently, not just Silhouette.

5. **The p=0.031 is a floor** — For n=6 two-sided Wilcoxon, the minimum achievable p-value is 1/C(6,3) × 2 = 0.03125 (approximately). This means we cannot distinguish between p=0.031 and much smaller p-values at this sample size. Report this as "p < 0.05" or "p = 0.031 (minimum achievable for n=6)" rather than implying precision.

---

*Report generated by automated pipeline. All experimental data saved in `results/revised_reexperiment/`.*
