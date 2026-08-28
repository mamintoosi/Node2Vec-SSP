# Technical Audit Report: DeepWalk-SSP / Node2Vec Pipeline

**Date:** August 28, 2026  
**Status:** Complete audit. Manuscript NOT modified.  
**Auditor:** Buffy (Codebuff agent)

---

## Executive Summary

The experimental pipeline is **scientifically defensible** with one implementation bug that does not affect the primary conclusions. The main findings — that graph representation learning methods (Node2Vec/DeepWalk) significantly outperform conventional baselines (BoW, PCA, Spectral) on student sectioning — are robust.

**One implementation bug was found** in the Holm correction function (`experiments/stage_e.py`). The bug was in the monotonicity enforcement (backward sweep with `min` instead of forward cumulative maximum with `max`). The corrected p-values are 0.1875 instead of the previous 0.0625, but since both values are above α=0.05, the scientific conclusions are unchanged. **This bug has been fixed and Stage E rerun.**

**Recommendation: READY FOR MANUSCRIPT REVISION** — Holm correction bug has been fixed.

---

## 1. Isolated Student Handling

### Finding
Courses 1 and 3 contain isolated students after filtering:
- **Course 1**: 2 isolated students (indices 29, 30) out of 74
- **Course 3**: 1 isolated student (index 13) out of 48

All three isolated students have **all-zero rows** after filtering — their only enrolled course was the universal column, which was removed.

### Code Path for Isolated Students

1. **Walk generation**: `generate_node2vec_walks()` starts a walk from the isolated node. Since `G.neighbors(isolated)` returns `[]`, the walk loop `for _ in range(walk_length-1)` immediately breaks. The walk is `[node]` — a length-1 singleton.

2. **Word2Vec training**: The singleton walk `[node]` IS included in the training corpus. Word2Vec creates a vocabulary entry for the node (since `min_count=1`). The hierarchical softmax trains an embedding from the single-token context (node appears as both input and context in a window of size 5). The resulting embedding is **non-zero** but essentially noise — it depends on the RNG state and walk position in the corpus.

3. **KMeans clustering**: All students (including isolated) receive embeddings and are clustered. The isolated student's embedding is random but low-dimensional (d=2).

4. **Evaluation metrics**: Silhouette, DBI, and CH are computed over ALL students including isolated ones.

### Impact Assessment

| Course | Isolated Students | Sil (all) | Sil (without isolated) | Difference |
|--------|-------------------|-----------|------------------------|------------|
| 1 | 2/74 (2.7%) | 0.5347 | 0.5331 | -0.0016 |
| 3 | 1/48 (2.1%) | 0.5704 | 0.5693 | -0.0011 |

**Impact: Negligible.** The difference is < 0.2% in Silhouette Score.

### Correctness Assessment

The isolated students represent a genuine edge case: students who share NO courses with any other student after filtering. They are valid members of the sectioning problem and should be included in the evaluation. The current handling is acceptable because:

1. Isolated students are few (3 total across all courses)
2. They are included in ALL methods equally (BoW, PCA, Spectral, Node2Vec)
3. The impact on metrics is negligible
4. Removing them would artificially inflate scores

**Severity: LOW** — No action needed for the paper, but should be mentioned in the methodology.

---

## 2. Data Leakage Audit

### Target Course Identification

The data format is: `index student_id binary_course_vector`

Each file represents one sectioning instance. The binary vector encodes enrollment in OTHER courses (not the target course itself). **The target course is NOT represented as a feature in the data.**

### Verification

For each course, column 0 (which might naively be assumed to be the target course) has LOW enrollment:
- Course 1: 1/74 (1.4%)
- Course 2: 2/51 (3.9%)
- Course 3: 10/48 (20.8%)
- Course 4: 7/49 (14.3%)
- Course 5: 1/52 (1.9%)
- Course 6: 5/67 (7.5%)

None of these are 100%, confirming column 0 is NOT the target course.

### Universal Columns

Each file has exactly ONE universal column (100% enrollment):
- Course 1: column 3 (the only course ALL 74 students take)
- Course 2: column 8
- Course 3: column 13
- Course 4: column 19
- Course 5: column 22
- Course 6: column 25

These are correctly removed as constant features. They are NOT the target course.

### Leakage Assessment

**No data leakage exists.** The target course is not in the feature set. The universal columns are correctly removed. All methods receive the same filtered data.

**Severity: NONE** — Pipeline is clean.

---

## 3. Preprocessing Consistency

### Verification

All methods receive identical data from `load_course_data()`:
- **BoW + KMeans**: Uses `course_data[i]["matrix"]` (filtered)
- **PCA + KMeans**: Uses `course_data[i]["matrix"]` (filtered), reduces to d=2
- **Spectral**: Uses `course_data[i]["graph"]` (built from filtered matrix)
- **DeepWalk**: Uses `course_data[i]["graph"]` (built from filtered matrix)
- **Node2Vec**: Uses `course_data[i]["graph"]` (built from filtered matrix)

### Matrix Consistency Check

| Course | Matrix Match | Graph Match |
|--------|-------------|-------------|
| 1 | True | True |
| 2 | True | True |
| 3 | True | True |
| 4 | True | True |
| 5 | True | True |
| 6 | True | True |

**All methods use identical filtered data.** No method receives information unavailable to others.

**Severity: NONE** — Preprocessing is consistent.

---

## 4. Graph Construction Audit

### Implementation

```python
def create_graph(matrix):
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n): G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            w = int(np.sum(matrix[i] * matrix[j]))
            if w > 0: G.add_edge(i, j, weight=w)
    return G
```

### Properties Verified

| Course | Nodes | Edges | Manual Verify | Max Edges | Density | Self-loops |
|--------|-------|-------|---------------|-----------|---------|------------|
| 1 | 74 | 1828 | 1828 ✓ | 2701 | 0.6768 | 0 |
| 2 | 51 | 866 | 866 ✓ | 1275 | 0.6792 | 0 |
| 3 | 48 | 894 | 894 ✓ | 1128 | 0.7926 | 0 |
| 4 | 49 | 1043 | 1043 ✓ | 1176 | 0.8869 | 0 |
| 5 | 52 | 912 | 912 ✓ | 1326 | 0.6878 | 0 |
| 6 | 67 | 1325 | 1325 ✓ | 2211 | 0.5993 | 0 |

### Key Properties

- **Edge definition**: Two students are connected if they share ≥ 1 course (after filtering)
- **Edge weight**: Number of shared courses (integer, range 1-10)
- **Weighted**: Yes, weights are used in Node2Vec transition probabilities
- **Self-loops**: None
- **Duplicate edges**: Not possible (undirected simple graph)
- **Isolated vertices**: Preserved (added with `add_node` before edges)

### Graph Density

All graphs are **non-complete** after filtering. Density ranges from 0.599 to 0.887.

**Severity: NONE** — Graph construction is correct and consistent.

---

## 5. Node2Vec Implementation Audit

### Transition Probability Verification

The implementation correctly follows Grover & Leskovec (2016):

```
α_{pq}(t, x) =
    1/p     if d_t(x) = 0  (return to previous node)
    1       if d_t(x) = 1  (neighbor of previous node)
    1/q     if d_t(x) = 2  (non-neighbor of previous node)
```

### Empirical Verification

Test graph: 0--1--2--3 (chain, with weights)

**q parameter test** (from node 1, prev=2):
- Node 0 is NOT a neighbor of prev=2 → gets α = 1/q
- Node 2 IS the previous node → gets α = 1/p

| q | Theory P(0) | Empirical P(0) | Theory P(2) | Empirical P(2) |
|---|-------------|----------------|-------------|----------------|
| 0.5 | 0.8000 | 0.8018 | 0.2000 | 0.1982 |
| 1.0 | 0.6667 | 0.6682 | 0.3333 | 0.3318 |
| 2.0 | 0.5000 | 0.5020 | 0.5000 | 0.4980 |

**p parameter test** (from node 1, prev=2):

| p | Theory P(0) | Empirical P(0) | Theory P(2) | Empirical P(2) |
|---|-------------|----------------|-------------|----------------|
| 0.5 | 0.5000 | 0.5020 | 0.5000 | 0.4980 |
| 1.0 | 0.6667 | 0.6682 | 0.3333 | 0.3318 |
| 2.0 | 0.8000 | 0.8018 | 0.2000 | 0.1982 |

**All empirical values match theory within expected sampling noise.**

### Edge Weight Integration

Edge weights are correctly incorporated: `alpha * G[cur][z].get('weight', 1)`. This biases walks toward students who share more courses, which is appropriate for the student sectioning problem.

### Direction Semantics

- `p > 1` → less likely to return to previous node ✓
- `p < 1` → more likely to return ✓
- `q < 1` → BFS-like (prefers local neighbors) ✓
- `q > 1` → DFS-like (prefers distant nodes) ✓

**Severity: NONE** — Implementation is correct.

---

## 6. Parameter Sensitivity Audit

### Grid Execution

All 9 configurations were executed with seed=0:

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

### Consistency Check

- Same graph/data for all 9 configs ✓
- Same embedding dimension (d=2), walk length (t=10), num walks (γ=80), window (w=5), epochs (ε=30) ✓
- Same clustering procedure (KMeans, k=2, n_init=10) ✓
- Best config (p=1, q=0.5) selected AFTER examining predefined grid ✓

### Sensitivity Range

- Range: 0.605 to 0.622 (Δ = 0.017)
- All configurations outperform all baselines (best baseline PCA = 0.158)
- Lower q (BFS-like) slightly favors local structure

**Severity: NONE** — Sensitivity analysis is correct and defensible.

---

## 7. DeepWalk Equivalence Audit

### Theoretical

Node2Vec(p=1, q=1) produces **uniform transition probabilities**, which is mathematically equivalent to DeepWalk's unbiased random walks.

### Empirical

The implementation generates different random sequences because it uses a different code path and RNG state. This does NOT constitute an algorithmic difference.

| Course | Node2Vec (p=1,q=1) | DeepWalk | Difference |
|--------|-------------------|----------|------------|
| 1 | 0.563 | 0.561 | +0.002 |
| 2 | 0.606 | 0.611 | -0.005 |
| 3 | 0.572 | 0.562 | +0.010 |
| 4 | 0.623 | 0.573 | +0.050 |
| 5 | 0.743 | 0.774 | -0.031 |
| 6 | 0.599 | 0.634 | -0.035 |
| **Avg** | **0.618** | **0.619** | **-0.001** |

**Statistical test**: W=10.0, p=1.000, r=0.043 (negligible)

The avg Silhouette difference is 0.001. Per-course differences are due to different RNG streams.

**Severity: NONE** — Correctly presented as equivalent methods.

---

## 8. Randomness / Reproducibility Audit

### Seed Behavior

| Component | Seed Control | Deterministic? |
|-----------|-------------|----------------|
| Walk generation | `np.random.RandomState(seed)` | Yes (local RNG) |
| Word2Vec | `seed=seed` parameter | Yes (gensim internal) |
| KMeans | `random_state=seed` | Yes (sklearn internal) |

### Pipeline Reproducibility

- Same seed → identical walks, embeddings, labels, metrics ✓
- Different seed → different results ✓
- RNG state isolation verified (no leakage between calls) ✓

### Stability Data

- 120 entries: 20 seeds × 6 courses ✓
- Seeds: 0-19 ✓
- Courses: 1-6 ✓

### Important Note

The same integer seed is used for walks, Word2Vec, AND KMeans. This means the stability analysis measures **overall pipeline variability** (walk + embedding + clustering noise combined), not walk noise alone. This is acceptable for the paper's purpose but should be noted.

**Severity: NONE** — Reproducibility is properly controlled.

---

## 9. Statistical Analysis Audit

### Test Selection

- Wilcoxon signed-rank test (two-sided): appropriate for paired non-parametric comparison ✓
- Course-level analysis (n=6): correct statistical unit ✓
- Seeds treated as repeated measures, NOT independent observations ✓

### P-value Floor

With n=6 paired samples:
- Minimum achievable two-sided Wilcoxon p-value = 2/2⁶ = 0.03125
- This is an inherent limitation of the small sample size

### Holm Correction — BUG FOUND AND FIXED ✓

The `holm_correction()` function in `stage_e.py` had an implementation bug.

**Root cause**: The backward monotonicity loop used `min()` which propagates the **smallest** value upward, rather than the correct `np.maximum.accumulate()` which propagates the **largest** value upward. This failed when all raw p-values were equal (as in our case with three comparisons at p=0.03125).

**Effect on actual data** (pre-fix):

| Comparison | Corrected p (buggy) | Corrected p (correct) |
|-----------|--------------------|-----------------------|
| Node2Vec vs DeepWalk | 1.0000 | 1.0000 |
| Node2Vec vs BoW | 0.0625 | **0.1875** |
| Node2Vec vs PCA | 0.0625 | **0.1875** |
| Node2Vec vs Spectral | 0.0625 | **0.1875** |
| DeepWalk vs BoW | 0.0625 | **0.1875** |
| DeepWalk vs PCA | 0.0625 | **0.1875** |

The buggy values (0.0625) were **more conservative** (smaller) than the correct values (0.1875). Both are above α=0.05. **Stage E has been rerun with the corrected implementation.**

**Impact on conclusions**: NONE. The paper already correctly notes that:
- No test reaches significance at α=0.05 after correction
- Effect sizes (r=0.899, δ=1.000) are more informative
- n=6 is too small for reliable hypothesis testing

**Severity: MEDIUM** — Implementation bug, but does not affect conclusions. Must be fixed before manuscript submission.

### Effect Sizes

| Comparison | Effect r | Cliff's δ | Interpretation |
|-----------|----------|-----------|----------------|
| Graph vs BoW | 0.899 | 1.000 | Large, graph better on ALL courses |
| Graph vs PCA | 0.899 | 1.000 | Large, graph better on ALL courses |
| Graph vs Spectral | 0.899 | 1.000 | Large, graph better on ALL courses |
| Node2Vec vs DeepWalk | 0.043 | 0.056 | Negligible (as expected) |

---

## 10. Evaluation Metrics Audit

### Metric Implementations

All metrics use `sklearn.metrics` standard implementations:
- `silhouette_score(data, labels)` — uses Euclidean distance by default
- `davies_bouldin_score(data, labels)` — uses Euclidean distance
- `calinski_harabasz_score(data, labels)` — uses Euclidean distance

### Feature Spaces

| Method | Feature space for Silhouette | Notes |
|--------|------------------------------|-------|
| BoW | Original filtered matrix (d=34) | Direct enrollment features |
| PCA | Original filtered matrix (d=34) | Evaluated in INPUT space, not PCA space |
| Spectral | Adjacency matrix | Graph structure features |
| DeepWalk | Embedding space (d=2) | Learned representation |
| Node2Vec | Embedding space (d=2) | Learned representation |

### PCA Silhouette Space

The PCA method computes Silhouette on the **original BoW features**, not the PCA-reduced space. This is correct — it evaluates whether the PCA-based clustering produces well-separated clusters in the original feature space. The PCA reduction is only used for clustering, not evaluation.

### Student Set

All metrics use the SAME student set (all students, including isolated ones). No student is accidentally excluded.

### Clustering Validation

- All methods use exactly k=2 clusters (matching the sectioning problem)
- `n_init=10` for KMeans (reduces initialization sensitivity)
- `compute_metrics()` checks for degenerate clusterings (n_unique < 2 → NaN)

**Severity: NONE** — Metrics are correctly implemented and applied.

---

## 11. Result Consistency

### Cross-check: all_methods.json vs REPORT.md

| Method | Report Avg | File Avg | Match |
|--------|-----------|----------|-------|
| BoW | 0.153 | 0.153 | ✓ |
| PCA | 0.158 | 0.158 | ✓ |
| Spectral | 0.127 | 0.127 | ✓ |
| DeepWalk | 0.613 | 0.613 | ✓ |
| Node2Vec | 0.611 | 0.611 | ✓ |

### Sensitivity Grid

- All 9 (p,q) configurations executed ✓
- Results saved in `sensitivity.json` ✓
- Best config correctly identified (p=1, q=0.5, Sil=0.622) ✓

### Stability Data

- 120 entries (20 seeds × 6 courses) ✓
- Results saved in `stability_partial.json` ✓
- Stability statistics computed and saved in `stability.json` ✓

**Severity: NONE** — All results are consistent across files.

---

## 12. Figure Audit

All 7 expected figures exist in `results/final_reexperiment/figures/`:

| Figure | Size | Status |
|--------|------|--------|
| graph_density_comparison.pdf | 17,447 bytes | ✓ |
| method_comparison_all_metrics.pdf | 27,679 bytes | ✓ |
| method_comparison_silhouette.pdf | 21,795 bytes | ✓ |
| node2vec_vs_deepwalk.pdf | 19,482 bytes | ✓ |
| sensitivity_heatmap.pdf | 28,720 bytes | ✓ |
| stability_boxplot.pdf | 19,208 bytes | ✓ |
| runtime_comparison.pdf | 17,828 bytes | ✓ |

All figures are PDF (vector format), suitable for publication. No figures use stale Windows results.

**Severity: NONE** — Figures are complete and current.

---

## 13. Problems Found

| # | Issue | Severity | Impact on Conclusions | Action Required |
|---|-------|----------|----------------------|-----------------|
| 1 | Holm correction bug in `stage_e.py` | **MEDIUM** | None (conservative bias) | **FIXED** — Stage E rerun with corrected values |
| 2 | Isolated students get degenerate embeddings | **LOW** | Negligible (<0.2% Sil change) | Mention in methodology |
| 3 | Same seed controls walks + W2V + KMeans | **LOW** | Stability measures total pipeline variance | Note in methods |
| 4 | n=6 limits statistical power | **LIMITATION** | Cannot claim p<0.05 after correction | Report effect sizes |

---

## 14. Required Corrections

### Must Fix Before Manuscript

1. **FIXED**: Holm correction in `experiments/stage_e.py` — replaced backward `min` sweep with `np.maximum.accumulate` (forward cumulative maximum). Stage E rerun, `statistical_analysis.json` updated.

2. ~~Rerun Stage E after fixing~~ **DONE**: Stage E rerun, corrected p-values are 0.1875 (previously 0.0625).

### Should Mention in Methodology

3. Note the existence of isolated students (3 total) and their negligible impact.
4. Note that the same seed controls all three stochastic components.

---

## 15. Experiments That Were Rerun

- **Stage E only** (after Holm correction fix): Statistical analysis + final figures ✓ DONE
- Time: 0.8 seconds

---

## 16. Experiments That Do NOT Need Rerunning

- Stage A (preprocessing + all methods): ✓ Results are correct
- Stage B (sensitivity): ✓ Results are correct
- Stage C/D (stability): ✓ Results are correct
- Graph analysis: ✓ Verified independently
- Node2Vec implementation: ✓ Verified empirically

---

## 17. Explicit Recommendation

### READY FOR MANUSCRIPT REVISION

The pipeline is scientifically defensible after one fix:

1. ~~Fix the Holm correction bug~~ **DONE**
2. ~~Rerun Stage E~~ **DONE** (0.8s)
3. Proceed with manuscript revision

The core findings are robust:
- Graph methods dominate baselines on ALL 6 courses (Cliff's δ = 1.000)
- Effect sizes are large (r = 0.899)
- Node2Vec and DeepWalk are equivalent (as theoretically expected)
- Preprocessing (constant course removal) is methodologically sound
- All 7 publication figures are ready

---

## 18. Files Examined

### Source Code
- `experiments/shared.py` — Shared functions (Node2Vec, DeepWalk, metrics)
- `experiments/config.py` — Configuration parameters
- `experiments/core.py` — Original pipeline functions
- `experiments/stage_a.py` — Preprocessing + all methods
- `experiments/stage_b.py` — Node2Vec sensitivity
- `experiments/stage_c.py` — Stability seeds 0-4
- `experiments/stage_d.py` — Stability seeds 5-9
- `experiments/stage_d2.py` — Stability seeds 10-14
- `experiments/stage_d3.py` — Stability seeds 15-19
- `experiments/stage_e.py` — Statistical analysis + figures

### Result Files
- `results/final_reexperiment/all_methods.json`
- `results/final_reexperiment/sensitivity.json`
- `results/final_reexperiment/stability_partial.json`
- `results/final_reexperiment/stability.json`
- `results/final_reexperiment/statistical_analysis.json`
- `results/final_reexperiment/graph_analysis.json`
- `results/final_reexperiment/preprocessing_info.json`
- `results/final_reexperiment/runtime.json`
- `results/final_reexperiment/environment.json`
- 7 PDF figures in `results/final_reexperiment/figures/`

### Data Files
- `data/1.txt` through `data/6.txt`

---

*Report generated by technical audit on August 28, 2026.*
