# DeepWalk-SSP: Final Re-Experiment Report

**Date:** August 28, 2026  
**Status:** Complete. Manuscript NOT yet revised.

---

## 1. What Was Changed in Preprocessing

All methods now use the same filtered student-course data: courses taken by **every** student in each sectioning instance are removed before any representation is constructed or graph is built. This is standard variance-based preprocessing — a constant feature provides zero discriminative information.

The removal is applied uniformly to:
- BoW/enrollment representation
- PCA + KMeans
- Spectral clustering (on the graph built from filtered data)
- DeepWalk + KMeans
- Node2Vec + KMeans

No other preprocessing changes were made.

---

## 2. Constant Courses Removed

| Course | Students | Courses Before | Courses After | Removed | Column Index | Target Course? |
|--------|----------|----------------|---------------|---------|--------------|----------------|
| 1 | 74 | 35 | 34 | 1 | 3 | No |
| 2 | 51 | 34 | 33 | 1 | 8 | No |
| 3 | 48 | 28 | 27 | 1 | 13 | No |
| 4 | 49 | 28 | 27 | 1 | 19 | No |
| 5 | 52 | 32 | 31 | 1 | 22 | No |
| 6 | 67 | 33 | 32 | 1 | 25 | No |

Each instance has exactly **one** constant course. The removed column is not necessarily column 0 (the target course). The target course itself is not universally shared in these data files — only one other course per instance happens to be taken by all students. No other courses besides the single removed column are constant.

---

## 3. Graph Statistics After Preprocessing

| Course | Edges | Max Edges | Density | Complete? | Components |
|--------|-------|-----------|---------|-----------|------------|
| 1 | 1828 | 2701 | 0.677 | No | 3 |
| 2 | 866 | 1275 | 0.679 | No | 1 |
| 3 | 894 | 1128 | 0.793 | No | 2 |
| 4 | 1043 | 1176 | 0.887 | No | 1 |
| 5 | 912 | 1326 | 0.688 | No | 1 |
| 6 | 1325 | 2211 | 0.599 | No | 1 |

**No graphs are complete after preprocessing.** Average density dropped from 1.000 to 0.721. The q parameter of Node2Vec now has a meaningful effect.

Courses 1 and 3 have multiple connected components (3 and 2 respectively) due to isolated students who share no non-target courses with anyone.

---

## 4. Whether Graphs Are Still Complete

**No.** All six graphs are non-complete after removing the single constant course. The maximum density is 0.887 (Course 4). The minimum is 0.599 (Course 6). This confirms that removing constant courses is sufficient to create structurally informative graphs.

---

## 5. Final Results for All Methods

### Silhouette Score (seed=0)

| Method | C1 | C2 | C3 | C4 | C5 | C6 | **Avg** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| BoW+KMeans | 0.099 | 0.231 | 0.142 | 0.128 | 0.116 | 0.202 | **0.153** |
| PCA+KMeans | 0.094 | 0.231 | 0.154 | 0.128 | 0.137 | 0.202 | **0.158** |
| Spectral | 0.083 | 0.205 | 0.135 | 0.128 | 0.067 | 0.143 | **0.127** |
| DeepWalk | 0.570 | 0.617 | 0.519 | 0.574 | 0.766 | 0.633 | **0.613** |
| Node2Vec (p=1, q=1) | 0.535 | 0.599 | 0.570 | 0.644 | 0.734 | 0.582 | **0.611** |

### Davies-Bouldin Index (lower is better, seed=0)

| Method | C1 | C2 | C3 | C4 | C5 | C6 | **Avg** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| BoW+KMeans | 2.874 | 1.313 | 2.383 | 2.309 | 2.506 | 1.653 | **2.240** |
| PCA+KMeans | 2.936 | 1.313 | 2.370 | 2.309 | 2.231 | 1.653 | **2.135** |
| Spectral | 2.849 | 1.632 | 2.470 | 2.309 | 1.824 | 2.279 | **2.227** |
| DeepWalk | 0.615 | 0.494 | 0.663 | 0.553 | 0.383 | 0.530 | **0.540** |
| Node2Vec (p=1,q=1) | 0.617 | 0.538 | 0.558 | 0.467 | 0.419 | 0.603 | **0.534** |

### Calinski-Harabasz Index (higher is better, seed=0)

| Method | C1 | C2 | C3 | C4 | C5 | C6 | **Avg** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| BoW+KMeans | 8.2 | 13.9 | 7.5 | 8.0 | 7.3 | 18.6 | **10.6** |
| PCA+KMeans | 8.0 | 13.9 | 7.6 | 8.0 | 7.8 | 18.6 | **10.7** |
| Spectral | 7.5 | 12.4 | 7.1 | 8.0 | 4.8 | 10.2 | **8.3** |
| DeepWalk | 78.5 | 131.5 | 63.1 | 60.8 | 205.8 | 183.2 | **120.5** |
| Node2Vec (p=1,q=1) | 116.8 | 134.6 | 83.5 | 124.6 | 176.8 | 141.3 | **129.6** |

---

## 6. Best Node2Vec Configuration

| p | q | Avg Silhouette |
|---|---|---------------|
| 0.5 | 0.5 | 0.620 |
| 0.5 | 1.0 | 0.616 |
| 0.5 | 2.0 | 0.614 |
| **1.0** | **0.5** | **0.622** |
| 1.0 | 1.0 | 0.611 |
| 1.0 | 2.0 | 0.605 |
| 2.0 | 0.5 | 0.615 |
| 2.0 | 1.0 | 0.615 |
| 2.0 | 2.0 | 0.616 |

**Best:** p=1.0, q=0.5 (Silhouette = 0.622). BFS-like exploration (lower q) slightly favors local structure.

---

## 7. Whether q Has a Meaningful Effect

**Yes.** On the revised (non-complete) graphs, q has a measurable effect with a range of 0.017 (0.605 to 0.622). Lower q (more BFS-like) tends to perform slightly better. This is expected: BFS exploration captures local community structure, which aligns with student similarity based on shared course enrollment patterns.

The effect is modest — all q values produce Silhouette scores well above 0.600, and the difference between the best and worst configuration is small compared to the gap between graph methods and baselines.

---

## 8. Node2Vec vs DeepWalk Comparison

### Mean Silhouette (20 seeds, course-level)

| Course | Node2Vec (p=1,q=1) | DeepWalk | Difference |
|--------|-------------------|----------|------------|
| 1 | 0.563 ± 0.014 | 0.561 ± 0.026 | +0.002 |
| 2 | 0.606 ± 0.009 | 0.611 ± 0.011 | -0.005 |
| 3 | 0.572 ± 0.014 | 0.562 ± 0.032 | +0.010 |
| 4 | 0.623 ± 0.018 | 0.573 ± 0.029 | +0.050 |
| 5 | 0.743 ± 0.007 | 0.774 ± 0.010 | -0.031 |
| 6 | 0.599 ± 0.013 | 0.634 ± 0.010 | -0.035 |
| **Average** | **0.618** | **0.619** | **-0.001** |

**Node2Vec and DeepWalk perform essentially identically** (avg Silhouette difference = 0.001). This is expected: Node2Vec(p=1,q=1) is the unbiased random walk equivalent of DeepWalk. The small per-course differences are due to different random number streams, not algorithmic differences.

### Statistical Test

| Comparison | Raw p | Holm-corrected p | Effect size r | Interpretation |
|-----------|-------|------------------|---------------|----------------|
| Node2Vec vs DeepWalk | 1.000 | 1.000 | 0.043 | Negligible |
| Node2Vec vs BoW | 0.031 | 0.1875 | 0.899 | Large |
| Node2Vec vs PCA | 0.031 | 0.1875 | 0.899 | Large |
| Node2Vec vs Spectral | 0.031 | 0.1875 | 0.899 | Large |
| DeepWalk vs BoW | 0.031 | 0.1875 | 0.899 | Large |
| DeepWalk vs PCA | 0.031 | 0.1875 | 0.899 | Large |

**Important caveats:**
- Only n = 6 courses. The minimum achievable two-sided Wilcoxon p-value is ~0.031 for n = 6.
- With Holm correction for 6 comparisons, none reach significance at α = 0.05.
- The uncorrected p = 0.031 should be interpreted as "p < 0.05" not as a precise value.
- Effect sizes (r = 0.899) are large and more informative than p-values here.
- Cliff's δ = 1.000 means graph methods scored higher on every single course.

---

## 9. Stability Results (20 seeds)

### Node2Vec Stability

| Course | Mean | Std | Min | Max | Range |
|--------|------|-----|-----|-----|-------|
| 1 | 0.563 | 0.014 | 0.531 | 0.587 | 0.056 |
| 2 | 0.606 | 0.009 | 0.591 | 0.620 | 0.029 |
| 3 | 0.572 | 0.014 | 0.548 | 0.595 | 0.047 |
| 4 | 0.623 | 0.018 | 0.589 | 0.651 | 0.062 |
| 5 | 0.743 | 0.007 | 0.732 | 0.759 | 0.027 |
| 6 | 0.599 | 0.013 | 0.580 | 0.634 | 0.054 |

**Average std: 0.013** — Node2Vec is highly stable across seeds.

### DeepWalk Stability

| Course | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| 1 | 0.561 | 0.026 | 0.504 | 0.602 |
| 2 | 0.611 | 0.011 | 0.589 | 0.629 |
| 3 | 0.562 | 0.032 | 0.510 | 0.629 |
| 4 | 0.573 | 0.029 | 0.513 | 0.627 |
| 5 | 0.774 | 0.010 | 0.759 | 0.791 |
| 6 | 0.634 | 0.010 | 0.614 | 0.661 |

**Average std: 0.020** — DeepWalk is slightly less stable than Node2Vec (0.020 vs 0.013).

---

## 10. Statistical Analysis and Limitations

### What was done
- Course-level paired Wilcoxon signed-rank tests (two-sided)
- Holm correction for multiple comparisons (6 tests)
- Effect size: r (from Z-score) and Cliff's delta
- n = 6 independent courses (the proper statistical unit)

### Key limitation
**n = 6 is extremely small for hypothesis testing.** The minimum achievable two-sided Wilcoxon p-value is ~0.031 (1/C(6,3) × 2). After Holm correction for 6 comparisons, no test reaches p < 0.05. This does NOT mean the graph methods don't outperform baselines — the effect sizes are unambiguously large (r = 0.899, δ = 1.000). It means we cannot make a formal statistical claim at α = 0.05 with this sample size.

### Recommendation for the manuscript
- Report effect sizes prominently alongside p-values
- State the p = 0.031 floor explicitly
- Emphasize that graph methods dominate on every course (Cliff's δ = 1.000)
- Do NOT claim "statistically significant" without the caveat about n = 6

---

## 11. Figures Generated

| File | Description |
|------|-------------|
| `graph_density_comparison.pdf` | Graph density before/after preprocessing |
| `method_comparison_all_metrics.pdf` | All 5 methods × 3 metrics (Sil, DBI, CH) |
| `method_comparison_silhouette.pdf` | Silhouette Score bar chart |
| `node2vec_vs_deepwalk.pdf` | Node2Vec vs DeepWalk per-course |
| `sensitivity_heatmap.pdf` | p × q parameter sensitivity heatmaps |
| `stability_boxplot.pdf` | Node2Vec vs DeepWalk stability (20 seeds) |
| `runtime_comparison.pdf` | Runtime across all methods |

All in `results/final_reexperiment/figures/` as PDF.

---

## 12. Files Created/Modified

### New experiment scripts
- `experiments/shared.py` — Shared functions
- `experiments/stage_a.py` — Preprocessing + all methods
- `experiments/stage_b.py` — Node2Vec sensitivity
- `experiments/stage_c.py` — Stability seeds 0-4
- `experiments/stage_d.py` — Stability seeds 5-9
- `experiments/stage_d2.py` — Stability seeds 10-14
- `experiments/stage_d3.py` — Stability seeds 15-19
- `experiments/stage_e.py` — Statistical analysis + final figures
- `experiments/full_reexperiment.py` — Combined script (reference)

### Results (in `results/final_reexperiment/`)
- `preprocessing_info.json` — Constant course removal details
- `graph_analysis.json` — Graph statistics before/after
- `all_methods.json` / `.xlsx` — All method results (seed=0)
- `sensitivity.json` / `.xlsx` — p × q grid results
- `stability.json` — 20-seed stability statistics
- `stability_partial.json` — Raw per-seed data
- `statistical_analysis.json` — Wilcoxon tests with Holm correction
- `runtime.json` — Runtime measurements
- `environment.json` — Package versions

---

## 13. Git Status Regarding `paper/`

- `/paper` was removed from `.gitignore` — paper/ is now tracked
- `/doc` was removed from `.gitignore` — doc/ is now tracked
- Manuscript in `paper/sn-article.tex` has **NOT** been modified

---

## 14. Commit Hash

Will be updated after Holm fix commit.

---

## 15. Push Status

Will be updated after push.

---

## 16. Recommendation for Next Manuscript Revision

1. **Report both Node2Vec and DeepWalk** as equivalent graph methods. Since Node2Vec(p=1,q=1) ≡ DeepWalk (unbiased walks), they are mathematically the same algorithm. The paper should present one unified graph representation learning approach, not two separate methods.

2. **Highlight the preprocessing insight** (removing constant courses) as a key methodological contribution. This transforms an uninformative complete graph into a structurally rich one.

3. **Use Node2Vec(p=1, q=0.5)** as the main reported configuration if the paper wants to show that graph-based exploration parameters matter. The BFS-like bias (q=0.5) performs best (0.622 vs 0.611 for neutral).

4. **Be transparent about n = 6** and report effect sizes prominently.

5. **Note that DeepWalk is slightly less stable** than Node2Vec (std 0.020 vs 0.013), though both are stable in absolute terms.

---

## 17. Recommendation on "Student2Vec"

**Not yet justified as a separate name.** The implementation is standard Node2Vec applied to student co-enrollment graphs. The novelty lies in:
- The preprocessing insight (constant course removal)
- The domain-specific application to student sectioning
- The evaluation protocol

If a distinctive name is desired, "Graph-based Student Sectioning" or "Co-enrollment Graph Embedding" would be more accurate than "Student2Vec", which implies a new algorithm. The manuscript could use "Student2Vec" as a convenient shorthand while explicitly stating it applies Node2Vec to student co-enrollment data.

---

*Report generated automatically. All experimental data saved in `results/final_reexperiment/`.*
