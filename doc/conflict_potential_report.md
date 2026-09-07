# Inter-section Course Conflict Potential — Investigation Report

Date: 2026-09-08
Script: `scripts/conflict_potential.py` → `results/conflict_potential/`
Manuscript: **not modified** (per instruction; see §I for a proposal if ever needed).

---

## A. Data / artifact audit

| Item | Finding |
|---|---|
| Enrollment data | `data/{1..6}.txt`: line 1 = `n m`, then `idx student_id binary_vector(m)`. Course 1: 74×35 … course 6: 67×33. |
| Target course | Each instance contains **exactly one universal (all-1) column** — the target course itself (col. 3, 8, 13, 19, 22, 25). After the pipeline's constant-feature removal (`find_univ` + `np.delete`), the remaining columns are the non-target courses, and `(Mf Mfᵀ)_ij` = number of shared **non-target** courses. Cross-check: the fraction of pairs with ≥1 shared non-target course (0.677, 0.679, 0.793, 0.887, 0.688, 0.599) **exactly equals the graph densities in `tab:graph_stats`**. |
| Course universe | Each file's columns = courses taken by ≥1 student of the instance (within the 38-course dataset). A student's courses outside this universe are unobservable. |
| Node2Vec labels | Archived for **all 45 grid configs, seed 42** (`results/grid_search/labels_*.json`), incl. primary (1, 0.5, 2), quality-optimal (2, 1, 1), neutral (1, 1, 2). `final_d2/labels_course*_n105.json` is byte-identical to the grid file for the primary config. |
| Baseline labels | **Not archived anywhere** (only metrics). Recomputed here with the exact archived protocol (seed 42; `KMeans(n_init=10)`; `PCA(n_components=2)`; spectral = `SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42, assign_labels='kmeans')` on the weighted adjacency, per `experiments/run_all.py:spec`). **Validation: all 18 recomputed baseline Silhouettes match `results/reproduced/all_methods.json` exactly (≤1e-9)** — the recomputed labels are the ones behind the paper's tables. They are now saved (`results/conflict_potential/labels_{bow,pca,spec}_course*.json`) so future analyses never rerun anything. |
| Primary config | Confirmed still **p=1.0, q=0.5, d=2** (Pareto-selected); its label balances reproduce `tab:section_balance` exactly (0.609 / 0.594 / 0.714 / 0.815 / 0.368 / 0.811). |
| Reruns needed | **No Node2Vec reruns.** Only deterministic KMeans/PCA/spectral recomputation for the three baselines (<1 s CPU total). |

## B. Exact metric definition

For instance (target course c\*), with `Mf` the pipeline's constant-feature-filtered binary matrix:

- `s_ij = Σ_k Mf_ik·Mf_jk` = number of shared non-target courses (= co-enrollment edge weight).
- **A (primary):** `C_A` = pair-weighted mean of `s_ij` over **within-section** pairs (both sections pooled). Units: courses; interpretable directly.
- **B (secondary):** `C_B` = pair-weighted fraction of within-section pairs with `s_ij ≥ 1`.
- **Reference:** the same statistics over **all** n(n−1)/2 pairs = expectation under a random split (exchangeability). `gain_A = C_A / E_random[C_A]` is size-independent and shows whether sectioning beats a random split.
- Per-section values, section sizes, and pair counts are stored per course/method in the JSON.

Chosen aggregate: pair-weighted pooling (not the unweighted mean of the two sections), so a 2-student section contributes 1 pair and cannot distort the average. No singleton sections occur (min size 2).

## C. Methodological validity and leakage/circularity

1. **Same information source.** The metric is computed from the *same enrollment matrix* from which the co-enrollment graph — and hence the Node2Vec input — is built. It is **not an independent downstream-task evaluation**; it is a different *pooling* of the same information (raw pairwise overlap vs. learned-space geometry).
2. **Asymmetric near-tautology for the baselines.** For BoW, `s_ij = x_i·x_j` and KMeans distance satisfies ‖x_i−x_j‖² = ‖x_i‖²+‖x_j‖²−2s_ij: clustering raw vectors ≈ grouping high-overlap pairs, so the metric is nearly BoW's own objective. Spectral clusters rows of the adjacency (each row *is* a student's overlap profile) — likewise near-tautological. PCA preserves dot products. The baselines therefore *optimize approximately what the metric measures*; the comparison is structurally biased toward them.
3. **Direction is undetermined — this is decisive.** "More within-section overlap" can be read two ways, and plausible mechanisms point in **opposite** directions:
   - *Conflict reading (lower = better):* section-mates sharing many other courses are more entangled in downstream scheduling.
   - *Coherence / feasibility reading (higher = better):* a section whose members share other courses has a **smaller union** of "other courses" to avoid when its time slot is chosen, making the section easier to timetable; shared courses also mean a natural cohort travelling together.
   With no timetable data, the paper cannot justify either direction. The safe name is **"within-section shared-course overlap"** — *not* "conflict" (the actual number of timetable conflicts cannot be measured without a timetable).
4. **Insensitivity to degenerate splits.** Pair-weighting makes the metric nearly blind to the 72/2-type splits that the paper's Pareto analysis is about (quality-optimal d=1 scores ≈ the primary config on this metric). It cannot replace or augment the balance objective.

## D. Sufficiency of existing artifacts

Sufficient. Computed entirely from saved labels + a deterministic recomputation of the three baselines (validated exactly). No long experiments; nothing outside `results/conflict_potential/` was written.

## E. Python script

`scripts/conflict_potential.py` — loads data with the pipeline's own `read_class`/`find_univ` logic, recomputes+saves baseline labels, loads archived Node2Vec labels (with student-ID order assertions), computes A/B/reference/gain per course and method, validates Silhouettes against the archive, and writes `conflict_potential.json`.

## F. Bash script

Not needed (nothing had to be re-executed on your Linux machine).

## G. Numerical results

### C_A — mean shared non-target courses among within-section pairs

| Course | BoW+KMeans | PCA+KMeans | Spectral | N2V primary (1,0.5,2) | N2V quality (2,1,1) | N2V neutral (1,1,2) | Random-split ref. |
|---|---|---|---|---|---|---|---|
| 1 | 1.563 | 1.582 | 1.765 | 1.351 | 1.352 | 1.322 | 1.334 |
| 2 | 1.854 | 1.854 | 1.999 | 1.510 | 1.519 | 1.505 | 1.461 |
| 3 | 1.696 | 1.696 | 1.830 | 1.481 | 1.510 | 1.556 | 1.502 |
| 4 | 2.589 | 2.589 | 2.589 | 2.081 | 2.194 | 2.180 | 2.064 |
| 5 | 2.012 | 2.009 | 1.856 | 1.436 | 1.350 | 1.398 | 1.437 |
| 6 | 1.967 | 1.967 | 1.941 | 1.371 | 1.416 | 1.376 | 1.340 |
| **Mean** | **1.947** | **1.949** | **1.997** | **1.538** | **1.557** | **1.556** | — |
| Median | 1.910 | 1.910 | 1.899 | 1.458 | 1.463 | 1.452 | — |
| Gain over random (mean) | 1.282 | 1.284 | 1.317 | **1.010** | 1.020 | 1.019 | 1.000 |

### C_B — fraction of within-section pairs sharing ≥1 non-target course (mean over courses)

BoW 0.818 · PCA 0.820 · Spectral 0.867 · N2V primary 0.723 · quality 0.725 · neutral 0.724 (random reference: 0.60–0.89 per course).

### Win counts of the primary N2V config on C_A (out of 6 courses)

- vs BoW: **0/6** (mean −21.0%) · vs PCA: **0/6** (−21.1%) · vs Spectral: **0/6** (−23.0%) — *if higher overlap is desirable*.
- Mirrored (lower-is-better "conflict" reading): N2V primary would win **6/6, 6/6, 5/6**.

### Side observations

- PCA+KMeans reproduces BoW+KMeans partitions almost exactly (identical C_A/C_B in courses 2, 3, 4, 6) — consistent with PCA being a linear transform of BoW.
- Spectral attains the highest overlap (mean gain 1.317) — expected, since it clusters the overlap profiles directly.
- **All three Node2Vec configs sit at ≈ random overlap (gain 0.99–1.06)**: the 2D embedding's KMeans partition does not preserve raw pairwise overlap structure; its Silhouette advantage is geometric, not overlap-based.

## H. Recommendation: **do NOT add this metric to the manuscript**

1. **Direction ambiguity** (§C.3): the same number supports opposite conclusions; choosing a direction would be an untestable assumption, and choosing the one that favors Node2Vec would violate the objectivity this investigation was required to maintain.
2. **Near-tautology for baselines** (§C.2): the comparison does not evaluate sectioning quality; it largely re-measures which method clusters in raw overlap space — already conveyed conceptually by the paper's representation comparison.
3. **No added value over existing metrics** (§C.4): it is insensitive to the degenerate splits that the quality–balance analysis targets, adds nothing to Silhouette/DBI/CH + balance, and (being a transform of the same enrollment data) does not address the Limitations sentence about timetable feasibility.
4. Not suitable as a third Pareto objective either (worse: it would couple an ambiguous-direction quantity into configuration selection).

Kept for the record: script + full JSON + newly archived baseline labels in `results/conflict_potential/`. If actual timetabling data or a formal constraint model becomes available, a genuine downstream evaluation could be built on this foundation.

## I. Minimal manuscript change (proposal only — NOT applied)

If the authors ever decide otherwise, the smallest defensible integration would be:

- **Location:** one short paragraph at the end of *Section Balance Analysis* (or in *Limitations*), explicitly framed as *within-section shared-course overlap*, reported descriptively (not as a ranking), with the direction ambiguity stated, plus one sentence in *Limitations* noting that overlap is a proxy whose desirability depends on institutional scheduling policy. No new table, no figure, no change to the Pareto analysis.
- Would require a `% SOURCE:` provenance comment (`scripts/conflict_potential.py` → `results/conflict_potential/conflict_potential.json`), consistent with the manuscript's convention.
