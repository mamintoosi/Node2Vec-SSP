# Final Consistency Audit Report

**Date:** September 1, 2026  
**Status:** PASS WITH ISSUES  
**Auditor:** Buffy (Codebuff agent)

---

## A. Audit Status

**PASS WITH ISSUES**

All numerical values in the manuscript's main comparison table are consistent with the source-of-truth data files. The Wilcoxon tests, Holm correction, and effect sizes are verified. Four figures have been regenerated. However, several issues were identified:

1. **Manuscript error:** The abstract and Section 4.6 claim "Cliff's δ = 1.0" for Node2Vec vs PCA, but the actual value is δ = 0.833 (PCA scores higher on Course 2).
2. **Statistical analysis inconsistency:** The `statistical_analysis.json` compares Node2Vec (embedding space) against PCA (BoW space), while the manuscript Table 1 uses PCA in embedding space.
3. **Pseudocode mismatch:** Algorithm 1 describes a simple random walk (DeepWalk), not the Node2Vec biased walk described in the text.
4. **Citation issue:** The Node2Vec citation key `KAZEMI2020101794` references Kazemi & Abhari (2020), not the original Grover & Leskovec (KDD 2016) paper.
5. **No DeepWalk-SSP in active text** — confirmed absent. ✓

---

## B. Numerical Consistency

### 1. PCA Source of Truth (CRITICAL CHECK)

The `all_methods.json` contains **two** silhouette evaluations for PCA:

| Metric | Per-course values | Average |
|--------|------------------:|--------:|
| `silhouette_emb` (PCA 2-dim space) | 0.357, 0.603, 0.425, 0.428, 0.425, 0.524 | **0.460** |
| `silhouette_bow` (original 34-dim BoW space) | 0.094, 0.231, 0.154, 0.128, 0.137, 0.202 | **0.158** |

- **Manuscript value:** 0.460 (uses `silhouette_emb`)
- **Source-of-truth (`silhouette_emb`):** 0.4604
- **Source-of-truth (`silhouette_bow`):** 0.1579
- **Match:** YES — the manuscript consistently uses `silhouette_emb` (PCA-space evaluation)
- **File:** `results/final_reexperiment/all_methods.json`, `silhouette_emb` field
- **Code:** `experiments/stage_a.py`, `compute_metrics(pr, pl)` line ~128

**Note:** The REPORT.md (Section 5) reports PCA avg = 0.158 (BoW-space evaluation). The manuscript uses 0.460 (PCA-space evaluation). Both are valid but evaluate different things. The statistical analysis in `stage_e.py` uses the BoW-space PCA scores, creating an inconsistency with Table 1.

### 2. Table 1 Audit

| Method   | Manuscript | Source of Truth | Match |
| -------- | --------: | --------------: | ----- |
| BoW      |     0.153  |          0.1532 |  YES  |
| PCA      |     0.460  |          0.4604 |  YES  |
| Spectral |     0.213  |          0.2127 |  YES  |
| DeepWalk |     0.613  |          0.6132 |  YES  |
| Node2Vec |     0.611  |          0.6106 |  YES  |

**Per-course verification:** All 30 values (5 methods × 6 courses) match within rounding tolerance (±0.002).

**Source file:** `results/final_reexperiment/all_methods.json`, `silhouette_emb` column.

### 3. Wilcoxon Tests

Using 20-seed means for Node2Vec/DeepWalk and seed=0 `silhouette_emb` for baselines, with n=6 courses:

| Comparison | W | Raw p | n | Input vectors |
| ---------- | -: | -----: | -: | ------------- |
| Node2Vec vs DeepWalk | 10.0 | 1.000000 | 6 | n2v_means vs dw_means |
| Node2Vec vs BoW | 0.0 | 0.031250 | 6 | n2v_means vs bow_emb |
| Node2Vec vs PCA | 0.0 | 0.031250 | 6 | n2v_means vs pca_emb |
| Node2Vec vs Spectral | 0.0 | 0.031250 | 6 | n2v_means vs spec_emb |
| DeepWalk vs BoW | 0.0 | 0.031250 | 6 | dw_means vs bow_emb |
| DeepWalk vs PCA | 0.0 | 0.031250 | 6 | dw_means vs pca_emb |

**Comparison with `statistical_analysis.json`:** All p-values match ✓

**Note:** The minimum achievable two-sided Wilcoxon p-value for n=6 is 2/2⁶ = 0.03125.

### 4. Effect Size r

| Comparison | Recomputed r | JSON r | Match |
| ---------- | -----------: | -----: | ----- |
| Node2Vec vs DeepWalk | 0.043 | 0.043 | YES |
| Node2Vec vs BoW | 0.899 | 0.899 | YES |
| Node2Vec vs PCA | 0.899 | 0.899 | YES |
| Node2Vec vs Spectral | 0.899 | 0.899 | YES |
| DeepWalk vs BoW | 0.899 | 0.899 | YES |
| DeepWalk vs PCA | 0.899 | 0.899 | YES |

**Formula:** r = |z| / √n, where z = (W - μ_W) / σ_W

### 5. Cliff's Delta

| Comparison | Recomputed δ | JSON δ | Manuscript claim | Match JSON? |
| ---------- | ----------: | -----: | ---------------: | :---------: |
| Node2Vec vs DeepWalk | 0.056 | 0.056 | — | YES |
| Node2Vec vs BoW | 1.000 | 1.000 | 1.0 | YES |
| Node2Vec vs PCA | **0.833** | **1.0** | 1.0 | **NO** |
| Node2Vec vs Spectral | 1.000 | 1.000 | — | YES |
| DeepWalk vs BoW | 1.000 | 1.000 | — | YES |
| DeepWalk vs PCA | **0.833** | **1.0** | — | **NO** |

**DISCREPANCY:** The JSON reports δ = 1.0 for Node2Vec vs PCA, but recomputation yields δ = 0.833.

**Explanation:** The JSON's statistical analysis uses PCA scores in **BoW space** (`silhouette_score(m.astype(float), pl)` in `stage_e.py`), where PCA scores are much lower (avg 0.158). In that comparison, every Node2Vec course score exceeds every PCA course score, giving δ = 1.0. However, the manuscript Table 1 uses PCA scores in **PCA space** (`silhouette_emb`), where PCA C2 = 0.603 > Node2Vec C2 mean = 0.606 (at seed=0: PCA C2 = 0.603 > Node2Vec C2 = 0.599), giving δ = 0.833.

**Impact on manuscript claims:**
- Abstract: "Node2Vec outperforms PCA and the conventional representation on all six courses, with large effect sizes (Cliff's δ = 1.0)." → **PARTIALLY INCORRECT**: δ = 0.833 for Node2Vec vs PCA
- Section 4.6: Same claim → **PARTIALLY INCORRECT**
- The claim "Node2Vec outperforms PCA on all six courses" is only true for the BoW-space PCA evaluation, not the PCA-space evaluation reported in Table 1

### 6. Holm Correction

| Comparison | Raw p | Holm p (recomputed) | JSON p | Manuscript | All agree? |
| ---------- | ----: | -------------------: | -----: | ----------: | :--------: |
| Node2Vec vs DeepWalk | 1.0000 | 1.0000 | 1.0000 | 1.0000 | YES |
| Node2Vec vs BoW | 0.0313 | 0.1875 | 0.1875 | 0.188 | YES |
| Node2Vec vs PCA | 0.0313 | 0.1875 | 0.1875 | 0.188 | YES |
| Node2Vec vs Spectral | 0.0313 | 0.1875 | 0.1875 | — | YES |
| DeepWalk vs BoW | 0.0313 | 0.1875 | 0.1875 | — | YES |
| DeepWalk vs PCA | 0.0313 | 0.1875 | 0.1875 | — | YES |

**Implementation verified:** The `holm_correction()` in `experiments/stage_e.py` correctly implements the Holm step-down procedure:
1. Sort p-values ascending
2. Multiply i-th ordered p by (m - i) [0-based]
3. Enforce monotonicity via cumulative maximum
4. Clamp to 1.0

---

## C. Hyperparameter Consistency

| Parameter | Value | Manuscript | Match |
| --------- | ----- | ---------- | ----- |
| d (embedding dim) | 2 | d=2 | YES |
| t (walk length) | 10 | t=10 | YES |
| γ (num walks) | 80 | γ=80 | YES |
| w (context window) | 5 | w=5 | YES |
| ε (Word2Vec epochs) | 30 | ε=30 | YES |
| k (clusters) | 2 | k=2 | YES |
| n_init (KMeans) | 10 | n_init=10 | YES |
| p (return) | 1.0 | p=1 | YES |
| q (in-out) | 1.0 | q=1 | YES |

**Default configuration:** Node2Vec (p=1, q=1) = unbiased random walk  
**Best sensitivity configuration:** Node2Vec (p=1, q=0.5) → avg Silhouette 0.622

The manuscript correctly distinguishes between the default (p=1, q=1) and best (p=1, q=0.5) configurations. No accidental mixing detected.

---

## D. Pseudocode Consistency

**ISSUE FOUND: Algorithm 1 describes DeepWalk, not Node2Vec**

The manuscript's Algorithm 1 (`GenerateRandomWalks`, lines ~340-355) describes a **simple random walk**:

```
For each graph node as start:
    For k = 1 to γ:
        walk = [start]
        For j = 1 to t:
            start = randomly chosen node from list of neighbors of start
```

This is equivalent to DeepWalk's unbiased random walk. It does NOT implement the Node2Vec biased walk described in the surrounding text (Section 3.3), which includes:

- **Previous node tracking** (`t` in the transition probability)
- **Bias factor α_{pq}(t,x)** with three cases based on distance to previous node
- **Parameters p and q** controlling return and in-out bias

**The actual Node2Vec implementation** in `experiments/shared.py` (`generate_node2vec_walks()`) correctly implements all of these features.

**Mismatch:** The pseudocode shows a simple random walk (DeepWalk), while the text and implementation describe Node2Vec.

**Recommendation:** Algorithm 1 should be updated to reflect the Node2Vec biased walk with α_{pq} transition probabilities. However, this was NOT modified per the task scope.

---

## E. DeepWalk Residuals

**No `DeepWalk-SSP` occurrences found in active (non-commented) text of `sn-article.tex`.** ✓

Active references to `DeepWalk` in `sn-article.tex` (20 occurrences):

| Line | Classification | Reason |
| ---- | ------------- | ------ |
| 180 | ACCEPTABLE | Describes Node2Vec extending DeepWalk |
| 188 | ACCEPTABLE | Describes Node2Vec extending DeepWalk |
| 227 | ACCEPTABLE | DeepWalk citation for background |
| 235 | ACCEPTABLE | Positioning relative to existing work |
| 237 | ACCEPTABLE | Describes PCA baseline dimensionality |
| 313 | ACCEPTABLE | Explains p, q parameters; "recovers DeepWalk" |
| 340 | ACCEPTABLE | CBOW/Skip-gram figure caption mentions DeepWalk |
| 539 | ACCEPTABLE | Compares methods including DeepWalk |
| 545 | ACCEPTABLE | Table 1 caption lists DeepWalk as a method |
| 549 | ACCEPTABLE | Table 1 header includes DeepWalk column |
| 566 | ACCEPTABLE | Figure caption mentions DeepWalk |
| 570 | ACCEPTABLE | Discusses DeepWalk results |
| 574 | ACCEPTABLE | Discusses relative improvement |
| 578 | ACCEPTABLE | Explains theoretical equivalence |
| 707 | ACCEPTABLE | Sensitivity discussion mentions DeepWalk |
| 783 | ACCEPTABLE | Statistical comparison section header |
| 784 | ACCEPTABLE | Describes Node2Vec vs DeepWalk equivalence |
| 806 | ACCEPTABLE | t-SNE figure filename (generated by old script) |
| 883 | ACCEPTABLE | Explains theoretical equivalence |
| 891 | ACCEPTABLE | Stability discussion |

All remaining DeepWalk references are appropriate: they either explain the theoretical equivalence between Node2Vec(p=1,q=1) and DeepWalk, or refer to DeepWalk as one of the compared methods.

---

## F. Citation Audit

### DeepWalk Citation
- **Citation key:** `perozzi2014deepwalk`
- **Bibliography entry:** Perozzi, B., Al-Rfou, R., Skiena, S. (2014). DeepWalk: Online learning of social representations. *Proceedings of the 20th ACM SIGKDD*, pp. 701–710.
- **Status:** ✓ Correct and appropriately cited

### Node2Vec Citation
- **Citation key used in manuscript:** `KAZEMI2020101794`
- **Bibliography entry:** Kazemi, B., Abhari, A. (2020). Content-based Node2Vec for representation of papers in the scientific literature. *Data & Knowledge Engineering*, 127, 101794.
- **Status:** ⚠️ **This is NOT the original Node2Vec paper**

The original Node2Vec paper is:
> Grover, A., Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. *Proceedings of the 22nd ACM SIGKDD*, pp. 855–864.

This paper is **not present** in the bibliography (`paper/sn-bibliography.bib`). The manuscript cites a 2020 paper that uses Node2Vec, rather than the original method paper. This should be corrected for academic accuracy.

---

## G. Figure Regeneration

### Status: ALL FOUR FIGURES REGENERATED SUCCESSFULLY ✓

| Figure | PDF | PNG | Location | Labels |
| ------ | --- | --- | -------- | ------ |
| repro_silhouette_vs_d | ✓ | ✓ | paper/ + results/final_reexperiment/figures/ | Node2Vec (p=1, q=1), BoW + KMeans |
| silhouette_score_comparison_all_files | ✓ | ✓ | paper/ + results/final_reexperiment/figures/ | BoW + KMeans, PCA + KMeans, Spectral, DeepWalk, Node2Vec (p=1, q=1) |
| CHI_comparison_all_files | ✓ | ✓ | paper/ + results/final_reexperiment/figures/ | Same as above |
| DBI_comparison_all_files | ✓ | ✓ | paper/ + results/final_reexperiment/figures/ | Same as above |

**Data source:** `results/final_reexperiment/all_methods.json` (silhouette_emb, dbi_emb, ch_emb columns)

**Labels used:**
- `Node2Vec (p=1, q=1)` — for the default Node2Vec configuration ✓
- `DeepWalk` — for the DeepWalk comparison ✓
- `BoW + KMeans`, `PCA + KMeans`, `Spectral` — for baselines ✓

**`DeepWalk-SSP` ABSENT from all regenerated figures** ✓

**Verification:** Programmatic assertion confirmed no `DeepWalk-SSP` string in any legend or label.

**Generation script:** `experiments/final_audit_and_figures.py`

---

## H. Manuscript Integrity

**`paper/sn-article.tex` was NOT modified.** ✓

Verified via `git diff paper/sn-article.tex` — output is empty.

---

## Summary of Issues Found

| # | Issue | Severity | Impact | Action Needed |
| --- | ----- | -------- | ------ | ------------- |
| 1 | Cliff's δ for Node2Vec vs PCA is 0.833, not 1.0 | MEDIUM | Manuscript abstract and Section 4.6 make incorrect claim | Manuscript revision needed |
| 2 | Statistical analysis uses PCA in BoW space, Table 1 uses PCA in embedding space | MEDIUM | Inconsistent comparison spaces | Clarify in manuscript or harmonize |
| 3 | Algorithm 1 shows DeepWalk walk, not Node2Vec biased walk | LOW | Pseudocode-text mismatch | Update Algorithm 1 |
| 4 | Node2Vec citation references Kazemi (2020), not Grover & Leskovec (2016) | LOW | Missing original method paper | Add Grover & Leskovec citation |
| 5 | `exp_F_tSNE_DeepWalk_course5.png` filename in manuscript | LOW | Visual artifact | Rename file or update reference |

---

## Files Changed/Created

| File | Status |
|------|--------|
| `doc/FINAL_CONSISTENCY_AUDIT_REPORT.md` | Created |
| `experiments/final_audit_and_figures.py` | Created |
| `paper/repro_silhouette_vs_d.png` | Regenerated |
| `paper/repro_silhouette_vs_d.pdf` | Created |
| `paper/silhouette_score_comparison_all_files.png` | Regenerated |
| `paper/silhouette_score_comparison_all_files.pdf` | Created |
| `paper/CHI_comparison_all_files.png` | Regenerated |
| `paper/CHI_comparison_all_files.pdf` | Created |
| `paper/DBI_comparison_all_files.png` | Regenerated |
| `paper/DBI_comparison_all_files.pdf` | Created |
| `results/final_reexperiment/figures/repro_silhouette_vs_d.png` | Created |
| `results/final_reexperiment/figures/repro_silhouette_vs_d.pdf` | Created |
| `results/final_reexperiment/figures/silhouette_score_comparison_all_files.png` | Created |
| `results/final_reexperiment/figures/silhouette_score_comparison_all_files.pdf` | Created |
| `results/final_reexperiment/figures/CHI_comparison_all_files.png` | Created |
| `results/final_reexperiment/figures/CHI_comparison_all_files.pdf` | Created |
| `results/final_reexperiment/figures/DBI_comparison_all_files.png` | Created |
| `results/final_reexperiment/figures/DBI_comparison_all_files.pdf` | Created |

---

*Report generated by automated audit on September 1, 2026.*
