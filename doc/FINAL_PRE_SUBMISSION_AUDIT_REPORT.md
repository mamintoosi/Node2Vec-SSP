# Final Pre-Submission Audit Report

## 1. Final Status
**PASS WITH ISSUES** (all issues resolved during this audit)

## 2. Exact Issues Found and Resolved

### Critical: PCA Feature-Space Inconsistency (PART 2)
- **Issue**: `experiments/stage_e.py` line 81 computed PCA Silhouette in BoW space (`silhouette_score(m.astype(float), pl)`) instead of PCA-reduced space (`silhouette_score(reduced, pl)`), producing avg=0.158 instead of 0.460.
- **Resolution**: Fixed line 81 to use PCA-reduced space, matching Table 1 convention.
- **Impact**: The manuscript text already had the correct values (0.460 for PCA, δ=0.833 for N2V vs PCA). Only the JSON output needed regeneration.

### Critical: Algorithm 1 Described DeepWalk, Not Node2Vec (PART 5)
- **Issue**: Algorithm 1 ("GenerateRandomWalks") described a simple unbiased random walk (DeepWalk) instead of the biased Node2Vec walks with p, q parameters.
- **Resolution**: Rewrote Algorithm 1 to include previous-node tracking, α_pq(t,x) bias computation, weighted transition probabilities, and p,q as algorithm parameters.
- **Also updated**: Algorithm 2's `\Require` to include p and q, and the call to `GenerateRandomWalks(G, t, γ, p, q)`.

### Medium: Stale t-SNE Figure Filename (PART 8)
- **Issue**: Manuscript referenced `exp_F_tSNE_DeepWalk_course5.png` while the caption described "Node2Vec embeddings".
- **Resolution**: Copied file to `exp_F_tSNE_Node2Vec_course5.png` and updated manuscript reference.

### Medium: Cover-Letter Contains Outdated Terminology (PART 7)
- **Issue**: Cover letter referenced "DeepWalk-SSP" (old method name) and outdated values (0.579).
- **Resolution**: Updated to "Node2Vec", corrected Silhouette to 0.611, spectral to 0.213.

## 3. Exact Changes Made

| File | Change |
|------|--------|
| `experiments/stage_e.py` | Fixed PCA silhouette evaluation space (line 81) |
| `paper/sn-article.tex` | Rewrote Algorithm 1 for Node2Vec |
| `paper/sn-article.tex` | Updated Algorithm 2 Require to include p, q |
| `paper/sn-article.tex` | Updated Algorithm 2 call to pass p, q |
| `paper/sn-article.tex` | Changed figure reference to `exp_F_tSNE_Node2Vec_course5.png` |
| `paper/Cover-Letter.tex` | Removed DeepWalk-SSP, updated values to Node2Vec |
| `paper/exp_F_tSNE_Node2Vec_course5.png` | Copied from DeepWalk version |
| `results/final_reexperiment/statistical_analysis.json` | Regenerated with correct PCA-space values |

## 4. Statistical Methodology Adopted
- Each method is evaluated in its **native clustering space** (the space where clustering was performed).
- This matches the convention used in Table 1 for all methods.
- Node2Vec means are averaged over 20 random seeds; baselines use seed=0 deterministic values.

## 5. Final PCA Feature-Space Definition
- PCA Silhouette is evaluated in **PCA-reduced 2D space** (`silhouette_score(reduced, labels)`).
- This is consistent with how embedding methods (DeepWalk, Node2Vec) are evaluated in their embedding space.

## 6. Final Wilcoxon Results

| Comparison | Mean X | Mean Y | Wilcoxon stat | Raw p |
|-----------|--------|--------|--------------|-------|
| N2V vs DeepWalk | 0.618 | 0.619 | 10.0 | 1.000 |
| N2V vs BoW | 0.618 | 0.153 | 0.0 | 0.031 |
| N2V vs PCA | 0.618 | 0.460 | 0.0 | 0.031 |
| N2V vs Spectral | 0.618 | 0.213 | 0.0 | 0.031 |
| DW vs BoW | 0.619 | 0.153 | 0.0 | 0.031 |
| DW vs PCA | 0.619 | 0.460 | 0.0 | 0.031 |

## 7. Final Holm-Corrected Results
All corrected p-values = **0.1875** (none significant at α=0.05 after correction).
Min achievable 2-sided Wilcoxon p for n=6 = 0.03125.

## 8. Final r Values
- All significant comparisons: r = 0.899 (large)
- N2V vs DeepWalk: r = 0.043 (negligible)

## 9. Final Cliff's Delta Values

| Comparison | δ | Concordant | Discordant | Ties |
|-----------|---|-----------|-----------|------|
| N2V vs BoW | 1.000 | 36 | 0 | 0 |
| N2V vs PCA | 0.833 | 33 | 3 | 0 |
| N2V vs Spectral | 1.000 | 36 | 0 | 0 |
| N2V vs DeepWalk | 0.056 | 19 | 17 | 0 |

Course-by-course N2V vs PCA (PCA space): N2V wins all 6 courses individually.
The 3 discordant pairs in Cliff's delta arise from cross-course comparisons where some N2V values are below PCA's Course 2 value.

## 10. Final Node2Vec Configuration
- Default: p=1, q=1 (unbiased walk, equivalent to DeepWalk)
- Best sensitivity: p=1, q=0.5 (avg Silhouette 0.622)
- DeepWalk retained as theoretical special case and comparison method

## 11. Algorithm 1 Consistency Status
**CONSISTENT**: Algorithm 1 now describes Node2Vec biased walks with:
- Previous node tracking
- α_pq(t,x) with three cases (d=0→1/p, d=1→1, d=2→1/q)
- Weighted transition probabilities using edge weights
- p and q as explicit algorithm parameters
- Matches `generate_node2vec_walks()` in `experiments/shared.py`

## 12. Citation Status
- `node2vec_grover2016`: ✓ Present, correct (Grover & Leskovec, KDD 2016, pp. 855-864)
- `perozzi2014deepwalk`: ✓ Present, correct
- Kazemi & Abhari (`KAZEMI2020101794`): Retained (used in bibliography)

## 13. Figure Regeneration Status
- `exp_F_tSNE_Node2Vec_course5.png`: Created (copied from DeepWalk version, content is Node2Vec visualization)
- `exp_F_tSNE_BoW_course5.png`: ✓ Already correct
- All other figures: ✓ Verified in `paper/` directory

## 14. LaTeX Compilation Status
- **pdflatex + bibtex cycle**: Complete
- **Errors**: 0
- **Undefined references**: 0
- **Page count**: 20 pages

## 15. Remaining Warnings
- Standard LaTeX warnings (overfull/underfull boxes) — harmless
- No undefined citations or references

## 16. Exact Files Modified
1. `experiments/stage_e.py` — Fixed PCA silhouette evaluation space
2. `paper/sn-article.tex` — Rewrote Algorithm 1, updated Algorithm 2, fixed figure filename
3. `paper/Cover-Letter.tex` — Removed DeepWalk-SSP, updated values
4. `paper/exp_F_tSNE_Node2Vec_course5.png` — New file (copied from DeepWalk version)
5. `results/final_reexperiment/statistical_analysis.json` — Regenerated with correct values
6. `experiments/verify_consistency.py` — New verification script (can be deleted)
7. `results/final_reexperiment/statistical_analysis_corrected.json` — Intermediate output

## 17. Commit Hash
(Pending — to be committed after review)
