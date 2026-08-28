# Manuscript Revision Report

**Date:** 2026-08-28 23:27 UTC  
**Commit:** Pending (to be committed after review)

---

## 1. Summary

The manuscript (`paper/sn-article.tex`) has been revised to reflect the final validated experimental pipeline in which Node2Vec replaces DeepWalk as the primary graph representation-learning method, and constant-course preprocessing is documented.

## 2. Sections Changed

### Abstract
- Removed DeepWalk as a separately reported method
- Added Node2Vec as the primary graph embedding method
- Added mention of constant-feature removal preprocessing
- Updated all numerical results to match final experiments
- Added caveat about Holm-corrected statistical significance
- Updated keywords (removed DeepWalk)

### Introduction
- Revised research framing: Node2Vec as the main method instead of "DeepWalk-SSP"
- Updated the research question and approach description
- Revised all five contributions:
  - Contribution 1: Node2Vec instead of DeepWalk+Node2Vec
  - Contribution 3: Updated numbers (0.611, 0.460, 0.153, 0.213), added caveat about Holm correction
- Updated experimental results preview paragraph

### Related Work
- Minor update to position Node2Vec as the main method

### Methodology

#### New subsection: "Preprocessing: Removal of Constant Features" (Section 3.2)
- Documents the constant-course removal applied before all methods
- Notes that exactly one constant course was removed per sectioning instance
- States that the resulting graphs are non-complete (average density ~0.72)

#### Updated: "Random-walk-based Student Representation Learning" (Section 3.3)
- Replaced the simple random walk description with the full Node2Vec transition probability formulation
- Added the bias factor α_pq(t,x) with the three-case definition
- Explained the p (return) and q (in-out) parameters
- Noted that p=1, q=1 recovers DeepWalk

#### Updated: Algorithm 2 caption
- Changed from "Student Representation Generation using DeepWalk" to "...using Node2Vec"

### Experimental Results

#### Updated: Dataset and Experimental Setup
- Changed method name from "DeepWalk-SSP" to "Node2Vec"
- Updated default parameters: added p=1, q=1 as default
- Mentioned the Node2Vec parameter sensitivity experiment

#### New table: Graph Statistics (Table 3)
- Reports per-course graph statistics after preprocessing
- Shows non-complete graphs with density 0.599–0.887
- Notes isolated students in Courses 1 and 3

#### New section: "Node2Vec Parameter Sensitivity" (Section 4.3)
- Reports 3×3 grid of (p,q) configurations
- Best: p=1.0, q=0.5 → avg Silhouette 0.622
- Neutral: p=1.0, q=1.0 → 0.611
- Sensitivity range: 0.605–0.622 (modest)

#### Updated: Main Comparison Table (Table 1)
- Revised per-course values with new graph construction
- Methods: BoW, PCA, Spectral, DeepWalk, Node2Vec (p=1, q=1)
- New averages: 0.153, 0.460, 0.213, 0.613, 0.611

#### Updated: Results Discussion
- Node2Vec (p=1, q=1) and DeepWalk have near-identical scores (0.611 vs 0.613)
- Explained theoretical equivalence at p=1, q=1
- Noted modest effect of q parameter
- Updated spectral clustering from 0.151 to 0.213

#### Updated: Statistical Significance (Section 4.6)
- Restructured around Node2Vec comparisons (vs BoW, vs PCA, vs DeepWalk)
- Raw p=0.031, Holm-corrected p=0.188 (not significant at α=0.05)
- Effect sizes: r=0.899 (Large), Cliff's δ=1.0
- Honest about n=6 limitation
- Removed claims of "statistically significant improvement"

#### Updated: Clustering Algorithms section
- Changed "DeepWalk-SSP" to "Node2Vec" throughout

#### Updated: Stability section
- Changed "DeepWalk-SSP" to "Node2Vec" in table and text

#### Updated: DBI and CH comparison
- Changed "DeepWalk-SSP" to "Node2Vec"

#### Updated: Embedding Visualization
- Changed "DeepWalk-SSP" to "Node2Vec" in captions and text

#### Updated: Runtime Analysis
- Changed "DeepWalk-SSP" to "Node2Vec"

### Discussion
- Complete revision of the Discussion section
- Node2Vec as primary method throughout
- Documented graph density after preprocessing (not complete)
- Discussed theoretical equivalence of Node2Vec(p=1,q=1) and DeepWalk
- Noted modest parameter sensitivity (0.605–0.622)
- Added honest caveat about Holm correction
- Revised stability discussion (mean std 0.013)

### Conclusion
- Complete revision of the Conclusion
- Node2Vec as primary method
- Updated all numerical results
- Removed DeepWalk-specific claims
- Added honest caveat about statistical significance after Holm correction
- Revised future work section

## 3. Numerical Values Updated

| Metric | Old Value | New Value | Source |
|--------|-----------|-----------|--------|
| BoW avg Silhouette | 0.153 | 0.153 | Unchanged |
| PCA avg Silhouette | 0.460 | 0.460 | Unchanged |
| Spectral avg Silhouette | 0.151 | **0.213** | Revised graph construction |
| DeepWalk avg Silhouette | 0.579 | **0.613** | Revised graph construction |
| Node2Vec(p=1,q=1) avg Silhouette | (0.658 with p=0.5,q=1) | **0.611** | Revised graph construction |
| Node2Vec best (p=1,q=0.5) | N/A | **0.622** | New experiment |
| Statistical significance | p=0.016 (significant) | p=0.031 raw, **0.188 Holm** (not significant) | Holm correction |
| Cliff's δ | 1.0 | 1.0 | Unchanged |
| Effect size r | 0.899 | 0.899 | Unchanged |
| Graph density (avg) | 1.0 (complete) | **0.721** | After preprocessing |

## 4. How Node2Vec Is Positioned

- Node2Vec is the **primary graph representation-learning method** in the paper
- DeepWalk is mentioned as the special case of Node2Vec at p=1, q=1
- No new algorithm name (e.g., "Student2Vec") is introduced
- The paper uses "Node2Vec", "Node2Vec (p=1, q=1)", "Node2Vec (p=1, q=0.5)" as appropriate

## 5. How DeepWalk Is Positioned

- DeepWalk is presented as the unbiased random-walk special case of Node2Vec
- The near-identical scores (0.613 vs 0.611) are presented as consistent with theoretical equivalence
- DeepWalk is not presented as a separate proposed method
- The "DeepWalk-SSP" name is no longer used in active text (only in comments/GitHub URLs)

## 6. How Constant-Course Preprocessing Is Described

- Added as a new subsection "Preprocessing: Removal of Constant Features" in Methodology
- Described as standard zero-variance feature removal
- NOT presented as a novel contribution
- Applied consistently to all methods

## 7. How Statistical Significance Is Described

- **Raw** Wilcoxon p=0.031 for graph methods vs baselines
- **Holm-corrected** p=0.188 (not significant at α=0.05)
- n=6 courses explicitly noted as a limitation
- Effect sizes (r=0.899, δ=1.0) emphasized as more informative than p-values
- No claim of "statistically significant improvement" after correction

## 8. Figures/Tables Integrated

### New tables added:
- Table 3: Graph statistics after preprocessing (new)
- Table 5: Node2Vec parameter sensitivity (new)

### Updated tables:
- Table 1: Main comparison (revised numbers)

### Existing figures retained:
- Figure 1: Framework overview (unchanged)
- Figure 2: Random walk example (unchanged)
- Figure 3: CBOW/Skip-gram (unchanged)
- Figure 4: Baseline comparison (captions updated)
- Figure 5: Embedding dimension (captions updated)
- Figure 6-7: Clustering algorithm comparison (captions updated)
- Figure 8-9: DBI/CH comparison (captions updated)
- Figure 10: t-SNE visualization (captions updated)

### Figures NOT yet regenerated:
- The comparison figures in `paper/` still use old values
- New figures are available in `results/final_reexperiment/figures/`
- These need to be copied to `paper/` in a future step

## 9. Claims Removed or Softened

1. **Removed**: "Node2Vec-SSP achieves the highest average Silhouette Score of 0.658"
   - **Replaced with**: "Node2Vec achieves 0.611 with p=1,q=1; best config p=1,q=0.5 achieves 0.622"

2. **Removed**: "Both graph embedding methods outperform PCA and the conventional representation"
   - **Replaced with**: Honest comparison noting PCA+KMeans is competitive

3. **Removed**: "statistically significant improvement" (p=0.016)
   - **Replaced with**: "Raw p=0.031, Holm-corrected p=0.188; not significant after correction"

4. **Removed**: "all six graphs are fully connected"
   - **Replaced with**: Non-complete graphs after preprocessing (avg density 0.721)

5. **Removed**: "the q parameter has limited effect because graphs are fully connected"
   - **Replaced with**: "Modest q effect (range 0.605–0.622); BFS-like bias slightly better"

## 10. Files Modified

| File | Status |
|------|--------|
| `paper/sn-article.tex` | **Modified** - Main manuscript revision |
| `paper/sn-article.tex.bak` | **Created** - Backup of original |
| `experiments/revise_manuscript.py` | **Created** - Script for bulk edits |
| `experiments/fix_dsprem.py` | **Created** - Script for DeepWalk-SSP cleanup |
| `experiments/fix_dsprem2.py` | **Created** - Script for remaining replacements |
| `doc/MANUSCRIPT_REVISION_REPORT.md` | **Created** - This report |

## 11. Git Status

- `paper/` is tracked (confirmed)
- `doc/` is tracked (confirmed)
- Manuscript backup created before edits
- Ready for commit

## 12. Recommendation

The manuscript has been updated to accurately reflect the validated experimental results. Key remaining items for a future step:

1. **Regenerate comparison figures** with the new numbers (currently the figures in `paper/` use old values)
2. **Verify LaTeX compilation** (the manuscript should compile without errors)
3. **Review the revised tables and figures** for visual consistency
4. **Consider whether to include individual course breakdowns** for DBI and CH in addition to averages
5. **Add a formal limitations section** if required by the journal
