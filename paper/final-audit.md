# FINAL CONSISTENCY AUDIT + REGENERATION OF FOUR FIGURES

You are working inside the existing research repository.

The manuscript has already been revised, and the main experimental pipeline has already been executed and audited. Your task now is a **final consistency audit** followed by **regeneration of exactly four publication figures**.

## IMPORTANT — SCOPE

This is NOT a request to rerun the full experimental pipeline.

Do NOT rerun Stages A–D or any expensive experiments.

Do NOT modify the scientific content of the manuscript.

Do NOT invent a new method name such as Student2Vec.

Do NOT introduce new experiments unless a very small computation is absolutely necessary to verify an existing result.

You MAY:

* inspect the manuscript, source code, JSON/CSV result files, and existing figure-generation scripts;
* run small verification calculations;
* fix a figure-generation script if necessary;
* regenerate the four specified figures;
* update documentation describing what you found;
* commit and push the complete repository at the end.

The previous execution/audit reports are available in `doc/`. In particular, inspect the most recent relevant reports, including:

* `doc/REPORT.md`
* `doc/MANUSCRIPT_REVISION_REPORT.md`
* any final audit report concerning the Holm correction and the final re-experiment

Use the actual repository files as the source of truth. Do not rely blindly on numbers written in reports.

---

# PART 1 — FINAL CONSISTENCY AUDIT

Perform a read-only audit first.

Create a detailed Markdown report:

`doc/FINAL_CONSISTENCY_AUDIT_REPORT.md`

Do not modify the manuscript during this audit.

## 1. Source of truth for PCA

Determine the actual final PCA + KMeans result from the experiment output files and source code.

Specifically determine:

* the actual per-course Silhouette scores;
* the actual average Silhouette;
* which feature space is used for PCA clustering;
* which feature space is used for Silhouette evaluation;
* the exact source JSON/CSV/result file;
* the exact script/function producing the value.

There has previously been a possible discrepancy between approximately:

* PCA average Silhouette ≈ 0.158
* PCA average Silhouette ≈ 0.460

Do NOT assume either value is correct.

Trace the pipeline and determine the actual final value.

Report:

* manuscript value;
* source-of-truth value;
* whether they match;
* if not, explain exactly why;
* file and code location producing the source-of-truth value.

This is a CRITICAL check.

---

## 2. Source of truth for Table 1

Audit every numerical value in the manuscript's main comparison table.

The relevant methods are expected to include:

* BoW + KMeans
* PCA + KMeans
* Spectral
* DeepWalk
* Node2Vec

and Node2Vec configurations should be identified explicitly where appropriate.

For every method, compare:

* each course-level Silhouette value;
* reported average;
* any DBI/CHI values if present in the table.

Trace every number back to the final result files.

Report a table like:

| Method   | Manuscript | Source of Truth | Match  |
| -------- | ---------: | --------------: | ------ |
| BoW      |        ... |             ... | YES/NO |
| PCA      |        ... |             ... | YES/NO |
| Spectral |        ... |             ... | YES/NO |
| DeepWalk |        ... |             ... | YES/NO |
| Node2Vec |        ... |             ... | YES/NO |

Do not silently correct the manuscript.

---

## 3. Recompute and verify Wilcoxon tests

Using the FINAL per-course results, independently recompute the paired Wilcoxon signed-rank tests.

Use the six courses as the statistical units.

Do NOT treat the 20 random seeds as independent samples.

Verify at least:

* Node2Vec vs BoW
* Node2Vec vs PCA
* Node2Vec vs Spectral
* Node2Vec vs DeepWalk
* and any other pairwise comparison actually reported in the manuscript

For every comparison report:

* W statistic;
* raw p-value;
* n;
* exact input vectors.

Then compare these values against:

`results/final_reexperiment/statistical_analysis.json`

and against the manuscript.

---

## 4. Recompute and verify effect size r

Independently verify the reported effect size `r`.

For every comparison where `r` is reported:

* identify the exact formula used by the existing code;
* independently recompute it;
* report the resulting value;
* compare it with the manuscript and JSON.

Do not assume that a reported `r=0.899` is correct simply because it appears in the previous report.

---

## 5. Recompute and verify Cliff's delta

Independently verify Cliff's δ.

In particular check the important claim:

* graph method vs baseline has δ = 1.0

Determine whether this is actually true for the six paired course-level values.

Report the exact values and interpretation.

---

## 6. Verify Holm correction

Inspect the CURRENT implementation of `holm_correction()`.

The previous audit found and fixed a bug in this function.

Verify that the current implementation really uses the correct Holm step-down monotonicity procedure, e.g. an ascending p-value order followed by a cumulative maximum.

Then independently recompute the corrected p-values.

Expected final values from the previous correction were approximately:

* Node2Vec vs BoW: 0.1875
* Node2Vec vs PCA: 0.1875
* Node2Vec vs Spectral: 0.1875
* DeepWalk vs BoW: 0.1875
* DeepWalk vs PCA: 0.1875
* Node2Vec vs DeepWalk: 1.0000

But DO NOT assume these values are correct. Verify them from the actual current data.

Report:

* raw p;
* Holm-corrected p;
* manuscript value;
* JSON value;
* whether all agree.

---

## 7. Hyperparameter consistency audit

Check whether the manuscript's reported Node2Vec hyperparameters correspond to the FINAL pipeline.

Audit at least:

* embedding dimension `d`
* walk length `t`
* number of walks `γ`
* context/window size `w`
* Word2Vec epochs `ε`
* KMeans `k`
* KMeans `n_init`
* `p`
* `q`

Also distinguish clearly between:

### Main/default configuration

`Node2Vec (p=1, q=1)`

and:

### Best configuration from sensitivity analysis

`Node2Vec (p=1, q=0.5)`

Verify that the manuscript does not accidentally use the best sensitivity configuration as if it were the default main experiment.

Also verify whether any old sensitivity experiment using different `d`, `t`, `γ`, or `w` is still being referenced.

If old values remain anywhere in active manuscript text, report them.

---

## 8. Pseudocode vs actual Node2Vec implementation

Compare the Node2Vec pseudocode in the manuscript with the actual implementation.

Verify that the pseudocode correctly represents:

* current node;
* previous node;
* transition probability;
* `p`;
* `q`;
* distance-to-previous-node cases;
* edge weights;
* treatment of isolated nodes;
* random walk generation.

Pay particular attention to whether the manuscript's Algorithm 1 actually implements Node2Vec rather than the old DeepWalk/simple-random-walk algorithm.

Report every mismatch.

Do not modify the manuscript.

---

## 9. DeepWalk residuals in the manuscript

Search the entire `paper/` tree, especially:

`paper/sn-article.tex`

for:

* `DeepWalk-SSP`
* `DeepWalk SSP`
* `DeepWalk`
* old method names
* old claims associated with DeepWalk-SSP

Distinguish between:

### Acceptable references

References explaining that:

`Node2Vec (p=1,q=1)`

reduces to the unbiased random walk used by DeepWalk.

### Potentially problematic references

Statements that still present:

`DeepWalk-SSP`

as the proposed/main method.

Report every occurrence with:

* filename;
* line number;
* short surrounding snippet;
* classification: acceptable / problematic.

Do NOT modify the manuscript text in this task.

---

## 10. Node2Vec citation

Verify that the manuscript cites the original Node2Vec paper:

Grover and Leskovec, KDD 2016.

Also verify the DeepWalk citation if DeepWalk is discussed.

Check that:

* the citation exists;
* the bibliography entry is correct;
* the citation is attached to the appropriate methodological statement.

Report the citation keys and bibliography entries.

---

# PART 2 — REGENERATE THE FOUR FIGURES

There are four figures in the manuscript that still contain the old label:

`DeepWalk-SSP`

They must be regenerated.

The following files are specifically affected:

1. `repro_silhouette_vs_d.png`
2. `silhouette_score_comparison_all_files.png`
3. `CHI_comparison_all_files.png`
4. `DBI_comparison_all_files.png`

## Critical naming rule

The old proposed method was called `DeepWalk-SSP`.

That name must NOT appear in the regenerated figures.

The corresponding current method is:

**Node2Vec**

When the configuration is the neutral/default configuration, use the explicit label:

**Node2Vec (p=1, q=1)**

Do NOT call it:

* DeepWalk-SSP
* Node2Vec-SSP
* Student2Vec
* Node2Vec-DeepWalk

Use standard Node2Vec terminology.

---

## Figure 1: `repro_silhouette_vs_d.png`

Inspect the existing generation script and determine exactly what this figure represents.

Regenerate it using the FINAL Node2Vec pipeline/results.

If it compares representation quality as a function of embedding dimension `d`, make sure the curve/points correspond to the final Node2Vec implementation and current preprocessing.

The legend/label must use:

`Node2Vec (p=1, q=1)`

where that is the configuration represented.

Do not accidentally reuse old DeepWalk-generated values.

---

## Figure 2: `silhouette_score_comparison_all_files.png`

Regenerate using the final validated experimental results.

Use the current method names and values.

The old:

`DeepWalk-SSP`

label must be removed.

If the corresponding bar represents the old proposed graph method, it must now be labeled:

`Node2Vec (p=1, q=1)`

Preserve the scientific meaning and purpose of the original figure.

Do not fabricate new values.

---

## Figure 3: `CHI_comparison_all_files.png`

Regenerate using the final validated results.

Remove:

`DeepWalk-SSP`

and use the appropriate current label:

`Node2Vec (p=1, q=1)`

or another Node2Vec configuration only if the underlying data demonstrably correspond to that configuration.

Use the actual final CHI values from the result files.

---

## Figure 4: `DBI_comparison_all_files.png`

Regenerate using the final validated results.

Remove:

`DeepWalk-SSP`

and use:

`Node2Vec (p=1, q=1)`

when appropriate.

Use the actual final DBI values.

Remember:

* lower DBI is better;
* do not accidentally invert or normalize the values;
* preserve the original scientific interpretation.

---

# FIGURE GENERATION REQUIREMENTS

Do not simply edit the PNG labels manually.

Use the appropriate Python plotting script/code so that the figures are genuinely regenerated from the final data.

First inspect existing figure-generation scripts and reuse/refactor them where appropriate.

Use publication-quality output.

For each of the four figures generate:

* PDF
* PNG

Save copies in:

`results/final_reexperiment/figures/`

and:

`paper/`

Use the same filenames, with `.pdf` and `.png` extensions as appropriate.

Example:

`paper/repro_silhouette_vs_d.png`

`paper/repro_silhouette_vs_d.pdf`

etc.

After generation, programmatically verify that the old string:

`DeepWalk-SSP`

does not appear in the generated figure source/labels.

If possible, also inspect the rendered figures to ensure:

* labels are not clipped;
* legends are readable;
* axes are correct;
* no old labels remain;
* values correspond to final results.

---

# MANUSCRIPT SAFETY

Do NOT modify:

`paper/sn-article.tex`

Do NOT modify manuscript wording, tables, citations, or equations.

This task is only:

1. audit;
2. regenerate the four figures;
3. document findings.

The existing manuscript should remain byte-for-byte unchanged except for the fact that its referenced figure files may be replaced by regenerated versions.

If the audit discovers a manuscript error, report it in:

`doc/FINAL_CONSISTENCY_AUDIT_REPORT.md`

but do not fix it automatically.

---

# FINAL REPORT

At the end, update/create:

`doc/FINAL_CONSISTENCY_AUDIT_REPORT.md`

The report must contain:

## A. Audit status

* PASS / PASS WITH ISSUES / FAIL

## B. Numerical consistency

A table containing:

* PCA
* BoW
* Spectral
* DeepWalk
* Node2Vec
* Wilcoxon p-values
* Holm p-values
* r
* Cliff's δ

with manuscript/source-of-truth/match columns.

## C. Hyperparameter consistency

List all final values and whether they agree with the manuscript.

## D. Pseudocode consistency

List any mismatches.

## E. DeepWalk residuals

List all remaining occurrences and classify them.

## F. Citation audit

Report Node2Vec and DeepWalk citations.

## G. Figure regeneration

Report:

* all four regenerated successfully;
* PDF created;
* PNG created;
* location;
* data source;
* labels used.

Explicitly confirm:

`DeepWalk-SSP` is absent from the regenerated figures.

## H. Manuscript integrity

Explicitly verify whether:

`paper/sn-article.tex`

was modified.

Expected result: **NOT MODIFIED**.

---

# GIT REQUIREMENT — IMPORTANT

At the very end:

1. Check `git status`.
2. Inspect `git diff`.
3. Confirm that no unintended manuscript changes exist.
4. Confirm that `paper/` is tracked and not ignored.
5. Confirm that `doc/` is tracked.
6. Confirm the four regenerated figures are tracked.
7. Add all intended changes.
8. Commit everything.

Use a descriptive commit message, for example:

`Final consistency audit and regenerate Node2Vec figures`

Then PUSH the commit to the configured remote.

This work is being performed on a different system, so do NOT stop after creating the commit.

The final report MUST state:

* commit hash;
* branch;
* push result;
* final `git status`;
* whether the working tree is clean;
* exact files changed/created.

Do not claim success unless the commands actually succeed.

---

# FINAL RESPONSE

At the end provide a concise execution summary containing:

1. Audit status.
2. PCA source-of-truth result.
3. Table 1 consistency status.
4. Wilcoxon/Holm status.
5. Effect-size status.
6. Hyperparameter consistency status.
7. Pseudocode consistency status.
8. DeepWalk residual status.
9. Citation status.
10. Four figure regeneration status.
11. Confirmation that `paper/sn-article.tex` was not modified.
12. Commit hash.
13. Push status.
14. Final git status.
