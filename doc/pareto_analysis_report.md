# Pareto Analysis Report: Clustering Quality vs Section Balance

**Date:** 2026-09-06
**Method:** All numbers computed from saved artifacts only (no experiments re-run).
Script: `scripts/pareto_analysis.py` → output: `results/grid_search/pareto_analysis.json`.

Objectives (both maximized, no weighted combination):

1. Avg Silhouette Score (6 courses, seed 42)
2. Avg Section Balance `B = min(|S1|,|S2|)/max(|S1|,|S2|)` computed **directly from the saved
   cluster labels** `results/grid_search/labels_p*_q*_d*_course{1..6}.json` (seed 42).

A configuration is *dominated* if another is ≥ in both objectives and > in at least one.

---

## A. Data availability (audit)

| Artifact | Status |
|---|---|
| Per-course Silhouette, all 45 (p,q,d) configs, seed 42 | `results/grid_search/grid_search_complete.json` |
| **Cluster labels for all 45 configs × 6 courses (seed 42)** | `results/grid_search/labels_*.json` (270 files; contain student_index, student_labels, cluster_labels, p, q, d, seed, silhouette) |
| Per-course Silhouette, all 45 configs, seed 0 | `results/reproduced_seed0/grid_search_pqd.json` (no labels) |
| DBI/CH for all 9 (p,q) at d=1 and d=2 (seed 42) | `results/final_d1/sensitivity_pq.json`, `results/final_d2/sensitivity_pq.json` (Silhouettes match the grid exactly → same runs) |
| DBI/CH for neutral (1,1) at all dims (seed 42) | `results/reproduced/sensitivity_dim.json` |
| Baselines BoW/PCA/Spectral + N2V (1,1,2), (1,0.5,2): Sil/DBI/CH | `results/reproduced/all_methods.json` |
| Baseline section balances | `results/reproduced/section_balance.json` |
| 20-seed stability, (1,1,1) — **not** (2,1,1) | `results/final_d1/stability.json`, `ari_stability.json` |
| 20-seed stability, (1,1,2) | `results/reproduced/stability.json`, `ari_stability.json` |
| **20-seed stability, (2,1,1) primary — DONE 2026-09-06** | `results/final_d1_primary/stability.json`, `ari_stability.json`, `ari_stability_all_clusterers.json` |
| Old d=2 sweeps for L, γ, w (seed 42) | `results/reproduced/sensitivity_wl.json`, `_nw.json`, `_ws.json` |

**Consequences:**
- Section balance is computable for **all** grid configurations from saved labels → no reruns needed for the Pareto analysis (todo §3 satisfied).
- Embeddings on disk exist only for a few configs (`embeddings_p1.0_q*.npy`), but they are not needed: labels suffice for balance.
- ⚠ `results/final_d1/*` was produced by `experiments/final_d1_d2_experiments.py` with hardcoded `P=1.0, Q=1.0` (lines 54–55). Its `n11`/`n105` values match grid (1,1,1)/(1,0.5,1) exactly. Anywhere this folder is described as "(p=2, q=1)" is wrong.

## B. Pareto-optimal configurations (seed 42)

AvgSil = mean per-course Silhouette; AvgB = mean per-course Balance; sil_s0 = seed-0 AvgSil.

| p | q | d | AvgSil ↑ | AvgB ↑ | DBI ↓ | CH ↑ | sil_s0 |
|---|---|---|---------|--------|-------|------|--------|
| 2.0 | 1.0 | 1 | **0.7654** | 0.308 | 0.3595 | 304.0 | 0.7729 |
| 0.5 | 0.5 | 1 | 0.7634 | 0.420 | 0.3658 | 10575.4* | 0.7460 |
| 1.0 | 0.5 | 1 | 0.7557 | 0.486 | 0.3749 | 344.7 | 0.7615 |
| 0.5 | 0.5 | 2 | 0.6321 | 0.574 | 0.5143 | 127.8 | 0.6197 |
| 2.0 | 0.5 | 2 | 0.6287 | 0.633 | 0.5268 | 133.8 | 0.6150 |
| 1.0 | 0.5 | 2 | 0.6193 | 0.652 | 0.5416 | 121.2 | 0.6217 |
| 2.0 | 2.0 | 5 | 0.3327 | 0.664 | — | — | 0.3623 |
| 2.0 | 1.0 | 5 | 0.3312 | 0.668 | — | — | 0.3400 |
| 1.0 | 2.0 | 5 | 0.3217 | 0.714 | — | — | 0.3379 |

\* CH is inflated by a degenerate 47/1 split in course 3 — treat with caution.

Per-course detail (sil / B / sizes n1–n2):

- **(2,1,1)** primary: sil 0.963 0.675 0.930 0.619 0.824 0.582; B 0.028 0.645 0.021 0.324 0.156 0.675; sizes 72/2, 20/31, 47/1, 12/37, 45/7, 40/27
- **(0.5,0.5,1)**: sil 0.967 0.594 0.975 0.647 0.788 0.610; B 0.028 0.962 0.021 0.633 0.156 0.718; sizes 72/2, 26/25, 47/1, 19/30, 45/7, 39/28
- **(1,0.5,1)**: sil 0.969 0.646 0.941 0.614 0.778 0.587; B 0.028 0.889 0.021 0.885 0.182 0.914; sizes 72/2, 27/24, 47/1, 23/26, 44/8, 35/32
- **(1,1,1)** neutral d=1: sil 0.963 0.593 0.927 0.634 0.838 0.613; B 0.028 0.645 0.021 0.361 0.156 0.811; sizes 72/2, 20/31, 47/1, 13/36, 45/7, 30/37
- **(1,1,2)** neutral d=2: sil 0.599 0.614 0.605 0.628 0.745 0.583; B 0.542 0.645 0.600 0.633 0.333 0.595; sizes 48/26, 31/20, 30/18, 30/19, 39/13, 42/25

Baselines (seed 42): BoW 0.1572/0.4651 · PCA+KMeans 0.4656/0.4684 · Spectral 0.1279/0.4711 ·
N2V(1,1,2) 0.6290/0.5581 · N2V(1,0.5,2) 0.6193/0.6518 (AvgSil/AvgB).

**Findings**

1. **Every d=1 config collapses courses 1 and 3** (sizes 72/2 and 47/1; B ≈ 0.02–0.03). The high Silhouette is partly an artifact of near-singleton clusters. No configuration with AvgSil ≥ 0.7 achieves AvgB ≥ 0.6 — the "close to frontier" set (sil ≥ 0.7 ∧ B ≥ 0.6) is **empty**.
2. The quality objective and the balance objective are in direct conflict: the best-balance configs sit at d=5 (sil ≈ 0.33, useless), and the best practical compromise zone is d=2.
3. The currently promoted primary (2,1,1) is Pareto-optimal but has the **worst balance (0.308)** among all d≤2 configurations; baselines average B ≈ 0.47.
4. The neutral (1,1,2) is **not** on the frontier (dominated by (0.5,0.5,2): 0.6321 > 0.6290 and 0.574 > 0.558) but is close, fully balanced-looking (no B below 0.33), DeepWalk-equivalent, and already used throughout the paper.
5. Seed robustness: at seed 0 every d=1 config again beats every d≥2 config (best d=2 ≈ 0.622), so *dimension* conclusions are seed-robust; the *ranking within d=1* is not (seed 0 best is (2,2,1)=0.7794; (2,1,1) is 4th of 9). Avoid claims tied to the exact (p,q) winner at d=1.

## C. Recommended primary configuration

**Strategy B** — keep **(p=2.0, q=1.0, d=1)** as the primary configuration for clustering quality (it remains the seed-42 grid winner and is defensible), **but** present it together with an explicit balance-oriented counterpart and a clear trade-off statement:

- Primary (quality): **(2, 1, 1)** — AvgSil 0.7654, AvgB 0.308. State openly that it produces degenerate sections in courses 1 and 3.
- Trade-off counterpart: **(1, 1, 2)** — AvgSil 0.6290, AvgB 0.558, no course below B=0.33; still +35% Silhouette over PCA. It is the natural "practical" choice and is already in every table.
- Optionally mention **(1, 0.5, 2)** as the balance-optimal frontier member at d≤2 (AvgB 0.652, AvgSil 0.6193).

Rationale: the paper's claim is that Node2Vec is an *effective representation*; the evidence shows the representation is excellent for separation at d=1 but that uncoupled KMeans produces impractical section sizes there. Silhouette-maximization alone would be misleading; selecting (1,1,2) alone would discard the strongest quality result. Reporting the frontier honestly is the scientifically strongest presentation and requires no new experiments.

## D. Alternative configuration

Best quality/balance trade-off overall: **(1, 0.5, 2)** (AvgSil 0.6193, AvgB 0.652) or the neutral **(1, 1, 2)** (0.6290 / 0.558). Both dominate every baseline on Silhouette while matching or beating baseline balance. If a single configuration must serve both goals, (1,1,2) is the safest choice (it is DeepWalk-equivalent, visually plottable, and never degenerate).

## E. Manuscript items that need attention (report only — nothing edited)

1. **`tab:stability` + §"Clustering Stability"** — ✅ **RESOLVED 2026-09-06.** The 20-seed run at the true primary (2,1,1) completed (`run_stability_primary_d1.sh` → `results/final_d1_primary/`). GMM/Agglomerative/BoW columns were recomputed from the saved seed-42 embeddings via `scripts/fill_stability_table.py` → `ari_stability_all_clusterers.json`. `paper/sn-article.tex` now carries the real primary-config numbers: KMeans 0.980 (course 6: 0.881), GMM 0.907 (course 6: 0.442), Agglomerative 1.000, BoW 0.846 (course 1: 0.381); protocol sentence and caption updated accordingly. (The `.txt` mirror is intentionally left untouched per user instruction.)
2. **`tab:sensitivity_walk`, `tab:sensitivity_walks`, `tab:sensitivity_window`**: captions claim d=1, p=2, q=1; the values are the **old p=1, q=1, d=2 sweeps** (verified against `sensitivity_wl/nw/ws.json`, e.g. L=80 → 0.637). Either rerun the sweeps at (2,1,1) or restore captions to the neutral configuration. Note the current summary sentence ("we adopt d=1, p=2.0, q=1.0 … L=80 can provide further improvement") mixes two different configurations.
3. **`tab:n2v_sensitivity` caption** says "at d=2" while the row contains the d=1 averages (0.7634 … 0.7654). Caption-only fix.
4. **`fig:baseline_comparison`** (`paper/baseline_comparison.pdf` is byte-identical to the old `results/reproduced/figures/baseline_comparison.pdf`): bars show (1,1,d=2)/(1,0.5,d=2); the adjacent table's Node2Vec d=1 column has no visual counterpart. Regenerate from `final_d1` + `reproduced` data or adjust the caption.
5. **§Section Balance Analysis**: "The graph-based methods generally produce more balanced sections than the conventional baselines" is false for the primary d=1 config (0.308 < 0.465 BoW). Rewrite around the trade-off; optionally cite the Pareto frontier and the per-config `pareto_analysis.json`.
6. **Limitations §**: "B ranges 0.268–0.700 (average 0.530)" is stale. Current artifacts: primary d=1 range 0.021–0.675 (avg 0.308); neutral d=2 range 0.333–0.645 (avg 0.558).
7. **Abstract / Intro / Conclusion**: "consistently surpasses … (100% win rate)" holds for Silhouette only; add one clause acknowledging that the highest-separation configuration yields unbalanced sections and that the neutral d=2 setting is the balanced alternative.
8. **§grid_search**: add one sentence noting the seed-0 replication confirms the d≫1 degradation and d=1 superiority (top-5 all d=1) while the exact (p,q) ranking varies — pre-empting reviewer questions about single-seed selection.
9. Minor: `tab:runtime` caption should state it is the d=2 pipeline (it is the reproduced artifact); `results/grid_search/selection_rationale.json`'s "if d=1 is preferred" note is obsolete.

Figure check (todo §8): `paper/grid_heatmap_all_d_independent.pdf` is identical to the per-panel independent-normalization version in `results/grid_search/figures/` and matches the manuscript caption. Per-d heatmaps with independent scales also exist. **No figure work required.**

## F. Additional execution

**NO ADDITIONAL EXPERIMENTS REQUIRED** for the Pareto analysis, the configuration recommendation, or the balance table — everything above is derived from saved artifacts.

**The previously optional rerun is DONE** (2026-09-06, on the user's Linux system): 20-seed stability at (p=2, q=1, d=1) completed via `run_stability_primary_d1.sh`, writing only into `results/final_d1_primary/` (no existing result touched). Per-seed Silhouette: 0.941/0.620/0.960/0.570/0.825/0.616 (avg 0.755). ARI on seed-42 embeddings: 1.000 for courses 1–5, 0.881 ± 0.114 for course 6. Fix #1 is applied in the manuscript; remaining §E items 2–9 are still open.

## G. Bash script (executed 2026-09-06)

```bash
./run_stability_primary_d1.sh
```

- Targets the Linux repo path `/data/git/mamintoosi/Node2Vec-SSP` (falls back to the script's own directory); interpreter `/data/python-envs/pytorch/bin/python` (override with `PYTHON_BIN=...`).
- Limits to ≤ 2 cores: `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2` + `taskset -c 0,1`.
- Reuses the existing pipeline (`experiments/final_d1_d2_experiments.py`) with `P=2.0, Q=1.0` — no parallel implementation.
- Saved per-seed labels (`labels_course{i}_n11_seed{0..19}.json`), `stability.json`, `ari_stability.json`, and seed-42 embeddings under `results/final_d1_primary/`. Protocol identical to the existing stability analysis (per-seed walks/Word2Vec/KMeans; all-pairs ARI on seed-42 embeddings).
- Resume support: if `stability.json` already exists, the 20-seed step is skipped and only the ARI step runs.
- Follow-up table filler: `python scripts/fill_stability_table.py` recomputes GMM/Agglomerative/BoW ARI columns from the saved embeddings + data and prints LaTeX rows → `ari_stability_all_clusterers.json`.

---

## H. Paper restructure (2026-09-07): Pareto winner promoted to primary

### H.1 Multi-seed verdict (20 seeds × 6 courses, identical protocol; `scripts/pareto_final_comparison.py` → `results/final_pareto_validation/comparison_3candidates.json`)

| Config | Sil (20 seeds) | B (20 seeds) | worst-course B | KMeans ARI |
|---|---|---|---|---|
| (2,1,1) quality-optimal | **0.755 ± 0.037** | 0.406 | 0.021 (degenerate) | 0.980 |
| (1,1,2) neutral | 0.618 ± 0.012 | **0.606** | 0.292 | 0.984 |
| **(1,0.5,2) balance-optimal** | 0.629 ± 0.018 | 0.602 | **0.349** | **0.994** |

**(1,0.5,2) dominates (1,1,2)** on Silhouette, worst-course balance, and ARI (tied on mean balance) and avoids (2,1,1)'s degenerate sections → adopted as the **primary configuration**; (2,1,1) and (1,1,2) are now presented as ablations. Seed-42 per-course values for the primary: Sil 0.594/0.608/0.569/0.597/0.746/0.602 (avg 0.619, +33% over PCA, 100% win rate vs BoW and PCA); B 0.609/0.594/0.714/0.815/0.368/0.811 (avg 0.652 — best of all methods, no course below 0.368).

### H.2 Manuscript changes (`paper/sn-article.tex` only)

- Abstract/Intro/Contributions: 64%→33% improvement claim; Pareto-based selection replaces the "d=1 is best" framing.
- Methods §grid_search: adoption paragraph rewritten to point to the trade-off analysis.
- **New subsection §"Quality--Balance Trade-off and Primary Configuration Selection" (`sec:pareto`) with new Table `tab:pareto_selection`** (3 candidates, seed-42 + 20-seed Sil/B, min-course B, ARI).
- Core tables now show a single Node2Vec column = primary (1,0.5,2): `tab:baseline_comparison` (avg 0.619), `tab:section_balance` (avg 0.652), `tab:stability` (KMeans 0.994, GMM 0.897, Agg 1.000, BoW 0.846 — recomputed at the primary config).
- **New subsection §"Ablation: Alternative Node2Vec Configurations" (`sec:ablation`)**: quality-optimal (2,1,1) and neutral (1,1,2) analyses incl. degenerate splits and course-6 bimodality.
- Sensitivity captions corrected: `tab:n2v_sensitivity` d=2→**d=1** (values are the d=1 grid row); `tab:sensitivity_walk/walks/window` d=1,p=2,q=1 → **d=2, p=1, q=1** (values are the neutral sweeps — resolves report items E2/E3).
- Figure: `fig:baseline_comparison` now uses the newly generated `baseline_comparison_primary.pdf` (4 methods, primary config) — resolves E4. Balance narrative (E5), Limitations range (E6: 0.368–0.815, avg 0.652), Abstract/Intro/Conclusion (E7), and the default-configuration sentence (E9) all updated. Visualization section simplified (primary is itself d=2) and the **pre-existing broken `\ref{fig:visualization}` fixed** to `fig:tsne_course5`.
- Verified: compiles with pdflatex, zero errors; the only remaining overfull boxes (≤10pt) are pre-existing.

### H.3 Scripts created/used in this phase — status and what YOU need to run

**You do NOT need to run anything.** All scripts below were already executed locally against saved artifacts; their outputs are committed under `results/` and `paper/`.

| Script | Purpose | Status | Output |
|---|---|---|---|
| `run_pareto_validation.sh` | 20-seed validation of (1,0.5,2) | ✅ run by user on Linux | `results/final_pareto_validation/` |
| `scripts/pareto_final_comparison.py` | 3-candidate comparison on identical footing | ✅ run locally (light) | `results/final_pareto_validation/comparison_3candidates.json` |
| `scripts/fill_stability_table_pareto.py` | KMeans/GMM/Agg/BoW ARI at the primary config from saved seed-42 embeddings | ✅ run locally (light, no Node2Vec) | `results/final_pareto_validation/ari_stability_all_clusterers.json` |
| `scripts/plot_baseline_comparison_primary.py` | Regenerate baseline figure with the primary config | ✅ run locally (light) | `paper/baseline_comparison_primary.{pdf,png}` + `results/final_pareto_validation/figures/` |

Optional reproduction (only if you want to regenerate on Linux; each takes < 1 min on 2 cores and writes only the files above):

```bash
python scripts/fill_stability_table_pareto.py
python scripts/plot_baseline_comparison_primary.py
```

No experiment reruns are needed anywhere: every number in the revised manuscript traces to an existing artifact (grid labels, `final_d1_primary/`, `final_d2/`, `final_pareto_validation/`, `reproduced/`).
