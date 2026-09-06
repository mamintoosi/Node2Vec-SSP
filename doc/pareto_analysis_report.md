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

1. **`tab:stability` + §"Clustering Stability"** (claim: "grid-search-selected primary configuration (p=2.0,q=1.0,d=1)"). The underlying `results/final_d1/stability.json` is (1,1,1). Fix by rerunning (script provided, §G) or by relabeling the text to the neutral d=1 configuration.
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

**One optional rerun** (only if you want `tab:stability` to truly describe the primary configuration): 20-seed stability at (p=2, q=1, d=1), ≈5–10 min on 2 cores. Provided as `run_stability_primary_d1.sh`; writes to a new folder `results/final_d1_primary/` and never deletes or overwrites existing results. If you skip it, apply fix #1 by relabeling instead.

## G. Bash script (manual execution, ≤ 2 cores)

```bash
./run_stability_primary_d1.sh
```

- Sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2` and, where available, `taskset -c 0,1` (Linux) or `START /AFFINITY 3` (Windows) as a second enforcement layer.
- Reuses the existing pipeline (`experiments/final_d1_d2_experiments.py`) with `P=2.0, Q=1.0` — no parallel implementation.
- Saves labels per course/seed (`labels_course{i}_n11_seed{s}.json`), metrics (`stability.json`, `ari_stability.json`), and seed-42 embeddings under `results/final_d1_primary/`. Protocol is identical to the existing stability analysis (per-seed walks/Word2Vec/KMeans; all-pairs ARI on seed-42 embeddings).
