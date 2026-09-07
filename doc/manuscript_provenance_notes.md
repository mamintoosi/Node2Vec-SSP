# Manuscript Provenance Notes

Date: 2026-09-08
Scope: provenance audit of every table and figure in `paper/sn-article.tex`.

## 1. SOURCE comments added to the manuscript

Every table and figure float in `paper/sn-article.tex` now carries a `% SOURCE:`
LaTeX comment (not rendered) recording the data artifact and/or the script that
produced it, for future reference. Inventory (22 floats):

| Float | Values source | Producing script |
|---|---|---|
| `fig:framework` | author-prepared schematic (`graphical_abstract.pdf`) | — |
| `fig:random-walk` | author-prepared illustration (`random-walk-2-node40.png`) | — |
| `fig:cbow_skipgram` | author-prepared diagram (`CBOW-Skip-gram.png`) | — |
| `fig:grid_search_heatmaps` | `results/grid_search/grid_search_complete.json` (seed 42) | `experiments/grid_search_final.py` |
| `tab:grid_search_top5` | `results/grid_search/grid_search_complete.json` (seed 42) | `run_grid_search.sh` / `experiments/grid_search_final.py` |
| `tab:pareto_selection` | `results/final_pareto_validation/comparison_3candidates.json` (20-seed data in `final_d1_primary/`, `final_d2/`, `final_pareto_validation/`) | `scripts/pareto_final_comparison.py` |
| `tab:baseline_comparison` | `results/final_d2/all_methods.json` ("n105") + `results/reproduced/all_methods.json` (seed 42) | `experiments/final_d1_d2_experiments.py`, `experiments/run_all.py` |
| `fig:baseline_comparison` | same JSONs as above | `scripts/plot_baseline_comparison_primary.py` |
| `tab:graph_stats` | ⚠ see flag F1 | values inserted by `experiments/revise_manuscript.py` |
| `tab:section_balance` | `results/grid_search/labels_p1.0_q0.5_d2_course{1..6}.json` (seed 42) + `results/reproduced/section_balance.json` | `scripts/pareto_analysis.py`, `experiments/section_balance.py` |
| `tab:n2v_sensitivity` | `results/grid_search/grid_search_complete.json`, d=1 row (≡ `results/final_d1/sensitivity_pq.json`) | grid-search run |
| `fig:silhouette` | `results/reproduced/sensitivity_dim.json` (neutral p=1, q=1) | `experiments/run_all.py` (`fig_sil_vs_d`) |
| `tab:ablation_dimension` | `results/reproduced/sensitivity_dim.json` | `experiments/run_all.py` |
| `tab:sensitivity_walk` | `results/reproduced/sensitivity_wl.json` | sensitivity suite |
| `tab:sensitivity_walks` | `results/reproduced/sensitivity_nw.json` | sensitivity suite |
| `tab:sensitivity_window` | `results/reproduced/sensitivity_ws.json` | sensitivity suite |
| `fig:silhouette_score_comparison` | `results/reproduced/all_methods.json` (bow, pca, spec, n11, n105) | `experiments/run_all.py` (`fig_all_metrics`) |
| `tab:stability` | `results/final_pareto_validation/ari_stability_all_clusterers.json` (ARI over 20 KMeans inits on fixed seed-42 embeddings of the primary config) | `scripts/fill_stability_table_pareto.py` |
| `fig:DBI_comparison` | `results/reproduced/all_methods.json` (per-course DBI) | `experiments/run_all.py` (`fig_all_metrics`) |
| `fig:CHI_comparison` | `results/reproduced/all_methods.json` (per-course CH) | `experiments/run_all.py` (`fig_all_metrics`) |
| `fig:tsne_course5` | `exp_F_tSNE_*_course5.png`; Node2Vec panel = neutral (1,1,2) via `experiments/core.run_pipeline` (DEFAULT_PARAMS) | `experiments/visualize_courses.py` |
| `tab:runtime` | ⚠ see flag F2 | measured on a separate computer |

(`alg:random_walk` is pseudo-code, not data — no comment needed.)

## 2. Stale caption fixed (F0)

`fig:silhouette_score_comparison` previously claimed it compared *clustering
algorithms* (KMeans / affinity propagation / GMM / hierarchical) applied to BoW
and Node2Vec. The actual PDF (verified from its embedded legend) compares
*representations*: BoW+KMeans, PCA+KMeans, Spectral, Node2Vec (p=1, q=1),
Node2Vec (p=1, q=0.5) — all with KMeans clustering. Fixed:

- caption rewritten to describe the representation comparison;
- the intro sentence of §"Clustering Algorithms and Representation Quality" now
  points the clustering-algorithm robustness claim to `tab:stability` and
  describes the figure correctly;
- the follow-up paragraph ("advantage not restricted to KMeans ...") rewritten
  to claims actually supported by current artifacts (per-course wins over BoW,
  6/6 wins of the primary config over PCA per `sec:statistical`);
- §setup: removed "affinity propagation" from the list of evaluated clustering
  methods (it appears in no current table);
- Discussion: "improvement is largely independent of the clustering algorithm"
  softened to the stability-evidence statement (`tab:stability`).

## 3. Open flags (kept as comments in the .tex; fix if the paper is revised)

- **F1 — `tab:graph_stats` is not reproducible from archived JSONs.** The
  values correspond to the co-enrollment graph *after* constant-feature removal
  (e.g., course 1: |E|=1828 of 2701 possible edges → density 0.677, 3
  components). `results/reproduced/graph_stats.json` stores the graph *before*
  removal (complete graph, 2701 edges, 1 component) and does not match. The
  numbers were inserted verbatim by `experiments/revise_manuscript.py`. If
  regenerating, recompute graph stats after constant-feature removal.

- **F2 — `tab:runtime` was measured on a separate computer.** Per the note in
  `experiments/compute_manuscript_values.py` ("Runtime — DO NOT CHANGE
  (measured on another computer)"), the per-course breakdown exists in no
  archived `results/*.json`. Do not regenerate without re-measuring on
  comparable hardware.

- **F3 — generator scripts reference unarchived folders.**
  `experiments/final_audit_and_figures.py` reads `results/final_reexperiment/`
  (no longer present); the figure PDFs currently in `paper/` were produced
  earlier from that run. The archived equivalents live in
  `results/reproduced/`. This is recorded in the SOURCE comments.
