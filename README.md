# Revisiting Student Sectioning via Node2Vec: A Graph Representation Learning Framework

A Python framework for solving the student sectioning problem in course timetabling using graph representation learning. The framework constructs student co-enrollment graphs and applies Node2Vec to learn low-dimensional student representations that encode higher-order proximity patterns, producing substantially better clustering inputs than traditional binary enrollment matrices.

**Paper:** "Revisiting the Student Sectioning Problem through Graph Representation Learning"
**Target Journal:** Progress in Artificial Intelligence (Springer)
**Repository:** <https://github.com/mamintoosi/Node2Vec-SSP>

## Key Results (Reproduced with seed=42)

| Method | Silhouette ↑ | DBI ↓ | CHI ↑ |
|--------|-------------|-------|-------|
| BoW + KMeans | 0.157 | 2.074 | 10.69 |
| PCA + KMeans | 0.466 | 0.854 | 50.79 |
| Spectral | 0.128 | 2.234 | 8.38 |
| **Node2Vec (p=1, q=1)** | **0.629** | **0.532** | **134.91** |
| Node2Vec (p=1, q=0.5) | 0.619 | 0.542 | 121.18 |

**Effect sizes (Node2Vec p=1,q=1 vs baselines):**
- vs BoW: Cliff's δ = 1.000 (Node2Vec wins all 6 courses)
- vs PCA: Cliff's δ = 0.889 (Node2Vec wins 5/6 courses)
- vs Spectral: Cliff's δ = 1.000 (Node2Vec wins all 6 courses)
- Effect size r = 0.899 (large) for all graph-vs-baseline comparisons

- Consistent improvement across 6 real-world courses and 3 evaluation metrics
- Node2Vec outperforms PCA+KMeans (33% relative improvement in Silhouette)
- Clustering stability: ARI ≥ 0.917 for KMeans across 20 random seeds
- Runtime: < 9 seconds per course

## Repository Structure

```
Node2Vec-SSP/
├── data/                          # Student enrollment data (6 courses)
│   ├── 1.txt                      # Course 1: 74 students, 35 courses
│   ├── 2.txt                      # Course 2: 51 students, 34 courses
│   └── 3.txt - 6.txt              # Courses 3-6
├── experiments/                   # Experiment modules
│   ├── config.py                  # Central configuration (parameters, paths)
│   ├── shared.py                  # Pipeline: graph construction, walks, Word2Vec
│   ├── evaluation.py              # Clustering algorithms and metrics
│   ├── plotting.py                # Publication-quality figure generation
│   ├── stats.py                   # Wilcoxon tests, effect sizes, bootstrap CI
│   ├── run_all.py                 # Full reproduction script (seed=42)
│   ├── run_experiments.py         # Complete experiment runner (all steps)
│   └── fix_two_figs.py            # Quick figure regeneration from saved JSONs
├── paper/                         # LaTeX source of the paper
│   ├── sn-article.tex             # LaTeX source (Springer template)
│   ├── sn-jnl.cls                 # Springer document class
│   ├── sn-bibliography.bib        # Bibliography
│   ├── graphical-abstract.tex     # Graphical abstract
│   ├── Cover-Letter.tex           # Cover letter
│   └── *.png, *.pdf              # Figures referenced in the paper
├── results/
│   ├── reproduced/                # Latest reproducible results (seed=42)
│   │   ├── all_methods.json       # Main comparison results
│   │   ├── statistical_analysis.json
│   │   ├── sensitivity_pq.json    # (p,q) parameter sweep
│   │   ├── stability.json         # 20-seed stability analysis
│   │   └── figures/               # All generated figures (PDF + PNG)
│   └── final_reexperiment/        # Previous experiment results
├── doc/                           # Reports and documentation
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- NetworkX
- gensim (Word2Vec)
- scikit-learn
- matplotlib
- seaborn
- pandas
- openpyxl

## Installation

```bash
git clone https://github.com/mamintoosi/Node2Vec-SSP.git
cd Node2Vec-SSP
pip install numpy scipy networkx gensim scikit-learn matplotlib seaborn pandas openpyxl
```

## Reproducing Results

Run the full reproduction (all methods, sensitivity, stability, figures — takes ~25–30 minutes):

```bash
cd Node2Vec-SSP
python experiments/run_all.py 2>&1 | tee results/reproduced/run.log
```

This executes all steps sequentially:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Data loading & graph construction | `graph_stats.json` |
| 2 | Main comparison (5 methods × 6 courses) | `all_methods.json` |
| 3 | Node2Vec (p,q) sensitivity (3×3 grid) | `sensitivity_pq.json` |
| 4 | Hyperparameter sensitivity (d, t, γ, w) | `sensitivity_dim.json`, etc. |
| 5 | 20-seed stability analysis | `stability.json`, `ari_stability.json` |
| 6 | Statistical comparisons | `statistical_analysis.json` |
| 7 | Figure generation (11 figures) | `results/reproduced/figures/` |
| 8 | Runtime comparison | `runtime.json` |

All results are saved incrementally to `results/reproduced/`. If the run is interrupted, you can regenerate just the figures from saved JSONs:

```bash
python experiments/fix_two_figs.py
```

### Reproducibility

All experiments use **seed=42** for full reproducibility. The random seed controls Node2Vec random walks, Word2Vec initialization, and KMeans clustering.

## Algorithm

The pipeline consists of three stages:

1. **Graph Construction:** Build a co-enrollment graph where nodes represent students and edges connect students sharing at least one course. Edge weight = number of shared courses.

2. **Random Walk + Word2Vec:** Generate random walks on the graph, then apply Skip-gram Word2Vec to learn low-dimensional embeddings that encode graph proximity.

3. **Clustering:** Apply k-means (or other clustering algorithms) to the learned embeddings for student sectioning.

### Default Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Embedding dimension | *d* | 2 | Vector size (optimal: 1-2) |
| Walk length | *t* | 10 | Steps per random walk |
| Walks per node | γ | 80 | Number of walks starting from each node |
| Window size | *w* | 5 | Skip-gram context window |
| Epochs | ε | 30 | Word2Vec training epochs |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{amintoosi2026graph,
  title={Revisiting the Student Sectioning Problem through Graph Representation Learning},
  author={Amintoosi, Mahmood},
  year={2026},
  journal={Progress in Artificial Intelligence},
  url={https://github.com/mamintoosi/Node2Vec-SSP}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

The author would like to thank Mr. Hasan Fahimian and Mrs. Samira Shahraeeni for their contributions in inputting student and course information.
