# Revisiting Student Sectioning via Node2Vec: A Graph Representation Learning Framework

A Python framework for solving the student sectioning problem in course timetabling using graph representation learning. The framework constructs student co-enrollment graphs and applies Node2Vec to learn low-dimensional student representations that encode higher-order proximity patterns, producing substantially better clustering inputs than traditional binary enrollment matrices.

**Paper:** "Revisiting the Student Sectioning Problem through Graph Representation Learning" 

## Key Results

| Metric | Traditional (BoW) | PCA+KMeans | Node2Vec (p=1,q=1) | Improvement vs BoW |
|--------|-------------------|------------|--------------|-------------------|
| Silhouette Score ↑ | 0.153 | 0.460 | 0.611 | **+299%** |
| Wilcoxon *p*-value (vs BoW) | — | — | — | 0.031 (raw) |
| Wilcoxon *p*-value (vs PCA) | — | — | — | 0.031 (raw) |
| Cliff's δ (vs BoW) | — | — | — | 1.000 |
| Cliff's δ (vs PCA) | — | — | — | 0.833 |

- Consistent improvement across 6 courses, 4 clustering algorithms, and 3 evaluation metrics
- Statistical significance validated with Wilcoxon signed-rank tests
- Node2Vec outperforms PCA+KMeans (Cliff's δ = 0.833)
- Clustering stability: ARI ≥ 0.917 for KMeans across 20 random seeds
- Runtime: < 9 seconds per course

## Repository Structure

```
Deepwalk-SSP/
├── data/                          # Student enrollment data (6 courses)
│   ├── 1.txt                      # Course 1: 74 students, 35 courses
│   ├── 2.txt                      # Course 2: 51 students, 34 courses
│   ├── 3.txt - 6.txt             # Courses 3-6
├── experiments/                    # Experiment modules
│   ├── config.py                  # Central configuration (parameters, paths)
│   ├── core.py                    # Pipeline: graph construction, walks, Word2Vec
│   ├── evaluation.py              # Clustering algorithms and metrics
│   ├── plotting.py                # Publication-quality figure generation
│   ├── stats.py                   # Wilcoxon tests, effect sizes, bootstrap CI
│   └── run_experiments.py         # Complete experiment runner (all steps)
├── paper/                         # LaTeX source of the paper
│   ├── sn-article.tex            # LaTeX source (Springer template)
│   ├── sn-jnl.cls                # Springer document class
│   ├── *.png, *.pdf              # Figures referenced in the paper
│   └── MyReferences.bib
├── results/                       # Generated results and figures
│   ├── exp_*.json                 # Experiment data (JSON)
│   ├── exp_*.xlsx                 # Experiment data (Excel)
│   ├── figures/                   # All generated figures (PNG)
│   ├── df.xlsx                    # Original results
│   └── graphs.pkl                 # Original graph objects
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

Run the complete experiment suite (takes ~15-20 minutes depending on hardware):

```bash
python -m experiments.run_experiments
```

This executes all steps sequentially:

| Step | Description | Output |
|------|-------------|--------|
| 2 | Reproduce existing experiments | `df_reproduced.xlsx`, silhouette figures |
| 4A | Hyperparameter sensitivity (d, t, γ, w) | `exp_A_hyperparameter_sensitivity.json` |
| 4B | Runtime analysis | `exp_B_runtime.json` |
| 4C | Random seed stability (20 seeds) | `exp_C_seed_stability.json` |
| 4D | Clustering stability (ARI) | `exp_D_clustering_stability.json` |
| 4E | Additional metrics (DBI, CH, WCSS) | `exp_E_additional_metrics.xlsx` |
| 4F | Embedding visualization (PCA + t-SNE) | `results/figures/exp_F_*.png` |
| 5 | Statistical significance (Wilcoxon tests) | `exp_statistical_analysis.json` |
| 7 | Practical improvements (cosine graph, etc.) | `exp_F_improvements.xlsx` |

All results will be saved to `results/` and `results/figures/`.

### Reproducibility

All experiments use fixed random seeds (`seed=0` by default) for full reproducibility. Two independent runs produce identical results. The seed stability experiments (Steps 4C-4D) use 20 different seeds (0-19) to report mean ± standard deviation.

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
  title={Revisiting Student Sectioning via Node2Vec: A Graph Representation Learning Framework},
  author={Amintoosi, Mahmood},
  year={2026},
  publisher={Submitted},
  url={https://github.com/mamintoosi/Node2Vec-SSP}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

The author would like to thank Mr. Hasan Fahimian and Mrs. Samira Shahraeeni for their contributions in inputting student and course information.
