# Graph Representation Learning for Student Sectioning: A Node2Vec-Based Approach

A Python framework for student sectioning in course timetabling using graph representation learning. The code builds weighted student co-enrollment graphs, learns low-dimensional student embeddings with Node2Vec, and clusters the embeddings into instructional sections.

**Paper:** *Graph Representation Learning for Student Sectioning: A Node2Vec-Based Approach*  
**Target journal:** Progress in Artificial Intelligence (Springer)  
**Repository:** <https://github.com/mamintoosi/Node2Vec-SSP>

## Method (short)

1. **Graph construction** — students are nodes; an edge joins students who share at least one course (after removing constant courses). Edge weight = number of shared courses.
2. **Node2Vec embeddings** — biased random walks + Skip-gram Word2Vec produce low-dimensional student vectors.
3. **Clustering** — KMeans (or other standard algorithms) partitions embeddings into sections.

## Primary configuration

Configuration is selected by a **quality–balance Pareto analysis** (Silhouette vs. section balance \(B=\min/\max\) section sizes), validated over 20 seeds.

| Role | \((p,q,d)\) | Notes |
|------|-------------|--------|
| **Primary (core tables)** | **\(p=1.0,\ q=0.5,\ d=2\)** | Pareto-selected; matched dimensionality with PCA |
| Quality-optimal (ablation) | \(p=2.0,\ q=1.0,\ d=1\) | Highest Silhouette; severely imbalanced sections |
| Neutral / DeepWalk-equivalent (ablation) | \(p=1.0,\ q=1.0,\ d=2\) | Unbiased walks |

Fixed walk/Skip-gram settings: walk length \(L=10\), walks per node \(\gamma=80\), window \(w=5\), epochs \(\epsilon=30\), \(k=2\) sections. Default seed **42**.

## Headline results (primary config, seed 42)

Silhouette scores averaged over six courses:

| Method | Silhouette ↑ | DBI ↓ | CH ↑ |
|--------|-------------:|------:|-----:|
| BoW + KMeans | 0.157 | — | — |
| PCA + KMeans | 0.466 | — | — |
| Spectral | 0.128 | — | — |
| **Node2Vec (\(p{=}1.0,q{=}0.5,d{=}2\))** | **0.619** | — | — |
| Node2Vec (\(p{=}1.0,q{=}1.0,d{=}2\), ablation) | 0.629 | — | — |

- About **33%** relative Silhouette improvement over PCA (strongest baseline) at matched \(d=2\).
- Primary config wins on all six courses at the reference seed (Course 2 margin over PCA is small).
- Section balance for the primary config averages **0.652**, higher than all baselines (~0.47).
- KMeans ARI on fixed primary embeddings averages **0.994** over 20 initializations.
- Runtime: **&lt; 9 s** per course (Word2Vec dominates).

See the paper for DBI/CH tables, Pareto multi-seed validation, sensitivity, and ablations.

## Repository structure

```
Node2Vec-SSP/
├── data/                     # Enrollment matrices for six courses (1.txt … 6.txt)
├── experiments/              # Pipeline and experiment runners
│   ├── config.py             # Paths and default parameters
│   ├── shared.py / core.py   # Graph construction, walks, Word2Vec
│   ├── evaluation.py         # Clustering and metrics
│   ├── run_all.py            # Full reproduction (seed=42)
│   └── …
├── paper/                    # Manuscript (sn-article.tex), figures, cover letter
├── results/
│   ├── reproduced/           # Baseline + sensitivity + stability artifacts
│   ├── grid_search/          # (p,q,d) grid, labels, embeddings, heatmaps
│   ├── final_d1/ final_d2/   # Dimension-specific experiment outputs
│   └── final_pareto_validation/  # 20-seed Pareto validation for primary config
├── scripts/                  # Table fillers, Pareto helpers, figure regeneration
├── doc/                      # Audit / research notes
└── README.md
```

## Installation

```bash
git clone https://github.com/mamintoosi/Node2Vec-SSP.git
cd Node2Vec-SSP
pip install numpy scipy networkx gensim scikit-learn matplotlib seaborn pandas
```

Python 3.10+ recommended.

## Reproducing results

Full pipeline (methods, sensitivity, stability, figures; ~25–30 minutes):

```bash
python experiments/run_all.py 2>&1 | tee results/reproduced/run.log
```

Useful entry points (see `run_*.sh` at the repo root):

| Script / command | Purpose |
|------------------|---------|
| `experiments/run_all.py` | Baselines, sensitivity, stability, figures → `results/reproduced/` |
| `experiments/grid_search_final.py` | Full \((p,q,d)\) grid (expensive) |
| `scripts/regen_grid_heatmap_no_colorbar.py` | Rebuild grid-search heatmap from saved JSON |
| `scripts/fill_section_balance_table.py` | Recompute section balance from saved labels |
| `scripts/pareto_final_comparison.py` | Aggregate 20-seed Pareto validation |

All numerical claims in the paper are tied to artifacts under `results/` (SOURCE comments in `paper/sn-article.tex`).

## Citation

```bibtex
@article{amintoosi2026n2vssp,
  title   = {Graph Representation Learning for Student Sectioning: A Node2Vec-Based Approach},
  author  = {Amintoosi, Mahmood},
  year    = {2026},
  journal = {Progress in Artificial Intelligence},
  note    = {Under review},
  url     = {https://github.com/mamintoosi/Node2Vec-SSP}
}
```

## License

See [LICENSE](LICENSE) (MIT).

## Acknowledgments

The author thanks Mr. Hasan Fahimian and Mrs. Samira Shahraeeni for their contributions in inputting student and course information.
