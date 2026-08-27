# -*- coding: utf-8 -*-
"""
Experiment Configuration for DeepWalk-SSP
==========================================
Central configuration for all experiments.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Dataset Configuration ──────────────────────────────────────────────────────
FILE_INDICES = [1, 2, 3, 4, 5, 6]
FILE_PATHS = [os.path.join(DATA_DIR, f"{i}.txt") for i in FILE_INDICES]

# Student counts per course (from the data files)
STUDENT_COUNTS = {
    1: 74,  # Course 1: 74 students, 35 courses
    2: 51,  # Course 2: 51 students, 34 courses
    3: 48,  # Course 3: 48 students
    4: 49,  # Course 4: 49 students
    5: 52,  # Course 5: 52 students
    6: 67,  # Course 6: 67 students
}

# ── Default DeepWalk Parameters ───────────────────────────────────────────────
DEFAULT_PARAMS = {
    "vector_size": 2,       # d: embedding dimension (optimal per paper)
    "walk_length": 10,      # t: random walk length
    "num_walks": 80,        # γ: number of walks per node
    "window": 5,            # w: context window size
    "epochs": 30,           # ε: Word2Vec training epochs
    "hs": 1,                # hierarchical softmax
    "sg": 1,                # skip-gram architecture
    "workers": 1,           # number of worker threads (1 for reproducibility)
    "n_clusters": 2,        # number of clusters for sectioning
}

# ── Default Node2Vec Parameters ──────────────────────────────────────────────
NODE2VEC_PARAMS = {
    "p": 1.0,              # Return parameter (same as DeepWalk when p=1)
    "q": 1.0,              # In-out parameter (same as DeepWalk when q=1)
    "vector_size": 2,       # Same embedding dimension as DeepWalk
    "walk_length": 10,      # Same walk length as DeepWalk
    "num_walks": 80,        # Same number of walks as DeepWalk
    "window": 5,            # Same window size as DeepWalk
    "epochs": 30,           # Same training epochs as DeepWalk
    "n_clusters": 2,        # Same number of clusters as DeepWalk
}

# Node2Vec parameter settings to compare with DeepWalk
# q < 1 favors BFS-like exploration (local structure)
# q > 1 favors DFS-like exploration (global structure)
NODE2VEC_Q_VALUES = [0.5, 1.0, 2.0, 4.0]

# ── Parameter Ranges for Sensitivity Analysis ─────────────────────────────────
VECTOR_SIZES = [1, 2, 3, 5, 10]
WALK_LENGTHS = [5, 10, 20, 40, 80]
NUM_WALKS_RANGE = [10, 20, 40, 80, 160]
WINDOW_SIZES = [1, 2, 5, 10, 20]
EPOCHS_RANGE = [10, 20, 30, 50, 100]

# ── Random Seeds for Stability Analysis ───────────────────────────────────────
NUM_SEEDS = 20
SEED_RANGE = list(range(NUM_SEEDS))  # Seeds 0..19

# ── Clustering Algorithms ─────────────────────────────────────────────────────
CLUSTERING_METHODS = ["kmeans", "affinity", "gmm", "agglomerative"]

# ── Plotting Configuration ─────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

PUBLICATION_STYLE = {
    "figure.figsize": (7, 5),
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

# Color palette for consistent plotting
COLORS = {
    "deepwalk": "#2196F3",      # Blue
    "node2vec": "#4CAF50",      # Green
    "bow": "#FF9800",           # Orange
    "pca_kmeans": "#795548",    # Brown
    "spectral": "#607D8B",      # Blue-grey
    "deepwalk_alpha": "#BBDEFB",
    "node2vec_alpha": "#C8E6C9",
    "bow_alpha": "#FFE0B2",
    "kmeans": "#E91E63",
    "affinity": "#9C27B0",
    "gmm": "#00BCD4",
    "agglomerative": "#4CAF50",
}
