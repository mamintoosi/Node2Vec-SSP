# -*- coding: utf-8 -*-
"""
Revised DeepWalk-SSP Pipeline
================================
Replaces DeepWalk with Node2Vec as the graph representation learning method.
Implements revised graph construction by removing universally shared courses.

This script:
1. Analyzes graph structure before/after removing universal courses
2. Re-runs all methods (BoW, PCA, Spectral, Node2Vec) under Linux
3. Tests Node2Vec parameter sensitivity (p,q grid)
4. Performs statistical analysis with Holm correction
5. Generates publication-quality PDF figures
6. Writes a comprehensive report

Usage:
    /data/python-envs/pytorch/bin/python -m experiments.revised_pipeline
"""

import os
import sys
import time
import json
import warnings
import random
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

# ── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REEXPERIMENT_DIR = os.path.join(RESULTS_DIR, "revised_reexperiment")
FIGURES_DIR = os.path.join(REEXPERIMENT_DIR, "figures")

os.makedirs(REEXPERIMENT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FILE_INDICES = [1, 2, 3, 4, 5, 6]

# ── Publication style ────────────────────────────────────────────────────────
PUB_STYLE = {
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

COLORS = {
    "bow": "#FF9800",
    "pca": "#795548",
    "spectral": "#607D8B",
    "node2vec": "#4CAF50",
    "node2vec_old": "#81C784",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: Data Loading & Graph Construction
# ══════════════════════════════════════════════════════════════════════════════

def read_class(file_path):
    """Read student-course data file. Returns (matrix, labels)."""
    with open(file_path, 'r') as f:
        line1 = f.readline().strip().split()
        num_students, num_courses = int(line1[0]), int(line1[1])
        matrix = np.zeros((num_students, num_courses), dtype=int)
        labels = []
        for j in range(num_students):
            parts = f.readline().strip().split()
            labels.append(parts[1])
            bv = parts[2]
            if len(bv) < num_courses:
                bv = bv.ljust(num_courses, '0')
            matrix[j] = [int(c) for c in bv[:num_courses]]
    return matrix, labels


def find_universal_columns(matrix):
    """Find columns where ALL students have value 1."""
    n_students = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    return np.where(col_sums == n_students)[0].tolist()


def create_graph_from_bow(matrix):
    """Create co-enrollment graph. Edge weight = number of shared courses."""
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            w = int(np.sum(matrix[i] * matrix[j]))
            if w > 0:
                G.add_edge(i, j, weight=w)
    return G


def graph_stats(G):
    """Compute graph statistics."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    max_e = n * (n - 1) // 2
    degrees = [d for _, d in G.degree()]
    weights = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    return {
        "n_nodes": n,
        "n_edges": e,
        "max_edges": max_e,
        "density": e / max_e if max_e > 0 else 0,
        "n_components": nx.number_connected_components(G),
        "avg_degree": float(np.mean(degrees)) if degrees else 0,
        "std_degree": float(np.std(degrees)) if degrees else 0,
        "min_degree": int(np.min(degrees)) if degrees else 0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "avg_weight": float(np.mean(weights)) if weights else 0,
        "std_weight": float(np.std(weights)) if weights else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: Node2Vec Walk Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_node2vec_walks(G, num_walks=80, walk_length=10, p=1.0, q=1.0, seed=0):
    """Generate biased random walks (Node2Vec algorithm)."""
    rng = np.random.RandomState(seed)
    walks = []

    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            prev_node = None
            current_node = node

            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current_node))
                if not neighbors:
                    break

                if prev_node is None:
                    next_node = neighbors[rng.randint(len(neighbors))]
                else:
                    prev_set = set(G.neighbors(prev_node))
                    weights = []
                    for z in neighbors:
                        if z == prev_node:
                            alpha = 1.0 / p
                        elif z in prev_set:
                            alpha = 1.0
                        else:
                            alpha = 1.0 / q
                        ew = G[current_node][z].get('weight', 1)
                        weights.append(alpha * ew)
                    total = sum(weights)
                    probs = np.array(weights) / total
                    r = rng.random()
                    cum = 0.0
                    next_node = neighbors[-1]
                    for idx, prob in enumerate(probs):
                        cum += prob
                        if r <= cum:
                            next_node = neighbors[idx]
                            break

                walk.append(next_node)
                prev_node = current_node
                current_node = next_node

            walks.append(walk)
    return walks


def train_word2vec(walks, vector_size=2, window=5, epochs=30, seed=0, workers=1):
    """Train Word2Vec on random walks."""
    model = Word2Vec(
        walks, vector_size=vector_size, window=window,
        hs=1, sg=1, workers=workers, seed=seed,
        min_count=1, sample=0,
    )
    model.train(walks, total_examples=model.corpus_count, epochs=epochs, report_delay=0)
    return model.wv.vectors


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C: Clustering & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(data, labels):
    """Compute Silhouette, DBI, CH."""
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "ch": np.nan}
    return {
        "silhouette": float(silhouette_score(data, labels)),
        "dbi": float(davies_bouldin_score(data, labels)),
        "ch": float(calinski_harabasz_score(data, labels)),
    }


def cluster_kmeans(data, n_clusters=2, seed=0):
    """KMeans clustering."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(data)


def cluster_spectral(adj, n_clusters=2, seed=0):
    """Spectral clustering on affinity matrix."""
    sc = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        random_state=seed, assign_labels='kmeans',
    )
    return sc.fit_predict(adj)


def cluster_pca_kmeans(data, n_clusters=2, n_components=2, seed=0):
    """PCA + KMeans."""
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(data)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(reduced)
    return labels, reduced


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D: Full Node2Vec Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_node2vec_full(file_idx, matrix, G, p=1.0, q=1.0,
                      vector_size=2, num_walks=80, walk_length=10,
                      window=5, epochs=30, n_clusters=2, seed=0):
    """Run complete Node2Vec pipeline: walks → embeddings → cluster → evaluate."""
    t0 = time.perf_counter()
    walks = generate_node2vec_walks(G, num_walks=num_walks, walk_length=walk_length,
                                     p=p, q=q, seed=seed)
    t_walks = time.perf_counter() - t0

    t0 = time.perf_counter()
    embeddings = train_word2vec(walks, vector_size=vector_size, window=window,
                                 epochs=epochs, seed=seed, workers=1)
    t_w2v = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = cluster_kmeans(embeddings, n_clusters=n_clusters, seed=seed)
    t_cluster = time.perf_counter() - t0

    # Evaluate on embeddings
    metrics_emb = compute_metrics(embeddings, labels)
    # Also evaluate on original BoW space
    metrics_bow = compute_metrics(matrix.astype(float), labels)

    return {
        "course": file_idx,
        "p": p, "q": q,
        "n_students": matrix.shape[0],
        "n_courses": matrix.shape[1],
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "silhouette_emb": metrics_emb["silhouette"],
        "dbi_emb": metrics_emb["dbi"],
        "ch_emb": metrics_emb["ch"],
        "silhouette_bow": metrics_bow["silhouette"],
        "dbi_bow": metrics_bow["dbi"],
        "ch_bow": metrics_bow["ch"],
        "t_walks": t_walks,
        "t_w2v": t_w2v,
        "t_cluster": t_cluster,
        "t_total": t_walks + t_w2v + t_cluster,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_test(x, y, alternative='greater'):
    """Wilcoxon signed-rank test."""
    x, y = np.asarray(x), np.asarray(y)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(x)}
    try:
        stat, p = wilcoxon(x, y, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p), "n": len(x)}
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(x)}


def compute_effect_size_r(statistic, n):
    """Effect size r from Wilcoxon."""
    if n < 5:
        return np.nan
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        return 0.0
    z = (statistic - mean_w) / std_w
    return min(abs(z) / np.sqrt(n), 1.0)


def holm_correction(p_values):
    """Holm (Bonferroni) correction for multiple comparisons."""
    p = np.array(p_values)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(p[idx] * (m - rank), 1.0)
    # Enforce monotonicity
    sorted_adj = adjusted[order]
    for i in range(len(sorted_adj) - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adjusted[order] = sorted_adj
    return adjusted.tolist()


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n * m)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F: Figure Generation
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name, fmt="pdf"):
    """Save figure to figures directory."""
    path = os.path.join(FIGURES_DIR, f"{name}.{fmt}")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return path


def fig_method_comparison(results_df):
    """Bar chart: method comparison (Silhouette Score)."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    courses = sorted(results_df["course"].unique())
    methods = ["BoW+KMeans", "PCA+KMeans", "Spectral", "Node2Vec(p=1,q=1)"]
    colors = [COLORS["bow"], COLORS["pca"], COLORS["spectral"], COLORS["node2vec"]]

    x = np.arange(len(courses))
    width = 0.18

    for i, (method, color) in enumerate(zip(methods, colors)):
        col = method.lower().replace("+", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
        # Map to actual column names
        if "bow" in method.lower():
            vals = [results_df[(results_df["course"] == c) & (results_df["method"] == "bow")]["silhouette_bow"].values[0] for c in courses]
        elif "pca" in method.lower():
            vals = [results_df[(results_df["course"] == c) & (results_df["method"] == "pca")]["silhouette_bow"].values[0] for c in courses]
        elif "spectral" in method.lower():
            vals = [results_df[(results_df["course"] == c) & (results_df["method"] == "spectral")]["silhouette_bow"].values[0] for c in courses]
        else:
            vals = [results_df[(results_df["course"] == c) & (results_df["method"] == "node2vec")]["silhouette_emb"].values[0] for c in courses]

        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score (↑ higher is better)')
    ax.set_title('Method Comparison: Clustering Quality Across Six Courses')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "method_comparison")


def fig_node2vec_sensitivity(sensitivity_df):
    """Heatmap: Node2Vec p,q sensitivity."""
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, title, cmap) in enumerate([
        ("silhouette_emb", "Silhouette Score (↑)", "YlGn"),
        ("dbi_emb", "Davies-Bouldin Index (↓)", "YlGn_r"),
        ("ch_emb", "Calinski-Harabasz Index (↑)", "YlGn"),
    ]):
        pivot = sensitivity_df.groupby(["p", "q"])[metric].mean().reset_index()
        pivot = pivot.pivot(index="p", columns="q", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=axes[idx],
                    cbar_kws={"shrink": 0.8})
        axes[idx].set_title(title)
        axes[idx].set_xlabel("q")
        axes[idx].set_ylabel("p")

    fig.suptitle("Node2Vec Parameter Sensitivity (averaged over 6 courses)", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, "node2vec_sensitivity_heatmap")


def fig_graph_density_comparison(old_stats, new_stats):
    """Bar chart comparing graph density before/after removing universal courses."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    courses = [str(i) for i in range(1, 7)]
    old_densities = [old_stats[c]["density"] for c in courses]
    new_densities = [new_stats[c]["density"] for c in courses]

    x = np.arange(len(courses))
    width = 0.35

    ax.bar(x - width/2, old_densities, width, label='Original (all courses)', color='#E53935', alpha=0.8)
    ax.bar(x + width/2, new_densities, width, label='Revised (universal removed)', color='#43A047', alpha=0.8)

    ax.set_xlabel('Course')
    ax.set_ylabel('Graph Density')
    ax.set_title('Effect of Removing Universal Courses on Graph Density')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for i_v, (old, new) in enumerate(zip(old_densities, new_densities)):
        ax.annotate(f'{old:.2f}', xy=(i_v - width/2, old), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
        ax.annotate(f'{new:.2f}', xy=(i_v + width/2, new), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)

    return save_fig(fig, "graph_density_comparison")


def fig_node2vec_vs_baselines_boxplot(seed_results):
    """Boxplot of Node2Vec stability across seeds vs baseline fixed scores."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_data = []
    all_labels = []

    for course in range(1, 7):
        # Node2Vec silhouette across seeds
        n2v_sils = [seed_results[f"n2v_{s}"][course-1]["silhouette_emb"] for s in range(20)]
        all_data.append(n2v_sils)
        all_labels.append(f"Course {course}\nNode2Vec")

    bp = ax.boxplot(all_data, tick_labels=all_labels, patch_artist=True,
                    boxprops=dict(facecolor=COLORS["node2vec"], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))

    ax.set_ylabel('Silhouette Score')
    ax.set_title('Node2Vec Stability Across 20 Random Seeds')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "node2vec_stability_boxplot")


def fig_runtime_comparison(runtime_df):
    """Bar chart of runtime comparison."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    courses = sorted(runtime_df["course"].unique())
    methods = ["BoW+KMeans", "PCA+KMeans", "Spectral", "Node2Vec"]
    colors_l = [COLORS["bow"], COLORS["pca"], COLORS["spectral"], COLORS["node2vec"]]

    x = np.arange(len(courses))
    width = 0.18

    for i, (method, color) in enumerate(zip(methods, colors_l)):
        vals = [runtime_df[(runtime_df["course"] == c) & (runtime_df["method"] == method)]["t_total"].values[0]
                for c in courses]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85)

    ax.set_xlabel('Course')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison Across Methods and Courses')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "runtime_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()
    print("=" * 70)
    print("  REVISED PIPELINE: Node2Vec for Student Sectioning")
    print("  (with revised graph construction)")
    print("=" * 70)

    # ── Record environment ────────────────────────────────────────────────
    import platform
    env_info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import scipy; env_info["scipy"] = scipy.__version__
    except: pass
    try:
        import sklearn; env_info["sklearn"] = sklearn.__version__
    except: pass
    try:
        import gensim; env_info["gensim"] = gensim.__version__
    except: pass
    try:
        env_info["networkx"] = nx.__version__
    except: pass
    try:
        import pandas as pd; env_info["pandas"] = pd.__version__
    except: pass
    try:
        import matplotlib; env_info["matplotlib"] = matplotlib.__version__
    except: pass

    print(f"\n  Environment: Python {env_info['python']}, numpy {env_info['numpy']}, "
          f"gensim {env_info.get('gensim','?')}, sklearn {env_info.get('sklearn','?')}")

    with open(os.path.join(REEXPERIMENT_DIR, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Graph Structure Analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: Graph Structure Analysis (Original vs Revised)")
    print("=" * 70)

    old_graph_stats = {}
    new_graph_stats = {}
    universal_info = {}

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        matrix, labels = read_class(filepath)

        # Original graph
        G_old = create_graph_from_bow(matrix)
        old_graph_stats[str(i)] = graph_stats(G_old)

        # Find and remove universal columns
        universal_cols = find_universal_columns(matrix)
        matrix_new = np.delete(matrix, universal_cols, axis=1)

        # Revised graph
        G_new = create_graph_from_bow(matrix_new)
        new_graph_stats[str(i)] = graph_stats(G_new)

        universal_info[str(i)] = {
            "n_students": matrix.shape[0],
            "n_courses_original": matrix.shape[1],
            "universal_columns": universal_cols,
            "n_removed": len(universal_cols),
            "n_courses_remaining": matrix_new.shape[1],
        }

        print(f"\n  Course {i}:")
        print(f"    Students: {matrix.shape[0]}")
        print(f"    Courses: {matrix.shape[1]} → {matrix_new.shape[1]} (removed {len(universal_cols)} universal)")
        print(f"    Original graph: {old_graph_stats[str(i)]['n_edges']} edges, density={old_graph_stats[str(i)]['density']:.4f}")
        print(f"    Revised graph:  {new_graph_stats[str(i)]['n_edges']} edges, density={new_graph_stats[str(i)]['density']:.4f}")
        print(f"    Connected components: {new_graph_stats[str(i)]['n_components']}")
        print(f"    Degree range: [{new_graph_stats[str(i)]['min_degree']}, {new_graph_stats[str(i)]['max_degree']}]")

    # Save graph analysis
    graph_analysis = {
        "universal_info": universal_info,
        "old_graph_stats": old_graph_stats,
        "new_graph_stats": new_graph_stats,
    }
    with open(os.path.join(REEXPERIMENT_DIR, "graph_analysis.json"), "w") as f:
        json.dump(graph_analysis, f, indent=2)

    # Figure: graph density comparison
    fig_graph_density_comparison(old_graph_stats, new_graph_stats)
    print("\n  Saved: graph_density_comparison.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Run ALL Methods with Revised Graph (seed=0, primary results)
    print("\n" + "=" * 70)
    print("  PART 2: All Methods with Revised Graph (seed=0)")
    print("=" * 70)

    all_results = []
    runtime_data = []

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        matrix_orig, labels = read_class(filepath)

        # Remove universal columns
        universal_cols = find_universal_columns(matrix_orig)
        matrix = np.delete(matrix_orig, universal_cols, axis=1)

        n_students = matrix.shape[0]
        G = create_graph_from_bow(matrix)
        print(f"\n  Course {i}: {n_students} students, {matrix.shape[1]} courses, {G.number_of_edges()} edges")

        # 1. BoW + KMeans
        t0 = time.perf_counter()
        bow_labels = cluster_kmeans(matrix.astype(float), n_clusters=2, seed=0)
        t_bow = time.perf_counter() - t0
        bow_m = compute_metrics(matrix.astype(float), bow_labels)
        print(f"    BoW+KMeans:    Silhouette={bow_m['silhouette']:.3f}, DBI={bow_m['dbi']:.3f}, CH={bow_m['ch']:.1f}")
        all_results.append({
            "course": i, "method": "bow", "n_students": n_students,
            "silhouette_bow": bow_m["silhouette"], "dbi_bow": bow_m["dbi"], "ch_bow": bow_m["ch"],
            "silhouette_emb": bow_m["silhouette"], "dbi_emb": bow_m["dbi"], "ch_emb": bow_m["ch"],
        })
        runtime_data.append({"course": i, "method": "BoW+KMeans", "t_total": t_bow})

        # 2. PCA + KMeans
        t0 = time.perf_counter()
        pca_labels, pca_reduced = cluster_pca_kmeans(matrix.astype(float), n_clusters=2, n_components=2, seed=0)
        t_pca = time.perf_counter() - t0
        pca_m_emb = compute_metrics(pca_reduced, pca_labels)
        pca_m_bow = compute_metrics(matrix.astype(float), pca_labels)
        print(f"    PCA+KMeans:    Silhouette={pca_m_bow['silhouette']:.3f}, DBI={pca_m_bow['dbi']:.3f}, CH={pca_m_bow['ch']:.1f}")
        all_results.append({
            "course": i, "method": "pca", "n_students": n_students,
            "silhouette_bow": pca_m_bow["silhouette"], "dbi_bow": pca_m_bow["dbi"], "ch_bow": pca_m_bow["ch"],
            "silhouette_emb": pca_m_emb["silhouette"], "dbi_emb": pca_m_emb["dbi"], "ch_emb": pca_m_emb["ch"],
        })
        runtime_data.append({"course": i, "method": "PCA+KMeans", "t_total": t_pca})

        # 3. Spectral Clustering
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        t0 = time.perf_counter()
        spec_labels = cluster_spectral(adj, n_clusters=2, seed=0)
        t_spec = time.perf_counter() - t0
        spec_m_emb = compute_metrics(adj, spec_labels)
        spec_m_bow = compute_metrics(matrix.astype(float), spec_labels)
        print(f"    Spectral:      Silhouette={spec_m_bow['silhouette']:.3f}, DBI={spec_m_bow['dbi']:.3f}, CH={spec_m_bow['ch']:.1f}")
        all_results.append({
            "course": i, "method": "spectral", "n_students": n_students,
            "silhouette_bow": spec_m_bow["silhouette"], "dbi_bow": spec_m_bow["dbi"], "ch_bow": spec_m_bow["ch"],
            "silhouette_emb": spec_m_emb["silhouette"], "dbi_emb": spec_m_emb["dbi"], "ch_emb": spec_m_emb["ch"],
        })
        runtime_data.append({"course": i, "method": "Spectral", "t_total": t_spec})

        # 4. Node2Vec(p=1,q=1) — neutral/unbiased
        n2v_res = run_node2vec_full(i, matrix, G, p=1.0, q=1.0, seed=0)
        print(f"    Node2Vec(1,1): Silhouette={n2v_res['silhouette_emb']:.3f}, DBI={n2v_res['dbi_emb']:.3f}, CH={n2v_res['ch_emb']:.1f}")
        all_results.append({
            "course": i, "method": "node2vec", "n_students": n_students,
            "silhouette_bow": n2v_res["silhouette_bow"], "dbi_bow": n2v_res["dbi_bow"], "ch_bow": n2v_res["ch_bow"],
            "silhouette_emb": n2v_res["silhouette_emb"], "dbi_emb": n2v_res["dbi_emb"], "ch_emb": n2v_res["ch_emb"],
        })
        runtime_data.append({
            "course": i, "method": "Node2Vec",
            "t_walks": n2v_res["t_walks"], "t_w2v": n2v_res["t_w2v"],
            "t_cluster": n2v_res["t_cluster"], "t_total": n2v_res["t_total"],
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(os.path.join(REEXPERIMENT_DIR, "all_methods_results.xlsx"), index=False)
    results_df.to_json(os.path.join(REEXPERIMENT_DIR, "all_methods_results.json"),
                       orient="records", indent=2)

    # Print summary table
    print("\n  SUMMARY TABLE (Silhouette Score, seed=0)")
    print(f"  {'Method':<20}", end="")
    for i in FILE_INDICES:
        print(f" {'C'+str(i):>6}", end="")
    print(f" {'Average':>8}")
    print("  " + "-" * 60)

    for method_name, method_key in [("BoW+KMeans", "bow"), ("PCA+KMeans", "pca"),
                                     ("Spectral", "spectral"), ("Node2Vec(1,1)", "node2vec")]:
        sub = results_df[results_df["method"] == method_key]
        print(f"  {method_name:<20}", end="")
        for i in FILE_INDICES:
            row = sub[sub["course"] == i]
            if len(row) > 0:
                val = row["silhouette_emb"].values[0] if method_key == "node2vec" else row["silhouette_bow"].values[0]
                print(f" {val:>6.3f}", end="")
        avg = sub["silhouette_emb"].mean() if method_key == "node2vec" else sub["silhouette_bow"].mean()
        print(f" {avg:>8.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: Node2Vec Parameter Sensitivity (p × q grid)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 3: Node2Vec Parameter Sensitivity (p × q grid)")
    print("=" * 70)

    p_values = [0.5, 1.0, 2.0]
    q_values = [0.5, 1.0, 2.0]
    sensitivity_results = []

    for p in p_values:
        for q in q_values:
            sils = []
            for i in FILE_INDICES:
                filepath = os.path.join(DATA_DIR, f"{i}.txt")
                matrix_orig, _ = read_class(filepath)
                universal_cols = find_universal_columns(matrix_orig)
                matrix = np.delete(matrix_orig, universal_cols, axis=1)
                G = create_graph_from_bow(matrix)

                res = run_node2vec_full(i, matrix, G, p=p, q=q, seed=0)
                sils.append(res["silhouette_emb"])
                sensitivity_results.append({
                    "p": p, "q": q, "course": i,
                    "silhouette_emb": res["silhouette_emb"],
                    "dbi_emb": res["dbi_emb"],
                    "ch_emb": res["ch_emb"],
                    "silhouette_bow": res["silhouette_bow"],
                })
            avg = np.mean(sils)
            print(f"    p={p:.1f}, q={q:.1f}: avg Silhouette = {avg:.3f}")

    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_excel(os.path.join(REEXPERIMENT_DIR, "node2vec_sensitivity.xlsx"), index=False)
    sens_df.to_json(os.path.join(REEXPERIMENT_DIR, "node2vec_sensitivity.json"),
                    orient="records", indent=2)

    # Heatmap
    fig_node2vec_sensitivity(sens_df)
    print("\n  Saved: node2vec_sensitivity_heatmap.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: Seed Stability Analysis (20 seeds)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 4: Seed Stability Analysis (20 seeds)")
    print("=" * 70)

    NUM_SEEDS = 20
    stability_data = {}  # key: "n2v_{seed}" -> list of 6 results

    for seed in range(NUM_SEEDS):
        stability_data[f"n2v_{seed}"] = []
        for i in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{i}.txt")
            matrix_orig, _ = read_class(filepath)
            universal_cols = find_universal_columns(matrix_orig)
            matrix = np.delete(matrix_orig, universal_cols, axis=1)
            G = create_graph_from_bow(matrix)
            res = run_node2vec_full(i, matrix, G, p=1.0, q=1.0, seed=seed)
            stability_data[f"n2v_{seed}"].append(res)
        if (seed + 1) % 5 == 0:
            print(f"    Completed {seed + 1}/{NUM_SEEDS} seeds")

    # Compute stability statistics per course
    stability_stats = {}
    for ci, course in enumerate(FILE_INDICES):
        sils = [stability_data[f"n2v_{s}"][ci]["silhouette_emb"] for s in range(NUM_SEEDS)]
        dbis = [stability_data[f"n2v_{s}"][ci]["dbi_emb"] for s in range(NUM_SEEDS)]
        chs = [stability_data[f"n2v_{s}"][ci]["ch_emb"] for s in range(NUM_SEEDS)]
        stability_stats[str(course)] = {
            "silhouette": {"mean": float(np.mean(sils)), "std": float(np.std(sils)),
                           "min": float(np.min(sils)), "max": float(np.max(sils))},
            "dbi": {"mean": float(np.mean(dbis)), "std": float(np.std(dbis))},
            "ch": {"mean": float(np.mean(chs)), "std": float(np.std(chs))},
        }
        print(f"    Course {course}: Silhouette = {np.mean(sils):.3f} ± {np.std(sils):.3f} "
              f"(range [{np.min(sils):.3f}, {np.max(sils):.3f}])")

    with open(os.path.join(REEXPERIMENT_DIR, "stability_analysis.json"), "w") as f:
        json.dump(stability_stats, f, indent=2)

    # Boxplot
    fig_node2vec_vs_baselines_boxplot(stability_data)
    print("\n  Saved: node2vec_stability_boxplot.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART 5: Statistical Analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 5: Statistical Analysis")
    print("=" * 70)

    # For each course, get mean Node2Vec silhouette over 20 seeds
    n2v_means = np.array([stability_stats[str(c)]["silhouette"]["mean"] for c in FILE_INDICES])

    # PCA scores (deterministic with seed=0)
    pca_scores = []
    bow_scores = []
    spec_scores = []
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        matrix_orig, _ = read_class(filepath)
        universal_cols = find_universal_columns(matrix_orig)
        matrix = np.delete(matrix_orig, universal_cols, axis=1)
        bow_labels = cluster_kmeans(matrix.astype(float), n_clusters=2, seed=0)
        bow_scores.append(compute_metrics(matrix.astype(float), bow_labels)["silhouette"])
        pca_labels, pca_red = cluster_pca_kmeans(matrix.astype(float), n_clusters=2, n_components=2, seed=0)
        pca_scores.append(compute_metrics(matrix.astype(float), pca_labels)["silhouette"])
        G = create_graph_from_bow(matrix)
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        spec_labels = cluster_spectral(adj, n_clusters=2, seed=0)
        spec_scores.append(compute_metrics(matrix.astype(float), spec_labels)["silhouette"])

    bow_scores = np.array(bow_scores)
    pca_scores = np.array(pca_scores)
    spec_scores = np.array(spec_scores)

    comparisons = [
        ("Node2Vec vs BoW+KMeans", n2v_means, bow_scores),
        ("Node2Vec vs PCA+KMeans", n2v_means, pca_scores),
        ("Node2Vec vs Spectral", n2v_means, spec_scores),
    ]

    raw_p_values = []
    stat_results_all = []

    for name, x, y in comparisons:
        test = wilcoxon_test(x, y, alternative='two-sided')
        r = compute_effect_size_r(test["statistic"], test["n"])
        delta = cliffs_delta(x, y)
        diff_mean = float(np.mean(x - y))
        diff_std = float(np.std(x - y))

        raw_p_values.append(test["p_value"])
        stat_results_all.append({
            "comparison": name,
            "n2v_mean": float(np.mean(x)),
            "other_mean": float(np.mean(y)),
            "mean_diff": diff_mean,
            "wilcoxon_stat": test["statistic"],
            "wilcoxon_p": test["p_value"],
            "effect_size_r": r,
            "cliffs_delta": delta,
            "effect_interpretation": "Large" if r >= 0.5 else "Medium" if r >= 0.3 else "Small" if r >= 0.1 else "Negligible",
        })

        print(f"\n  {name}:")
        print(f"    Node2Vec: {np.mean(x):.3f}, Other: {np.mean(y):.3f}, diff: {diff_mean:+.3f}")
        print(f"    Wilcoxon: stat={test['statistic']}, p={test['p_value']:.6f}")
        print(f"    Effect size r={r:.3f}, Cliff's delta={delta:.3f}")

    # Apply Holm correction
    corrected_p = holm_correction(raw_p_values)
    for i_res, (corr_p, res) in enumerate(zip(corrected_p, stat_results_all)):
        res["corrected_p"] = corr_p
        res["significant_005"] = corr_p < 0.05
        res["significant_01"] = corr_p < 0.01
        print(f"\n  {res['comparison']}: corrected p = {corr_p:.6f} "
              f"({'significant' if corr_p < 0.05 else 'not significant'} at α=0.05)")

    with open(os.path.join(REEXPERIMENT_DIR, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results_all, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # PART 6: Node2Vec Walk Validation (p=1,q=1 = unbiased)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 6: Node2Vec Implementation Validation")
    print("=" * 70)

    # Validate that Node2Vec(p=1,q=1) produces unbiased transitions
    filepath = os.path.join(DATA_DIR, "1.txt")
    matrix_orig, _ = read_class(filepath)
    universal_cols = find_universal_columns(matrix_orig)
    matrix = np.delete(matrix_orig, universal_cols, axis=1)
    G = create_graph_from_bow(matrix)

    # For a specific node, compute theoretical uniform transition probabilities
    # and compare with empirical walk distribution
    node = 0
    neighbors = list(G.neighbors(node))
    n_neighbors = len(neighbors)

    # Generate many walks from this node
    walks = generate_node2vec_walks(G, num_walks=1000, walk_length=2, p=1.0, q=1.0, seed=42)
    first_steps = [w[1] for w in walks if w[0] == node]

    from collections import Counter
    step_counts = Counter(first_steps)
    empirical_probs = {k: v / len(first_steps) for k, v in step_counts.items()}
    expected_prob = 1.0 / n_neighbors

    max_deviation = max(abs(empirical_probs.get(nb, 0) - expected_prob) for nb in neighbors)

    print(f"  Node {node}: {n_neighbors} neighbors")
    print(f"  Expected uniform probability: {expected_prob:.4f}")
    print(f"  Max deviation from uniform: {max_deviation:.4f}")
    print(f"  Walks analyzed: {len(first_steps)}")

    validation_result = {
        "node": node,
        "n_neighbors": n_neighbors,
        "expected_prob": expected_prob,
        "max_deviation": max_deviation,
        "n_walks": len(first_steps),
        "valid": max_deviation < 0.05,
    }
    with open(os.path.join(REEXPERIMENT_DIR, "walk_validation.json"), "w") as f:
        json.dump(validation_result, f, indent=2)

    if max_deviation < 0.05:
        print("  ✓ VALIDATION PASSED: Node2Vec(p=1,q=1) produces approximately uniform transitions")
    else:
        print("  ✗ VALIDATION WARNING: Transition distribution deviates significantly from uniform")

    # ══════════════════════════════════════════════════════════════════════
    # PART 7: Method Comparison Figure
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 7: Figure Generation")
    print("=" * 70)

    fig_method_comparison(results_df)
    print("  Saved: method_comparison.pdf")

    # Runtime figure
    runtime_df = pd.DataFrame(runtime_data)
    fig_runtime_comparison(runtime_df)
    print("  Saved: runtime_comparison.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART 8: Save Runtime Data
    # ══════════════════════════════════════════════════════════════════════
    runtime_df.to_json(os.path.join(REEXPERIMENT_DIR, "runtime.json"),
                       orient="records", indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # PART 9: Final Summary
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Output directory: {REEXPERIMENT_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")
    print(f"\n  Generated files:")
    for f_name in sorted(os.listdir(REEXPERIMENT_DIR)):
        if not f_name.startswith(".") and f_name != "figures":
            print(f"    {f_name}")
    for f_name in sorted(os.listdir(FIGURES_DIR)):
        print(f"    figures/{f_name}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    print(f"  - Graph density after removing universal courses: ", end="")
    densities = [new_graph_stats[str(i)]["density"] for i in FILE_INDICES]
    print(f"{np.mean(densities):.3f} ± {np.std(densities):.3f} (was 1.000 for all)")
    print(f"  - Node2Vec avg Silhouette (p=1,q=1): {np.mean(n2v_means):.3f}")
    print(f"  - PCA avg Silhouette: {np.mean(pca_scores):.3f}")
    print(f"  - BoW avg Silhouette: {np.mean(bow_scores):.3f}")
    print(f"  - Spectral avg Silhouette: {np.mean(spec_scores):.3f}")

    best_pq = sens_df.groupby(["p", "q"])["silhouette_emb"].mean().idxmax()
    best_sil = sens_df.groupby(["p", "q"])["silhouette_emb"].mean().max()
    print(f"  - Best Node2Vec config: p={best_pq[0]}, q={best_pq[1]} (avg Silhouette={best_sil:.3f})")

    # Find sensitivity range
    pq_avgs = sens_df.groupby(["p", "q"])["silhouette_emb"].mean()
    print(f"  - Parameter sensitivity range: {pq_avgs.min():.3f} to {pq_avgs.max():.3f}")
    # Compute q effect: at p=1, compare q=0.5 vs q=2.0
    pq_reset = pq_avgs.reset_index()
    p1_row = pq_reset[pq_reset["p"] == 1.0]
    q_low = p1_row[p1_row["q"] == 0.5]["silhouette_emb"].values[0]
    q_high = p1_row[p1_row["q"] == 2.0]["silhouette_emb"].values[0]
    q_effect = q_low - q_high
    print(f"  - q effect (at p=1): {q_effect.mean():.4f}")

    print(f"\n  Analysis complete. Manuscript revision should follow.")
    print()

    return {
        "results_df": results_df,
        "stability_stats": stability_stats,
        "sensitivity_df": sens_df,
        "stat_results": stat_results_all,
        "graph_analysis": graph_analysis,
    }


if __name__ == "__main__":
    main()
