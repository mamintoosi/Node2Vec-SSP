# -*- coding: utf-8 -*-
"""
Full Re-Experiment: Node2Vec + DeepWalk + Baselines
=====================================================
Consistent preprocessing: remove constant (universal) courses before
all methods. Re-run everything under Linux with controlled seeds.

All methods use the same filtered student-course data and the same graph.

Usage:
    /data/python-envs/pytorch/bin/python -m experiments.full_reexperiment
"""

import os, sys, time, json, warnings, random, platform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "final_reexperiment")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FILE_INDICES = [1, 2, 3, 4, 5, 6]
NUM_SEEDS = 20

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {
    "bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B",
    "deepwalk": "#2196F3", "node2vec": "#4CAF50",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data & Graph Construction
# ══════════════════════════════════════════════════════════════════════════════
def read_class(file_path):
    with open(file_path, 'r') as f:
        line1 = f.readline().strip().split()
        n, m = int(line1[0]), int(line1[1])
        matrix = np.zeros((n, m), dtype=int)
        labels = []
        for j in range(n):
            parts = f.readline().strip().split()
            labels.append(parts[1])
            bv = parts[2]
            if len(bv) < m:
                bv = bv.ljust(m, '0')
            matrix[j] = [int(c) for c in bv[:m]]
    return matrix, labels


def find_universal_columns(matrix):
    """Find columns where ALL students have value 1."""
    col_sums = matrix.sum(axis=0)
    return np.where(col_sums == matrix.shape[0])[0].tolist()


def create_graph(matrix):
    """Weighted co-enrollment graph: edge weight = shared courses."""
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
    n, e = G.number_of_nodes(), G.number_of_edges()
    max_e = n * (n - 1) // 2
    degs = [d for _, d in G.degree()]
    ws = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    return {
        "n_nodes": n, "n_edges": e, "max_edges": max_e,
        "density": e / max_e if max_e > 0 else 0,
        "n_components": nx.number_connected_components(G),
        "is_complete": e == max_e,
        "avg_degree": float(np.mean(degs)) if degs else 0,
        "std_degree": float(np.std(degs)) if degs else 0,
        "min_degree": int(np.min(degs)) if degs else 0,
        "max_degree": int(np.max(degs)) if degs else 0,
        "avg_weight": float(np.mean(ws)) if ws else 0,
        "std_weight": float(np.std(ws)) if ws else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Random Walk Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate_unbiased_walks(G, num_walks=80, walk_length=10, seed=0):
    """DeepWalk: uniform random transitions."""
    rng = np.random.RandomState(seed)
    walks = []
    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            current = node
            for _ in range(walk_length - 1):
                nbs = list(G.neighbors(current))
                if not nbs:
                    break
                current = nbs[rng.randint(len(nbs))]
                walk.append(current)
            walks.append(walk)
    return walks


def generate_node2vec_walks(G, num_walks=80, walk_length=10, p=1.0, q=1.0, seed=0):
    """Node2Vec: biased random walks with return parameter p and in-out parameter q."""
    rng = np.random.RandomState(seed)
    walks = []
    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            prev, current = None, node
            for _ in range(walk_length - 1):
                nbs = list(G.neighbors(current))
                if not nbs:
                    break
                if prev is None:
                    nxt = nbs[rng.randint(len(nbs))]
                else:
                    prev_set = set(G.neighbors(prev))
                    weights = []
                    for z in nbs:
                        if z == prev:
                            alpha = 1.0 / p
                        elif z in prev_set:
                            alpha = 1.0
                        else:
                            alpha = 1.0 / q
                        weights.append(alpha * G[current][z].get('weight', 1))
                    total = sum(weights)
                    probs = np.array(weights) / total
                    r = rng.random()
                    cum, nxt = 0.0, nbs[-1]
                    for idx, prob in enumerate(probs):
                        cum += prob
                        if r <= cum:
                            nxt = nbs[idx]
                            break
                walk.append(nxt)
                prev, current = current, nxt
            walks.append(walk)
    return walks


def train_word2vec(walks, vector_size=2, window=5, epochs=30, seed=0, workers=1):
    """Train Word2Vec on random walks. Returns embedding matrix."""
    model = Word2Vec(
        walks, vector_size=vector_size, window=window,
        hs=1, sg=1, workers=workers, seed=seed,
        min_count=1, sample=0,
    )
    model.train(walks, total_examples=model.corpus_count,
                epochs=epochs, report_delay=0)
    return model.wv.vectors


# ══════════════════════════════════════════════════════════════════════════════
# Clustering & Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(data, labels):
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "ch": np.nan}
    return {
        "silhouette": float(silhouette_score(data, labels)),
        "dbi": float(davies_bouldin_score(data, labels)),
        "ch": float(calinski_harabasz_score(data, labels)),
    }


def cluster_kmeans(data, n_clusters=2, seed=0):
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(data)


def cluster_spectral(adj, n_clusters=2, seed=0):
    return SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        random_state=seed, assign_labels='kmeans',
    ).fit_predict(adj)


def cluster_pca_kmeans(data, n_clusters=2, n_components=2, seed=0):
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(data)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(reduced)
    return labels, reduced


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline runners
# ══════════════════════════════════════════════════════════════════════════════
def run_deepwalk(matrix, G, seed=0):
    """DeepWalk pipeline: unbiased walks → Word2Vec → embeddings."""
    t0 = time.perf_counter()
    walks = generate_unbiased_walks(G, seed=seed)
    t_walks = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb = train_word2vec(walks, seed=seed)
    t_w2v = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = cluster_kmeans(emb, seed=seed)
    t_km = time.perf_counter() - t0
    m_emb = compute_metrics(emb, labels)
    m_bow = compute_metrics(matrix.astype(float), labels)
    return {
        "silhouette_emb": m_emb["silhouette"], "dbi_emb": m_emb["dbi"], "ch_emb": m_emb["ch"],
        "silhouette_bow": m_bow["silhouette"], "dbi_bow": m_bow["dbi"], "ch_bow": m_bow["ch"],
        "t_walks": t_walks, "t_w2v": t_w2v, "t_km": t_km,
        "t_total": t_walks + t_w2v + t_km,
    }


def run_node2vec(matrix, G, p=1.0, q=1.0, seed=0):
    """Node2Vec pipeline: biased walks → Word2Vec → embeddings."""
    t0 = time.perf_counter()
    walks = generate_node2vec_walks(G, p=p, q=q, seed=seed)
    t_walks = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb = train_word2vec(walks, seed=seed)
    t_w2v = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = cluster_kmeans(emb, seed=seed)
    t_km = time.perf_counter() - t0
    m_emb = compute_metrics(emb, labels)
    m_bow = compute_metrics(matrix.astype(float), labels)
    return {
        "silhouette_emb": m_emb["silhouette"], "dbi_emb": m_emb["dbi"], "ch_emb": m_emb["ch"],
        "silhouette_bow": m_bow["silhouette"], "dbi_bow": m_bow["dbi"], "ch_bow": m_bow["ch"],
        "t_walks": t_walks, "t_w2v": t_w2v, "t_km": t_km,
        "t_total": t_walks + t_w2v + t_km,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
def wilcoxon_test(x, y, alternative='two-sided'):
    x, y = np.asarray(x), np.asarray(y)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(x)}
    try:
        stat, p = scipy_wilcoxon(x, y, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p), "n": len(x)}
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(x)}


def effect_size_r(statistic, n):
    if n < 5:
        return np.nan
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        return 0.0
    z = (statistic - mean_w) / std_w
    return min(abs(z) / np.sqrt(n), 1.0)


def cliffs_delta(x, y):
    n, m = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n * m)


def holm_correction(p_values):
    p = np.array(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(p[idx] * (m - rank), 1.0)
    sorted_adj = adjusted[order]
    for i in range(len(sorted_adj) - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adjusted[order] = sorted_adj
    return adjusted.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name, fmt="pdf"):
    path = os.path.join(FIG_DIR, f"{name}.{fmt}")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return path


def fig_method_comparison(methods_data):
    """Bar chart: all methods Silhouette comparison."""
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    courses = sorted(set(d["course"] for d in methods_data))
    x = np.arange(len(courses))
    width = 0.15

    method_info = [
        ("bow", "BoW+KMeans", COLORS["bow"]),
        ("pca", "PCA+KMeans", COLORS["pca"]),
        ("spectral", "Spectral", COLORS["spectral"]),
        ("deepwalk", "DeepWalk", COLORS["deepwalk"]),
        ("node2vec", "Node2Vec(1,1)", COLORS["node2vec"]),
    ]

    metrics = [
        ("silhouette", "Silhouette Score (↑)", True),
        ("dbi", "Davies-Bouldin Index (↓)", False),
        ("ch", "Calinski-Harabasz Index (↑)", True),
    ]

    for ax_idx, (met_key, met_label, higher_better) in enumerate(metrics):
        ax = axes[ax_idx]
        for i_m, (mk, ml, mc) in enumerate(method_info):
            vals = []
            for c in courses:
                match = [d for d in methods_data if d["course"] == c and d["method"] == mk]
                if match:
                    vals.append(match[0].get(f"{met_key}_emb", match[0].get(f"{met_key}_bow", 0)))
                else:
                    vals.append(0)
            offset = (i_m - 2) * width
            bars = ax.bar(x + offset, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h) and h != 0:
                    fmt = f'{h:.1f}' if met_key == "ch" else f'{h:.3f}'
                    ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=5.5)

        ax.set_xlabel('Course')
        ax.set_ylabel(met_label)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Course {i}' for i in courses])
        if ax_idx == 0:
            ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle('Method Comparison: All Metrics (Revised Graph Construction)', fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, "method_comparison_all_metrics")


def fig_sensitivity_heatmap(sens_df):
    """Heatmaps for p,q sensitivity."""
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (met, title, cmap) in enumerate([
        ("silhouette", "Silhouette Score (↑)", "YlGn"),
        ("dbi", "Davies-Bouldin Index (↓)", "YlGn_r"),
        ("ch", "Calinski-Harabasz Index (↑)", "YlGn"),
    ]):
        pv = sens_df.groupby(["p", "q"])[met].mean().reset_index().pivot(index="p", columns="q", values=met)
        sns.heatmap(pv, annot=True, fmt=".3f" if met != "ch" else ".1f",
                    cmap=cmap, ax=axes[idx], cbar_kws={"shrink": 0.8})
        axes[idx].set_title(title)
        axes[idx].set_xlabel("q (in-out)")
        axes[idx].set_ylabel("p (return)")
    fig.suptitle("Node2Vec Parameter Sensitivity (averaged over 6 courses)", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, "sensitivity_heatmap")


def fig_stability_boxplot(stability_per_seed):
    """Boxplot: Node2Vec stability across 20 seeds."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    courses = sorted(stability_per_seed.keys(), key=int)
    all_data = [[r["silhouette_emb"] for r in stability_per_seed[c]] for c in courses]
    bp = ax.boxplot(all_data, tick_labels=[f"Course {c}" for c in courses],
                    patch_artist=True, boxprops=dict(facecolor=COLORS["node2vec"], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Node2Vec Stability Across 20 Random Seeds (Revised Graph)')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "stability_boxplot")


def fig_runtime(runtime_data):
    """Runtime comparison bar chart."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    courses = sorted(set(d["course"] for d in runtime_data))
    x = np.arange(len(courses))
    width = 0.15
    method_info = [
        ("BoW+KMeans", COLORS["bow"]), ("PCA+KMeans", COLORS["pca"]),
        ("Spectral", COLORS["spectral"]), ("DeepWalk", COLORS["deepwalk"]),
        ("Node2Vec(1,1)", COLORS["node2vec"]),
    ]
    for i_m, (ml, mc) in enumerate(method_info):
        vals = []
        for c in courses:
            match = [d for d in runtime_data if d["course"] == c and d["method"] == ml]
            vals.append(match[0]["t_total"] if match else 0)
        offset = (i_m - 2) * width
        ax.bar(x + offset, vals, width, label=ml, color=mc, alpha=0.85)
    ax.set_xlabel('Course')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(framealpha=0.9, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "runtime_comparison")


def fig_graph_density(old_stats, new_stats):
    """Graph density before/after preprocessing."""
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    courses = [str(i) for i in range(1, 7)]
    x = np.arange(6)
    w = 0.35
    ax.bar(x - w / 2, [old_stats[c]["density"] for c in courses], w,
           label='Original (all courses)', color='#E53935', alpha=0.8)
    ax.bar(x + w / 2, [new_stats[c]["density"] for c in courses], w,
           label='Revised (constant removed)', color='#43A047', alpha=0.8)
    for i_v, (old, new) in enumerate(zip(
        [old_stats[c]["density"] for c in courses],
        [new_stats[c]["density"] for c in courses])):
        ax.annotate(f'{old:.2f}', xy=(i_v - w / 2, old), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
        ax.annotate(f'{new:.2f}', xy=(i_v + w / 2, new), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Course')
    ax.set_ylabel('Graph Density')
    ax.set_title('Effect of Removing Constant Courses on Graph Density')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in range(1, 7)])
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return save_fig(fig, "graph_density_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    print("=" * 70)
    print("  FULL RE-EXPERIMENT: Node2Vec + DeepWalk + Baselines")
    print("  (with consistent constant-course removal)")
    print("=" * 70)

    # ── Environment ────────────────────────────────────────────────────────
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import numpy; env["numpy"] = numpy.__version__
        import scipy; env["scipy"] = scipy.__version__
        import sklearn; env["sklearn"] = sklearn.__version__
        import gensim; env["gensim"] = gensim.__version__
        import networkx; env["networkx"] = networkx.__version__
        import pandas; env["pandas"] = pandas.__version__
        import matplotlib; env["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    print(f"\n  Python {env['python']}, gensim {env.get('gensim','?')}, "
          f"numpy {env.get('numpy','?')}, sklearn {env.get('sklearn','?')}")
    with open(os.path.join(OUT_DIR, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    # ═══ PART 1: Preprocessing & Graph Analysis ═══
    print("\n" + "=" * 70)
    print("  PART 1: Preprocessing — Remove Constant Courses")
    print("=" * 70)

    course_data = {}  # {course_id: {matrix, matrix_orig, labels, graph, univ_cols, ...}}
    old_graph_stats, new_graph_stats = {}, {}

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        matrix_orig, labels = read_class(filepath)

        # Find constant (universal) columns
        univ_cols = find_universal_columns(matrix_orig)
        matrix_filtered = np.delete(matrix_orig, univ_cols, axis=1)

        # Build graphs before and after
        G_old = create_graph(matrix_orig)
        G_new = create_graph(matrix_filtered)

        old_graph_stats[str(i)] = graph_stats(G_old)
        new_graph_stats[str(i)] = graph_stats(G_new)

        course_data[i] = {
            "matrix_orig": matrix_orig,
            "matrix": matrix_filtered,
            "labels": labels,
            "graph": G_new,
            "univ_cols": univ_cols,
            "n_students": matrix_orig.shape[0],
            "n_courses_orig": matrix_orig.shape[1],
            "n_courses_filtered": matrix_filtered.shape[1],
        }

        print(f"  Course {i}: {matrix_orig.shape[0]} students, "
              f"{matrix_orig.shape[1]}→{matrix_filtered.shape[1]} courses "
              f"(removed {len(univ_cols)} constant: cols {univ_cols})")
        print(f"    Old graph: {old_graph_stats[str(i)]['n_edges']} edges, "
              f"density={old_graph_stats[str(i)]['density']:.4f}, "
              f"complete={old_graph_stats[str(i)]['is_complete']}")
        print(f"    New graph: {new_graph_stats[str(i)]['n_edges']} edges, "
              f"density={new_graph_stats[str(i)]['density']:.4f}, "
              f"complete={new_graph_stats[str(i)]['is_complete']}, "
              f"components={new_graph_stats[str(i)]['n_components']}")
        print(f"    Degree: avg={new_graph_stats[str(i)]['avg_degree']:.1f}, "
              f"range=[{new_graph_stats[str(i)]['min_degree']}, {new_graph_stats[str(i)]['max_degree']}]")
        print(f"    Weights: avg={new_graph_stats[str(i)]['avg_weight']:.2f}, "
              f"std={new_graph_stats[str(i)]['std_weight']:.2f}")

    # Preprocessing summary
    preprocessing_info = {}
    for i in FILE_INDICES:
        cd = course_data[i]
        preprocessing_info[str(i)] = {
            "n_students": cd["n_students"],
            "n_courses_orig": cd["n_courses_orig"],
            "n_courses_filtered": cd["n_courses_filtered"],
            "n_removed": len(cd["univ_cols"]),
            "removed_columns": cd["univ_cols"],
            "target_course_removed": 0 in cd["univ_cols"],  # column 0 is typically the target
        }
        # Check if any other courses besides column 0 are constant
        other_constant = [c for c in cd["univ_cols"] if c != 0]
        preprocessing_info[str(i)]["other_constant_courses"] = other_constant

    with open(os.path.join(OUT_DIR, "preprocessing_info.json"), "w") as f:
        json.dump(preprocessing_info, f, indent=2)
    with open(os.path.join(OUT_DIR, "graph_analysis.json"), "w") as f:
        json.dump({"old": old_graph_stats, "new": new_graph_stats}, f, indent=2)

    # Figure: graph density
    fig_graph_density(old_graph_stats, new_graph_stats)
    print("\n  Saved: graph_density_comparison.pdf")

    # ═══ PART 2: All Methods (seed=0) ═══
    print("\n" + "=" * 70)
    print("  PART 2: All Methods (seed=0)")
    print("=" * 70)

    all_results = []
    runtime_data = []

    for i in FILE_INDICES:
        cd = course_data[i]
        matrix = cd["matrix"]
        G = cd["graph"]
        n = cd["n_students"]
        print(f"\n  Course {i}: {n} students, {matrix.shape[1]} courses, {G.number_of_edges()} edges")

        # 1. BoW + KMeans
        t0 = time.perf_counter()
        bow_labels = cluster_kmeans(matrix.astype(float), seed=0)
        t_bow = time.perf_counter() - t0
        bow_m = compute_metrics(matrix.astype(float), bow_labels)
        print(f"    BoW+KMeans:    Sil={bow_m['silhouette']:.3f}, DBI={bow_m['dbi']:.3f}, CH={bow_m['ch']:.1f}")
        all_results.append({
            "course": i, "method": "bow", "n_students": n,
            "silhouette_emb": bow_m["silhouette"], "dbi_emb": bow_m["dbi"], "ch_emb": bow_m["ch"],
            "silhouette_bow": bow_m["silhouette"], "dbi_bow": bow_m["dbi"], "ch_bow": bow_m["ch"],
        })
        runtime_data.append({"course": i, "method": "BoW+KMeans", "t_total": t_bow})

        # 2. PCA + KMeans
        t0 = time.perf_counter()
        pca_labels, pca_reduced = cluster_pca_kmeans(matrix.astype(float), seed=0)
        t_pca = time.perf_counter() - t0
        pca_m_bow = compute_metrics(matrix.astype(float), pca_labels)
        pca_m_pca = compute_metrics(pca_reduced, pca_labels)
        print(f"    PCA+KMeans:    Sil={pca_m_bow['silhouette']:.3f}, DBI={pca_m_bow['dbi']:.3f}, CH={pca_m_bow['ch']:.1f}")
        all_results.append({
            "course": i, "method": "pca", "n_students": n,
            "silhouette_emb": pca_m_pca["silhouette"], "dbi_emb": pca_m_pca["dbi"], "ch_emb": pca_m_pca["ch"],
            "silhouette_bow": pca_m_bow["silhouette"], "dbi_bow": pca_m_bow["dbi"], "ch_bow": pca_m_bow["ch"],
        })
        runtime_data.append({"course": i, "method": "PCA+KMeans", "t_total": t_pca})

        # 3. Spectral Clustering
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        t0 = time.perf_counter()
        spec_labels = cluster_spectral(adj, seed=0)
        t_spec = time.perf_counter() - t0
        spec_m_bow = compute_metrics(matrix.astype(float), spec_labels)
        spec_m_adj = compute_metrics(adj, spec_labels)
        print(f"    Spectral:      Sil={spec_m_bow['silhouette']:.3f}, DBI={spec_m_bow['dbi']:.3f}, CH={spec_m_bow['ch']:.1f}")
        all_results.append({
            "course": i, "method": "spectral", "n_students": n,
            "silhouette_emb": spec_m_adj["silhouette"], "dbi_emb": spec_m_adj["dbi"], "ch_emb": spec_m_adj["ch"],
            "silhouette_bow": spec_m_bow["silhouette"], "dbi_bow": spec_m_bow["dbi"], "ch_bow": spec_m_bow["ch"],
        })
        runtime_data.append({"course": i, "method": "Spectral", "t_total": t_spec})

        # 4. DeepWalk + KMeans
        dw = run_deepwalk(matrix, G, seed=0)
        print(f"    DeepWalk:      Sil={dw['silhouette_emb']:.3f}, DBI={dw['dbi_emb']:.3f}, CH={dw['ch_emb']:.1f}")
        all_results.append({
            "course": i, "method": "deepwalk", "n_students": n,
            "silhouette_emb": dw["silhouette_emb"], "dbi_emb": dw["dbi_emb"], "ch_emb": dw["ch_emb"],
            "silhouette_bow": dw["silhouette_bow"], "dbi_bow": dw["dbi_bow"], "ch_bow": dw["ch_bow"],
        })
        runtime_data.append({"course": i, "method": "DeepWalk", "t_total": dw["t_total"]})

        # 5. Node2Vec(p=1,q=1) + KMeans
        n2v = run_node2vec(matrix, G, p=1.0, q=1.0, seed=0)
        print(f"    Node2Vec(1,1): Sil={n2v['silhouette_emb']:.3f}, DBI={n2v['dbi_emb']:.3f}, CH={n2v['ch_emb']:.1f}")
        all_results.append({
            "course": i, "method": "node2vec", "n_students": n,
            "silhouette_emb": n2v["silhouette_emb"], "dbi_emb": n2v["dbi_emb"], "ch_emb": n2v["ch_emb"],
            "silhouette_bow": n2v["silhouette_bow"], "dbi_bow": n2v["dbi_bow"], "ch_bow": n2v["ch_bow"],
        })
        runtime_data.append({"course": i, "method": "Node2Vec(1,1)", "t_total": n2v["t_total"]})

    results_df = pd.DataFrame(all_results)
    results_df.to_json(os.path.join(OUT_DIR, "all_methods.json"), orient="records", indent=2)
    results_df.to_excel(os.path.join(OUT_DIR, "all_methods.xlsx"), index=False)
    rt_df = pd.DataFrame(runtime_data)
    rt_df.to_json(os.path.join(OUT_DIR, "runtime.json"), orient="records", indent=2)

    # Summary table
    print("\n  SUMMARY TABLE")
    print(f"  {'Method':<20}", end="")
    for i in FILE_INDICES:
        print(f" {'C'+str(i):>6}", end="")
    print(f" {'Avg':>8}")
    print("  " + "-" * 60)
    for mk, ml in [("bow", "BoW+KMeans"), ("pca", "PCA+KMeans"), ("spectral", "Spectral"),
                    ("deepwalk", "DeepWalk"), ("node2vec", "Node2Vec(1,1)")]:
        sub = results_df[results_df["method"] == mk]
        print(f"  {ml:<20}", end="")
        for i in FILE_INDICES:
            row = sub[sub["course"] == i]
            v = row["silhouette_bow"].values[0] if mk in ("bow", "pca", "spectral") else row["silhouette_emb"].values[0]
            print(f" {v:>6.3f}", end="")
        avg = sub["silhouette_bow"].mean() if mk in ("bow", "pca", "spectral") else sub["silhouette_emb"].mean()
        print(f" {avg:>8.3f}")

    # ═══ PART 3: Node2Vec Sensitivity (p,q grid) ═══
    print("\n" + "=" * 70)
    print("  PART 3: Node2Vec Parameter Sensitivity")
    print("=" * 70)

    p_vals, q_vals = [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]
    sens_results = []

    for p in p_vals:
        for q in q_vals:
            sils, dbis, chs = [], [], []
            for i in FILE_INDICES:
                cd = course_data[i]
                res = run_node2vec(cd["matrix"], cd["graph"], p=p, q=q, seed=0)
                sils.append(res["silhouette_emb"])
                dbis.append(res["dbi_emb"])
                chs.append(res["ch_emb"])
                sens_results.append({
                    "p": p, "q": q, "course": i,
                    "silhouette": res["silhouette_emb"],
                    "dbi": res["dbi_emb"],
                    "ch": res["ch_emb"],
                })
            print(f"    p={p:.1f}, q={q:.1f}: Sil={np.mean(sils):.3f}, DBI={np.mean(dbis):.3f}, CH={np.mean(chs):.1f}")

    sens_df = pd.DataFrame(sens_results)
    sens_df.to_json(os.path.join(OUT_DIR, "sensitivity.json"), orient="records", indent=2)
    sens_df.to_excel(os.path.join(OUT_DIR, "sensitivity.xlsx"), index=False)

    # Sensitivity heatmap
    fig_sensitivity_heatmap(sens_df)
    print("  Saved: sensitivity_heatmap.pdf")

    # ═══ PART 4: DeepWalk Equivalence Test ═══
    print("\n" + "=" * 70)
    print("  PART 4: DeepWalk vs Node2Vec(p=1,q=1) Validation")
    print("=" * 70)

    cd = course_data[1]  # Use course 1 for validation
    G = cd["graph"]
    node = 0
    nbs = list(G.neighbors(node))
    expected = 1.0 / len(nbs) if nbs else 0

    # Node2Vec(p=1,q=1) walks
    n2v_walks = generate_node2vec_walks(G, num_walks=1000, walk_length=2, p=1.0, q=1.0, seed=42)
    n2v_first = [w[1] for w in n2v_walks if w[0] == node]

    # DeepWalk walks
    dw_walks = generate_unbiased_walks(G, num_walks=1000, walk_length=2, seed=42)
    dw_first = [w[1] for w in dw_walks if w[0] == node]

    from collections import Counter
    n2v_counts = Counter(n2v_first)
    dw_counts = Counter(dw_first)
    n2v_probs = {k: v / len(n2v_first) for k, v in n2v_counts.items()}
    dw_probs = {k: v / len(dw_first) for k, v in dw_counts.items()}

    n2v_max_dev = max(abs(n2v_probs.get(nb, 0) - expected) for nb in nbs)
    dw_max_dev = max(abs(dw_probs.get(nb, 0) - expected) for nb in nbs)

    print(f"  Course 1, Node {node}: {len(nbs)} neighbors")
    print(f"  Expected uniform: {expected:.4f}")
    print(f"  Node2Vec(p=1,q=1) max deviation: {n2v_max_dev:.4f} {'✓' if n2v_max_dev < 0.05 else '✗'}")
    print(f"  DeepWalk max deviation:            {dw_max_dev:.4f} {'✓' if dw_max_dev < 0.05 else '✗'}")

    validation = {
        "node": node, "n_neighbors": len(nbs),
        "expected_prob": expected,
        "node2vec_max_dev": n2v_max_dev, "node2vec_valid": n2v_max_dev < 0.05,
        "deepwalk_max_dev": dw_max_dev, "deepwalk_valid": dw_max_dev < 0.05,
        "n_walks": len(n2v_first),
    }
    with open(os.path.join(OUT_DIR, "walk_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    # Also compare DeepWalk vs Node2Vec(p=1,q=1) embeddings
    print("\n  DeepWalk vs Node2Vec(p=1,q=1) embeddings (seed=0):")
    for i in FILE_INDICES:
        cd = course_data[i]
        dw_emb = train_word2vec(generate_unbiased_walks(cd["graph"], seed=0), seed=0)
        n2v_emb = train_word2vec(generate_node2vec_walks(cd["graph"], p=1.0, q=1.0, seed=0), seed=0)
        # Cosine similarity between embedding sets
        from sklearn.metrics.pairwise import cosine_similarity
        sim = np.mean(np.max(cosine_similarity(dw_emb, n2v_emb), axis=1))
        dw_sil = float(silhouette_score(dw_emb, cluster_kmeans(dw_emb, seed=0)))
        n2v_sil = float(silhouette_score(n2v_emb, cluster_kmeans(n2v_emb, seed=0)))
        print(f"    Course {i}: DW Sil={dw_sil:.3f}, N2V Sil={n2v_sil:.3f}, avg cosine sim={sim:.4f}")

    # ═══ PART 5: Stability Analysis (20 seeds) ═══
    print("\n" + "=" * 70)
    print(f"  PART 5: Stability Analysis ({NUM_SEEDS} seeds)")
    print("=" * 70)

    stability_per_seed = {str(i): [] for i in FILE_INDICES}

    for seed in range(NUM_SEEDS):
        for i in FILE_INDICES:
            cd = course_data[i]
            # Node2Vec(p=1,q=1)
            n2v_res = run_node2vec(cd["matrix"], cd["graph"], p=1.0, q=1.0, seed=seed)
            # DeepWalk
            dw_res = run_deepwalk(cd["matrix"], cd["graph"], seed=seed)

            stability_per_seed[str(i)].append({
                "seed": seed,
                "n2v_silhouette": n2v_res["silhouette_emb"],
                "n2v_dbi": n2v_res["dbi_emb"],
                "n2v_ch": n2v_res["ch_emb"],
                "dw_silhouette": dw_res["silhouette_emb"],
                "dw_dbi": dw_res["dbi_emb"],
                "dw_ch": dw_res["ch_emb"],
            })
        elapsed = time.time() - t_total
        if (seed + 1) % 5 == 0 or seed == NUM_SEEDS - 1:
            print(f"  Seed {seed + 1}/{NUM_SEEDS} done ({elapsed:.0f}s)")

    # Compute statistics
    stability_stats = {}
    for i in FILE_INDICES:
        n2v_sils = [r["n2v_silhouette"] for r in stability_per_seed[str(i)]]
        dw_sils = [r["dw_silhouette"] for r in stability_per_seed[str(i)]]
        stability_stats[str(i)] = {
            "node2vec": {
                "silhouette": {"mean": float(np.mean(n2v_sils)), "std": float(np.std(n2v_sils)),
                               "min": float(np.min(n2v_sils)), "max": float(np.max(n2v_sils))},
            },
            "deepwalk": {
                "silhouette": {"mean": float(np.mean(dw_sils)), "std": float(np.std(dw_sils)),
                               "min": float(np.min(dw_sils)), "max": float(np.max(dw_sils))},
            },
        }
        print(f"  Course {i}: N2V={np.mean(n2v_sils):.3f}±{np.std(n2v_sils):.3f}, "
              f"DW={np.mean(dw_sils):.3f}±{np.std(dw_sils):.3f}")

    with open(os.path.join(OUT_DIR, "stability.json"), "w") as f:
        json.dump(stability_stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "stability_per_seed.json"), "w") as f:
        json.dump(stability_per_seed, f, indent=2)

    # Stability boxplot
    fig_stability_boxplot(stability_per_seed)
    print("  Saved: stability_boxplot.pdf")

    # ═══ PART 6: Statistical Analysis ═══
    print("\n" + "=" * 70)
    print("  PART 6: Statistical Analysis")
    print("=" * 70)

    # Mean scores across seeds (the proper statistical unit: course-level)
    n2v_means = np.array([stability_stats[str(c)]["node2vec"]["silhouette"]["mean"] for c in FILE_INDICES])
    dw_means = np.array([stability_stats[str(c)]["deepwalk"]["silhouette"]["mean"] for c in FILE_INDICES])

    # Deterministic baselines
    bow_scores, pca_scores, spec_scores = [], [], []
    for i in FILE_INDICES:
        cd = course_data[i]
        bow_l = cluster_kmeans(cd["matrix"].astype(float), seed=0)
        bow_scores.append(float(silhouette_score(cd["matrix"].astype(float), bow_l)))
        pca_l, pca_r = cluster_pca_kmeans(cd["matrix"].astype(float), seed=0)
        pca_scores.append(float(silhouette_score(cd["matrix"].astype(float), pca_l)))
        adj = nx.to_numpy_array(cd["graph"], nodelist=sorted(cd["graph"].nodes()))
        spec_l = cluster_spectral(adj, seed=0)
        spec_scores.append(float(silhouette_score(cd["matrix"].astype(float), spec_l)))

    bow_scores = np.array(bow_scores)
    pca_scores = np.array(pca_scores)
    spec_scores = np.array(spec_scores)

    print(f"\n  Course-level Silhouette scores:")
    print(f"  {'Course':<10} {'Node2Vec':>10} {'DeepWalk':>10} {'BoW':>10} {'PCA':>10} {'Spectral':>10}")
    print("  " + "-" * 60)
    for ci, c in enumerate(FILE_INDICES):
        print(f"  Course {c:<4} {n2v_means[ci]:>10.3f} {dw_means[ci]:>10.3f} "
              f"{bow_scores[ci]:>10.3f} {pca_scores[ci]:>10.3f} {spec_scores[ci]:>10.3f}")
    print(f"  {'Average':<10} {np.mean(n2v_means):>10.3f} {np.mean(dw_means):>10.3f} "
          f"{np.mean(bow_scores):>10.3f} {np.mean(pca_scores):>10.3f} {np.mean(spec_scores):>10.3f}")

    # Pairwise comparisons
    comparisons = [
        ("Node2Vec vs DeepWalk", n2v_means, dw_means),
        ("Node2Vec vs BoW+KMeans", n2v_means, bow_scores),
        ("Node2Vec vs PCA+KMeans", n2v_means, pca_scores),
        ("Node2Vec vs Spectral", n2v_means, spec_scores),
        ("DeepWalk vs BoW+KMeans", dw_means, bow_scores),
        ("DeepWalk vs PCA+KMeans", dw_means, pca_scores),
    ]

    stat_results = []
    raw_p = []

    for name, x, y in comparisons:
        test = wilcoxon_test(x, y, alternative='two-sided')
        r = effect_size_r(test["statistic"], test["n"])
        delta = cliffs_delta(x, y)
        raw_p.append(test["p_value"])
        res = {
            "comparison": name,
            "mean_x": float(np.mean(x)), "mean_y": float(np.mean(y)),
            "mean_diff": float(np.mean(x - y)),
            "wilcoxon_stat": test["statistic"], "wilcoxon_p": test["p_value"],
            "effect_size_r": r, "cliffs_delta": delta,
        }
        stat_results.append(res)
        print(f"\n  {name}:")
        print(f"    Mean: {np.mean(x):.3f} vs {np.mean(y):.3f} (diff: {np.mean(x-y):+.3f})")
        print(f"    Wilcoxon: p={test['p_value']:.4f}, r={r:.3f}, Cliff's δ={delta:.3f}")

    # Holm correction
    corrected_p = holm_correction(raw_p)
    for cp, res in zip(corrected_p, stat_results):
        res["corrected_p"] = cp
        res["significant"] = cp < 0.05
        print(f"  {res['comparison']}: corrected p={cp:.4f} "
              f"({'*' if cp < 0.05 else 'ns'})")

    with open(os.path.join(OUT_DIR, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)

    # ═══ PART 7: Figures ═══
    print("\n" + "=" * 70)
    print("  PART 7: Generating Figures")
    print("=" * 70)

    fig_method_comparison(all_results)
    print("  Saved: method_comparison_all_metrics.pdf")
    fig_runtime(runtime_data)
    print("  Saved: runtime_comparison.pdf")

    # ═══ PART 8: Final Summary ═══
    total_time = time.time() - t_total
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Total runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Figures directory: {FIG_DIR}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    print(f"  - Constant courses removed: 1 per course (the target course)")
    avg_new_dens = np.mean([new_graph_stats[str(i)]["density"] for i in FILE_INDICES])
    print(f"  - Average graph density after preprocessing: {avg_new_dens:.3f} (was 1.000)")
    print(f"  - Any remaining complete graphs: "
          f"{any(new_graph_stats[str(i)]['is_complete'] for i in FILE_INDICES)}")
    print(f"  - Node2Vec(p=1,q=1) avg Silhouette: {np.mean(n2v_means):.3f}")
    print(f"  - DeepWalk avg Silhouette: {np.mean(dw_means):.3f}")
    print(f"  - BoW avg Silhouette: {np.mean(bow_scores):.3f}")
    print(f"  - PCA avg Silhouette: {np.mean(pca_scores):.3f}")
    print(f"  - Spectral avg Silhouette: {np.mean(spec_scores):.3f}")

    pq_avgs = sens_df.groupby(["p", "q"])["silhouette"].mean()
    best_pq = pq_avgs.idxmax()
    print(f"  - Best Node2Vec config: p={best_pq[0]}, q={best_pq[1]} (Sil={pq_avgs.max():.3f})")
    print(f"  - Sensitivity range: {pq_avgs.min():.3f} to {pq_avgs.max():.3f} (Δ={pq_avgs.max()-pq_avgs.min():.3f})")

    print(f"\n  Generated files:")
    for f_name in sorted(os.listdir(OUT_DIR)):
        if not f_name.startswith(".") and f_name != "figures":
            print(f"    {f_name}")
    for f_name in sorted(os.listdir(FIG_DIR)):
        print(f"    figures/{f_name}")

    print(f"\n  Analysis complete. Manuscript revision should follow.")
    print()


if __name__ == "__main__":
    main()
