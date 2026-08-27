# -*- coding: utf-8 -*-
"""
Complete Linux Re-Experiment for DeepWalk-SSP
================================================
Runs ALL methods consistently on the current Linux environment:
  1. Student-Course representation + KMeans (BoW)
  2. PCA + KMeans
  3. Spectral Clustering
  4. DeepWalk-SSP
  5. Node2Vec-SSP (multiple configurations)

Also performs:
  - Graph structure analysis
  - Node2Vec validation (walk-level comparison with DeepWalk)
  - Statistical analysis
  - Runtime measurement
  - Clustering stability
  - Publication-quality figure generation (PDF)

All results are saved to results/linux_reexperiment/
Original results in results/ are NOT modified.
"""

import os
import sys
import time
import json
import warnings
import platform
import subprocess

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import (
    FILE_INDICES, DATA_DIR, RESULTS_DIR, FIGURES_DIR, COLORS,
    PUBLICATION_STYLE, DEFAULT_PARAMS, NUM_SEEDS, SEED_RANGE
)
from experiments.core import (
    read_class, create_graph_from_bow, set_seed,
    generate_random_walks, generate_node2vec_walks,
    train_word2vec, run_pipeline, run_node2vec_pipeline
)
from experiments.evaluation import (
    compute_all_metrics, run_pca_kmeans, run_spectral,
    evaluate_clustering, compute_clustering_stability
)
from experiments.stats import (
    wilcoxon_signed_rank_test, compute_effect_size_r,
    compute_cliffs_delta, compute_bootstrap_ci,
    benjamini_hochberg_correction
)

# ── Output directories ───────────────────────────────────────────────────────
LINUX_DIR = os.path.join(RESULTS_DIR, "linux_reexperiment")
LINUX_FIG_DIR = os.path.join(LINUX_DIR, "figures")
os.makedirs(LINUX_DIR, exist_ok=True)
os.makedirs(LINUX_FIG_DIR, exist_ok=True)

# ── Custom JSON encoder ──────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n  --- {title} ---")


def save_fig(fig, filename, directory=LINUX_FIG_DIR, dpi=300):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return path


def setup_pub_style():
    plt.rcParams.update({
        "figure.figsize": (7, 5),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ============================================================
# SECTION A: Environment Record
# ============================================================
def record_environment():
    """Record the Python environment for reproducibility."""
    print_header("SECTION A: Environment")

    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }

    # Get package versions
    for pkg_name in ["scipy", "sklearn", "gensim", "networkx", "pandas", "matplotlib", "seaborn"]:
        try:
            mod = __import__(pkg_name)
            env_info[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env_info[pkg_name] = "NOT INSTALLED"

    # gensim is imported as gensim
    try:
        import gensim
        env_info["gensim"] = gensim.__version__
    except:
        env_info["gensim"] = "NOT INSTALLED"

    print(f"  Python: {env_info['python_version']}")
    print(f"  Platform: {env_info['platform']}")
    print(f"  numpy: {env_info['numpy']}")
    print(f"  scipy: {env_info['scipy']}")
    print(f"  scikit-learn: {env_info['sklearn']}")
    print(f"  gensim: {env_info['gensim']}")
    print(f"  networkx: {env_info['networkx']}")
    print(f"  pandas: {env_info['pandas']}")
    print(f"  matplotlib: {env_info['matplotlib']}")
    print(f"  seaborn: {env_info['seaborn']}")

    with open(os.path.join(LINUX_DIR, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    return env_info


# ============================================================
# SECTION B: Graph Structure Analysis
# ============================================================
def analyze_graphs():
    """Analyze co-enrollment graph structure for all six courses."""
    print_header("SECTION B: Graph Structure Analysis")

    graph_stats = {}

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, student_labels = read_class(filepath)
        if scm is None:
            continue

        n_students = scm.shape[0]
        n_courses = scm.shape[1]
        G = create_graph_from_bow(scm)

        n_edges = G.number_of_edges()
        max_possible_edges = n_students * (n_students - 1) // 2
        density = nx.density(G)
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)

        # Check if graph is complete
        is_complete = density >= 0.999
        is_connected = nx.is_connected(G)
        n_components = nx.number_connected_components(G)

        stats = {
            "n_students": n_students,
            "n_courses": n_courses,
            "n_edges": n_edges,
            "max_possible_edges": max_possible_edges,
            "density": float(density),
            "is_complete": is_complete,
            "is_connected": is_connected,
            "n_components": n_components,
            "avg_degree": float(avg_degree),
            "std_degree": float(std_degree),
            "min_degree": int(min_degree),
            "max_degree": int(max_degree),
        }
        graph_stats[str(file_idx)] = stats

        print(f"  Course {file_idx}: {n_students} students, {n_courses} courses")
        print(f"    Edges: {n_edges} / {max_possible_edges} possible")
        print(f"    Density: {density:.4f}")
        print(f"    Complete graph: {is_complete}")
        print(f"    Connected: {is_connected}, Components: {n_components}")
        print(f"    Degree: avg={avg_degree:.1f}, std={std_degree:.1f}, range=[{min_degree}, {max_degree}]")

    # Implication for Node2Vec
    all_complete = all(g["is_complete"] for g in graph_stats.values())
    print_subheader("Implication for Node2Vec")
    if all_complete:
        print("  ALL six co-enrollment graphs are COMPLETE (fully connected).")
        print("  This means every student shares at least one course with every other student.")
        print("  Implication: Node2Vec's q parameter (BFS/DFS bias) has LIMITED effect")
        print("  because the exploration neighborhoods overlap almost entirely.")
        print("  Only the return parameter p can meaningfully differentiate walks.")
    else:
        print("  Not all graphs are complete — q parameter may have a meaningful effect.")

    with open(os.path.join(LINUX_DIR, "graph_analysis.json"), "w") as f:
        json.dump(graph_stats, f, indent=2, cls=NumpyEncoder)

    return graph_stats


# ============================================================
# SECTION C: Re-run All Methods (seed=0)
# ============================================================
def run_all_methods():
    """Run all five methods on all six courses with seed=0."""
    print_header("SECTION C: Re-run All Methods (seed=0)")

    all_results = []

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, student_labels = read_class(filepath)
        if scm is None:
            print(f"  Course {file_idx}: FAILED to load")
            continue

        n_students = scm.shape[0]
        n_courses = scm.shape[1]
        print(f"\n  Course {file_idx}: {n_students} students, {n_courses} courses")

        # Build graph (shared)
        t0 = time.perf_counter()
        G = create_graph_from_bow(scm)
        t_graph = time.perf_counter() - t0

        # ── Method 1: BoW + KMeans ──────────────────────────────────────────
        t0 = time.perf_counter()
        bow_eval = evaluate_clustering(scm.astype(float), n_clusters=2, random_state=0)
        t_bow = time.perf_counter() - t0
        bow_sil = bow_eval["kmeans"]["metrics"]["silhouette"]
        bow_dbi = bow_eval["kmeans"]["metrics"]["davies_bouldin"]
        bow_ch = bow_eval["kmeans"]["metrics"]["calinski_harabasz"]
        print(f"    BoW+KMeans:       Sil={bow_sil:.4f}  DBI={bow_dbi:.4f}  CH={bow_ch:.1f}  ({t_bow:.3f}s)")

        # ── Method 2: PCA + KMeans ──────────────────────────────────────────
        t0 = time.perf_counter()
        pca_labels, pca_reduced = run_pca_kmeans(
            scm.astype(float), n_clusters=2, n_components=2, random_state=0
        )
        t_pca = time.perf_counter() - t0
        pca_metrics = compute_all_metrics(pca_reduced, pca_labels)
        pca_sil = pca_metrics["silhouette"]
        pca_dbi = pca_metrics["davies_bouldin"]
        pca_ch = pca_metrics["calinski_harabasz"]
        print(f"    PCA+KMeans:       Sil={pca_sil:.4f}  DBI={pca_dbi:.4f}  CH={pca_ch:.1f}  ({t_pca:.3f}s)")

        # ── Method 3: Spectral Clustering ───────────────────────────────────
        adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        t0 = time.perf_counter()
        spectral_labels = run_spectral(adj_matrix, n_clusters=2, random_state=0)
        t_spec = time.perf_counter() - t0
        spec_metrics = compute_all_metrics(scm.astype(float), spectral_labels)
        spec_sil = spec_metrics["silhouette"]
        spec_dbi = spec_metrics["davies_bouldin"]
        spec_ch = spec_metrics["calinski_harabasz"]
        print(f"    Spectral:         Sil={spec_sil:.4f}  DBI={spec_dbi:.4f}  CH={spec_ch:.1f}  ({t_spec:.3f}s)")

        # ── Method 4: DeepWalk-SSP ──────────────────────────────────────────
        t0 = time.perf_counter()
        dw_result = run_pipeline(filepath, seed=0)
        t_dw = time.perf_counter() - t0
        if dw_result["success"]:
            dw_eval = evaluate_clustering(dw_result["embeddings"], n_clusters=2, random_state=0)
            dw_sil = dw_eval["kmeans"]["metrics"]["silhouette"]
            dw_dbi = dw_eval["kmeans"]["metrics"]["davies_bouldin"]
            dw_ch = dw_eval["kmeans"]["metrics"]["calinski_harabasz"]
            dw_timing = dw_result["timing"]
        else:
            dw_sil = dw_dbi = dw_ch = np.nan
            dw_timing = {}
        print(f"    DeepWalk-SSP:     Sil={dw_sil:.4f}  DBI={dw_dbi:.4f}  CH={dw_ch:.1f}  ({t_dw:.3f}s)")

        # ── Method 5: Node2Vec-SSP (p=1.0, q=1.0 — baseline) ──────────────
        t0 = time.perf_counter()
        n2v_result = run_node2vec_pipeline(
            filepath, p=1.0, q=1.0, seed=0
        )
        t_n2v11 = time.perf_counter() - t0
        if n2v_result["success"]:
            n2v_eval = evaluate_clustering(n2v_result["embeddings"], n_clusters=2, random_state=0)
            n2v11_sil = n2v_eval["kmeans"]["metrics"]["silhouette"]
            n2v11_dbi = n2v_eval["kmeans"]["metrics"]["davies_bouldin"]
            n2v11_ch = n2v_eval["kmeans"]["metrics"]["calinski_harabasz"]
        else:
            n2v11_sil = n2v11_dbi = n2v11_ch = np.nan
        print(f"    Node2Vec(1.0,1.0): Sil={n2v11_sil:.4f}  DBI={n2v11_dbi:.4f}  CH={n2v11_ch:.1f}  ({t_n2v11:.3f}s)")

        # ── Method 5b: Node2Vec-SSP (p=0.5, q=1.0) ────────────────────────
        t0 = time.perf_counter()
        n2v_result_b = run_node2vec_pipeline(
            filepath, p=0.5, q=1.0, seed=0
        )
        t_n2v05_10 = time.perf_counter() - t0
        if n2v_result_b["success"]:
            n2v_eval_b = evaluate_clustering(n2v_result_b["embeddings"], n_clusters=2, random_state=0)
            n2v05_10_sil = n2v_eval_b["kmeans"]["metrics"]["silhouette"]
            n2v05_10_dbi = n2v_eval_b["kmeans"]["metrics"]["davies_bouldin"]
            n2v05_10_ch = n2v_eval_b["kmeans"]["metrics"]["calinski_harabasz"]
        else:
            n2v05_10_sil = n2v05_10_dbi = n2v05_10_ch = np.nan
        print(f"    Node2Vec(0.5,1.0): Sil={n2v05_10_sil:.4f}  DBI={n2v05_10_dbi:.4f}  CH={n2v05_10_ch:.1f}  ({t_n2v05_10:.3f}s)")

        # ── Method 5c: Node2Vec-SSP (p=0.5, q=2.0) ────────────────────────
        t0 = time.perf_counter()
        n2v_result_c = run_node2vec_pipeline(
            filepath, p=0.5, q=2.0, seed=0
        )
        t_n2v05_20 = time.perf_counter() - t0
        if n2v_result_c["success"]:
            n2v_eval_c = evaluate_clustering(n2v_result_c["embeddings"], n_clusters=2, random_state=0)
            n2v05_20_sil = n2v_eval_c["kmeans"]["metrics"]["silhouette"]
            n2v05_20_dbi = n2v_eval_c["kmeans"]["metrics"]["davies_bouldin"]
            n2v05_20_ch = n2v_eval_c["kmeans"]["metrics"]["calinski_harabasz"]
        else:
            n2v05_20_sil = n2v05_20_dbi = n2v05_20_ch = np.nan
        print(f"    Node2Vec(0.5,2.0): Sil={n2v05_20_sil:.4f}  DBI={n2v05_20_dbi:.4f}  CH={n2v05_20_ch:.1f}  ({t_n2v05_20:.3f}s)")

        # ── Method 5d: Node2Vec-SSP (p=1.0, q=0.5) ────────────────────────
        t0 = time.perf_counter()
        n2v_result_d = run_node2vec_pipeline(
            filepath, p=1.0, q=0.5, seed=0
        )
        t_n2v10_05 = time.perf_counter() - t0
        if n2v_result_d["success"]:
            n2v_eval_d = evaluate_clustering(n2v_result_d["embeddings"], n_clusters=2, random_state=0)
            n2v10_05_sil = n2v_eval_d["kmeans"]["metrics"]["silhouette"]
            n2v10_05_dbi = n2v_eval_d["kmeans"]["metrics"]["davies_bouldin"]
            n2v10_05_ch = n2v_eval_d["kmeans"]["metrics"]["calinski_harabasz"]
        else:
            n2v10_05_sil = n2v10_05_dbi = n2v10_05_ch = np.nan
        print(f"    Node2Vec(1.0,0.5): Sil={n2v10_05_sil:.4f}  DBI={n2v10_05_dbi:.4f}  CH={n2v10_05_ch:.1f}  ({t_n2v10_05:.3f}s)")

        # Collect all results
        row = {
            "course": file_idx,
            "n_students": n_students,
            "n_courses": n_courses,
            "n_edges": G.number_of_edges(),
            "graph_density": float(nx.density(G)),
            # BoW
            "bow_silhouette": float(bow_sil),
            "bow_davies_bouldin": float(bow_dbi),
            "bow_calinski_harabasz": float(bow_ch),
            "bow_runtime_s": float(t_bow),
            # PCA
            "pca_silhouette": float(pca_sil),
            "pca_davies_bouldin": float(pca_dbi),
            "pca_calinski_harabasz": float(pca_ch),
            "pca_runtime_s": float(t_pca),
            # Spectral
            "spectral_silhouette": float(spec_sil),
            "spectral_davies_bouldin": float(spec_dbi),
            "spectral_calinski_harabasz": float(spec_ch),
            "spectral_runtime_s": float(t_spec),
            # DeepWalk
            "deepwalk_silhouette": float(dw_sil),
            "deepwalk_davies_bouldin": float(dw_dbi),
            "deepwalk_calinski_harabasz": float(dw_ch),
            "deepwalk_runtime_s": float(t_dw),
            # Node2Vec (p=1.0, q=1.0)
            "n2v_p1.0_q1.0_silhouette": float(n2v11_sil),
            "n2v_p1.0_q1.0_davies_bouldin": float(n2v11_dbi),
            "n2v_p1.0_q1.0_calinski_harabasz": float(n2v11_ch),
            "n2v_p1.0_q1.0_runtime_s": float(t_n2v11),
            # Node2Vec (p=0.5, q=1.0)
            "n2v_p0.5_q1.0_silhouette": float(n2v05_10_sil),
            "n2v_p0.5_q1.0_davies_bouldin": float(n2v05_10_dbi),
            "n2v_p0.5_q1.0_calinski_harabasz": float(n2v05_10_ch),
            "n2v_p0.5_q1.0_runtime_s": float(t_n2v05_10),
            # Node2Vec (p=0.5, q=2.0)
            "n2v_p0.5_q2.0_silhouette": float(n2v05_20_sil),
            "n2v_p0.5_q2.0_davies_bouldin": float(n2v05_20_dbi),
            "n2v_p0.5_q2.0_calinski_harabasz": float(n2v05_20_ch),
            "n2v_p0.5_q2.0_runtime_s": float(t_n2v05_20),
            # Node2Vec (p=1.0, q=0.5)
            "n2v_p1.0_q0.5_silhouette": float(n2v10_05_sil),
            "n2v_p1.0_q0.5_davies_bouldin": float(n2v10_05_dbi),
            "n2v_p1.0_q0.5_calinski_harabasz": float(n2v10_05_ch),
            "n2v_p1.0_q0.5_runtime_s": float(t_n2v10_05),
        }
        all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_excel(os.path.join(LINUX_DIR, "all_methods_results.xlsx"), index=False)
    with open(os.path.join(LINUX_DIR, "all_methods_results.json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, cls=NumpyEncoder)

    # Print summary table
    print_subheader("Summary Table (Silhouette Score)")
    print(f"  {'Course':<10} {'BoW':<10} {'PCA':<10} {'Spectral':<10} {'DeepWalk':<10} {'N2V(1,1)':<10} {'N2V(0.5,1)':<12} {'N2V(0.5,2)':<12} {'N2V(1,0.5)':<12}")
    print(f"  {'-'*90}")
    for _, row in df.iterrows():
        c = int(row['course'])
        print(f"  Course {c:<4} {row['bow_silhouette']:<10.4f} {row['pca_silhouette']:<10.4f} "
              f"{row['spectral_silhouette']:<10.4f} {row['deepwalk_silhouette']:<10.4f} "
              f"{row['n2v_p1.0_q1.0_silhouette']:<10.4f} {row['n2v_p0.5_q1.0_silhouette']:<12.4f} "
              f"{row['n2v_p0.5_q2.0_silhouette']:<12.4f} {row['n2v_p1.0_q0.5_silhouette']:<12.4f}")
    print(f"  {'-'*90}")
    print(f"  {'Average':<10} "
          f"{df['bow_silhouette'].mean():<10.4f} {df['pca_silhouette'].mean():<10.4f} "
          f"{df['spectral_silhouette'].mean():<10.4f} {df['deepwalk_silhouette'].mean():<10.4f} "
          f"{df['n2v_p1.0_q1.0_silhouette'].mean():<10.4f} {df['n2v_p0.5_q1.0_silhouette'].mean():<12.4f} "
          f"{df['n2v_p0.5_q2.0_silhouette'].mean():<12.4f} {df['n2v_p1.0_q0.5_silhouette'].mean():<12.4f}")

    # Print DBI summary
    print_subheader("Summary Table (Davies-Bouldin Index, lower is better)")
    print(f"  {'Course':<10} {'BoW':<10} {'PCA':<10} {'Spectral':<10} {'DeepWalk':<10} {'N2V(1,1)':<10} {'N2V(0.5,1)':<12}")
    print(f"  {'-'*70}")
    for _, row in df.iterrows():
        c = int(row['course'])
        print(f"  Course {c:<4} {row['bow_davies_bouldin']:<10.4f} {row['pca_davies_bouldin']:<10.4f} "
              f"{row['spectral_davies_bouldin']:<10.4f} {row['deepwalk_davies_bouldin']:<10.4f} "
              f"{row['n2v_p1.0_q1.0_davies_bouldin']:<10.4f} {row['n2v_p0.5_q1.0_davies_bouldin']:<12.4f}")
    print(f"  {'-'*70}")
    print(f"  {'Average':<10} "
          f"{df['bow_davies_bouldin'].mean():<10.4f} {df['pca_davies_bouldin'].mean():<10.4f} "
          f"{df['spectral_davies_bouldin'].mean():<10.4f} {df['deepwalk_davies_bouldin'].mean():<10.4f} "
          f"{df['n2v_p1.0_q1.0_davies_bouldin'].mean():<10.4f} {df['n2v_p0.5_q1.0_davies_bouldin'].mean():<12.4f}")

    return df


# ============================================================
# SECTION D: Node2Vec Sensitivity Analysis
# ============================================================
def run_node2vec_sensitivity():
    """Run Node2Vec with a grid of (p, q) values to characterize behavior."""
    print_header("SECTION D: Node2Vec Parameter Sensitivity")

    p_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    q_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    sensitivity_results = []

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, student_labels = read_class(filepath)
        if scm is None:
            continue

        print(f"\n  Course {file_idx} ({scm.shape[0]} students):")

        for p_val in p_values:
            for q_val in q_values:
                t0 = time.perf_counter()
                n2v_result = run_node2vec_pipeline(
                    filepath, p=p_val, q=q_val, seed=0
                )
                elapsed = time.perf_counter() - t0

                if n2v_result["success"]:
                    n2v_eval = evaluate_clustering(
                        n2v_result["embeddings"], n_clusters=2, random_state=0
                    )
                    sil = n2v_eval["kmeans"]["metrics"]["silhouette"]
                    dbi = n2v_eval["kmeans"]["metrics"]["davies_bouldin"]
                    ch = n2v_eval["kmeans"]["metrics"]["calinski_harabasz"]
                else:
                    sil = dbi = ch = np.nan

                sensitivity_results.append({
                    "course": file_idx,
                    "p": p_val,
                    "q": q_val,
                    "silhouette": float(sil),
                    "davies_bouldin": float(dbi),
                    "calinski_harabasz": float(ch),
                    "runtime_s": float(elapsed),
                })

        # Print heatmap-style for this course
        course_data = [r for r in sensitivity_results if r["course"] == file_idx]
        print(f"    {'p\\q':<8}", end="")
        for q_val in q_values:
            print(f"  q={q_val:<6}", end="")
        print()
        for p_val in p_values:
            print(f"    p={p_val:<4}", end="")
            for q_val in q_values:
                matching = [r for r in course_data if r["p"] == p_val and r["q"] == q_val]
                if matching:
                    print(f"  {matching[0]['silhouette']:<8.4f}", end="")
                else:
                    print(f"  {'N/A':<8}", end="")
            print()

    # Save results
    with open(os.path.join(LINUX_DIR, "node2vec_sensitivity.json"), "w") as f:
        json.dump(sensitivity_results, f, indent=2, cls=NumpyEncoder)

    # Compute average across courses for each (p, q)
    print_subheader("Average Silhouette Across All Courses")
    print(f"    {'p\\q':<8}", end="")
    for q_val in q_values:
        print(f"  q={q_val:<6}", end="")
    print()
    for p_val in p_values:
        print(f"    p={p_val:<4}", end="")
        for q_val in q_values:
            matching = [r for r in sensitivity_results if r["p"] == p_val and r["q"] == q_val]
            if matching:
                avg_sil = np.mean([r["silhouette"] for r in matching])
                print(f"  {avg_sil:<8.4f}", end="")
            else:
                print(f"  {'N/A':<8}", end="")
        print()

    return sensitivity_results


# ============================================================
# SECTION E: Node2Vec Validation — Walk-Level Comparison
# ============================================================
def validate_node2vec_walks():
    """Verify Node2Vec generates genuinely different walks from DeepWalk."""
    print_header("SECTION E: Node2Vec Validation")

    # Use Course 1 (largest)
    filepath = os.path.join(DATA_DIR, "1.txt")
    scm, _ = read_class(filepath)
    G = create_graph_from_bow(scm)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"density={nx.density(G):.4f}")

    # Generate walks with same seed
    seed = 42
    n_walks = 5
    walk_len = 10

    dw_walks = generate_random_walks(G, num_walks_per_node=n_walks,
                                      walk_length=walk_len, seed=seed)
    n2v_11_walks = generate_node2vec_walks(G, num_walks_per_node=n_walks,
                                            walk_length=walk_len, p=1.0, q=1.0, seed=seed)
    n2v_05_10_walks = generate_node2vec_walks(G, num_walks_per_node=n_walks,
                                               walk_length=walk_len, p=0.5, q=1.0, seed=seed)
    n2v_10_05_walks = generate_node2vec_walks(G, num_walks_per_node=n_walks,
                                               walk_length=walk_len, p=1.0, q=0.5, seed=seed)

    print(f"\n  Total walks: DeepWalk={len(dw_walks)}, N2V(1,1)={len(n2v_11_walks)}, "
          f"N2V(0.5,1)={len(n2v_05_10_walks)}, N2V(1,0.5)={len(n2v_10_05_walks)}")

    # Compare walks
    def walk_similarity(walks_a, walks_b):
        """Fraction of walks that are identical."""
        n = min(len(walks_a), len(walks_b))
        identical = sum(1 for i in range(n) if walks_a[i] == walks_b[i])
        return identical / n if n > 0 else 0

    def avg_return_rate(walks, G):
        """Average fraction of walks that return to previous node."""
        returns = 0
        total_steps = 0
        for walk in walks:
            for i in range(2, len(walk)):
                total_steps += 1
                if walk[i] == walk[i-2]:
                    returns += 1
        return returns / total_steps if total_steps > 0 else 0

    def avg_neighbor_overlap(walks, G):
        """Average Jaccard overlap between consecutive steps' neighborhoods."""
        overlaps = []
        for walk in walks:
            for i in range(1, len(walk) - 1):
                neighbors_prev = set(G.neighbors(walk[i-1]))
                neighbors_curr = set(G.neighbors(walk[i]))
                if neighbors_prev and neighbors_curr:
                    jaccard = len(neighbors_prev & neighbors_curr) / len(neighbors_prev | neighbors_curr)
                    overlaps.append(jaccard)
        return np.mean(overlaps) if overlaps else 0

    print_subheader("Walk Comparison Results")
    sim_11 = walk_similarity(dw_walks, n2v_11_walks)
    sim_05 = walk_similarity(dw_walks, n2v_05_10_walks)
    sim_10 = walk_similarity(dw_walks, n2v_10_05_walks)
    print(f"  Walk identity rate (DW vs N2V(1,1)): {sim_11:.3f}")
    print(f"  Walk identity rate (DW vs N2V(0.5,1)): {sim_05:.3f}")
    print(f"  Walk identity rate (DW vs N2V(1,0.5)): {sim_10:.3f}")

    # For complete graphs, p parameter still affects transition probabilities
    # even though q has no structural effect
    print_subheader("Node2Vec Walk Behavior Analysis")
    for name, walks in [("DeepWalk", dw_walks), ("N2V(1,1)", n2v_11_walks),
                         ("N2V(0.5,1)", n2v_05_10_walks), ("N2V(1,0.5)", n2v_10_05_walks)]:
        return_rate = avg_return_rate(walks, G)
        print(f"  {name}: return-to-previous rate = {return_rate:.4f}")

    # Generate longer walks to see p parameter effect more clearly
    print_subheader("Long Walks Analysis (100 walks, length=100)")
    long_dw = generate_random_walks(G, num_walks_per_node=100, walk_length=100, seed=42)
    long_n2v_05 = generate_node2vec_walks(G, num_walks_per_node=100, walk_length=100, p=0.5, q=1.0, seed=42)
    long_n2v_10_05 = generate_node2vec_walks(G, num_walks_per_node=100, walk_length=100, p=1.0, q=0.5, seed=42)

    for name, walks in [("DeepWalk", long_dw), ("N2V(0.5,1)", long_n2v_05), ("N2V(1,0.5)", long_n2v_10_05)]:
        return_rate = avg_return_rate(walks, G)
        print(f"  {name}: return-to-previous rate = {return_rate:.4f}")

    return {
        "walk_identity_rates": {
            "dw_vs_n2v_1_1": float(sim_11),
            "dw_vs_n2v_0_5_1": float(sim_05),
            "dw_vs_n2v_1_0_5": float(sim_10),
        },
        "return_rates": {
            "deepwalk": float(avg_return_rate(dw_walks, G)),
            "n2v_1_1": float(avg_return_rate(n2v_11_walks, G)),
            "n2v_0_5_1": float(avg_return_rate(n2v_05_10_walks, G)),
            "n2v_1_0_5": float(avg_return_rate(n2v_10_05_walks, G)),
        },
    }


# ============================================================
# SECTION F: Clustering Stability
# ============================================================
def run_stability_analysis():
    """Run clustering stability for DeepWalk and Node2Vec."""
    print_header("SECTION F: Clustering Stability Analysis")

    stability = {}

    for method_name, pipeline_fn, extra_kwargs in [
        ("DeepWalk", lambda fp, s: run_pipeline(fp, seed=s), {}),
        ("Node2Vec(1,1)", lambda fp, s: run_node2vec_pipeline(fp, p=1.0, q=1.0, seed=s), {}),
        ("Node2Vec(0.5,1)", lambda fp, s: run_node2vec_pipeline(fp, p=0.5, q=1.0, seed=s), {}),
    ]:
        stability[method_name] = {}
        print(f"\n  Method: {method_name}")

        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            silhouettes = []
            dbis = []
            chs = []

            for seed in range(NUM_SEEDS):
                result = pipeline_fn(filepath, seed)
                if result["success"]:
                    eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                    m = eval_r["kmeans"]["metrics"]
                    silhouettes.append(m["silhouette"])
                    dbis.append(m["davies_bouldin"])
                    chs.append(m["calinski_harabasz"])

            stability[method_name][str(file_idx)] = {
                "silhouette": {
                    "mean": float(np.mean(silhouettes)),
                    "std": float(np.std(silhouettes)),
                    "min": float(np.min(silhouettes)),
                    "max": float(np.max(silhouettes)),
                    "values": silhouettes,
                },
                "davies_bouldin": {
                    "mean": float(np.mean(dbis)),
                    "std": float(np.std(dbis)),
                },
                "calinski_harabasz": {
                    "mean": float(np.mean(chs)),
                    "std": float(np.std(chs)),
                },
            }
            print(f"    Course {file_idx}: Sil={np.mean(silhouettes):.4f} +/- {np.std(silhouettes):.4f}")

    # Save
    with open(os.path.join(LINUX_DIR, "stability_analysis.json"), "w") as f:
        json.dump(stability, f, indent=2, cls=NumpyEncoder)

    return stability


# ============================================================
# SECTION G: Statistical Analysis
# ============================================================
def run_statistical_analysis(main_results_df, stability):
    """Statistical analysis: Node2Vec vs DeepWalk, Node2Vec vs PCA, Node2Vec vs BoW."""
    print_header("SECTION G: Statistical Analysis")

    stat_results = {}

    # ── Per-course paired comparisons (using stability data) ──
    for comparison_name, method_a, method_b in [
        ("DeepWalk_vs_BoW", "DeepWalk", "BoW"),
        ("Node2Vec(1,1)_vs_BoW", "Node2Vec(1,1)", "BoW"),
        ("Node2Vec(0.5,1)_vs_BoW", "Node2Vec(0.5,1)", "BoW"),
        ("Node2Vec(1,1)_vs_DeepWalk", "Node2Vec(1,1)", "DeepWalk"),
        ("Node2Vec(0.5,1)_vs_DeepWalk", "Node2Vec(0.5,1)", "DeepWalk"),
        ("Node2Vec(1,1)_vs_PCA", "Node2Vec(1,1)", "PCA"),
        ("Node2Vec(0.5,1)_vs_PCA", "Node2Vec(0.5,1)", "PCA"),
        ("DeepWalk_vs_PCA", "DeepWalk", "PCA"),
    ]:
        print_subheader(f"{comparison_name}")
        course_stats = {}

        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            scm, _ = read_class(filepath)

            # Get paired distributions from stability runs
            if method_a in stability and str(file_idx) in stability[method_a]:
                scores_a = np.array(stability[method_a][str(file_idx)]["silhouette"]["values"])
            else:
                # For BoW/PCA: deterministic, but create paired with DW seeds
                if method_a == "BoW":
                    bow_eval = evaluate_clustering(scm.astype(float), n_clusters=2, random_state=0)
                    scores_a = np.full(len(stability.get("DeepWalk", {}).get(str(file_idx), {}).get("silhouette", {}).get("values", [0])),
                                       bow_eval["kmeans"]["metrics"]["silhouette"])
                elif method_a == "PCA":
                    pca_scores = []
                    for seed in range(NUM_SEEDS):
                        labels, reduced = run_pca_kmeans(scm.astype(float), n_clusters=2, n_components=2, random_state=seed)
                        pca_scores.append(compute_all_metrics(reduced, labels)["silhouette"])
                    scores_a = np.array(pca_scores)
                else:
                    continue

            if method_b in stability and str(file_idx) in stability[method_b]:
                scores_b = np.array(stability[method_b][str(file_idx)]["silhouette"]["values"])
            else:
                if method_b == "BoW":
                    bow_eval = evaluate_clustering(scm.astype(float), n_clusters=2, random_state=0)
                    scores_b = np.full(len(scores_a), bow_eval["kmeans"]["metrics"]["silhouette"])
                elif method_b == "PCA":
                    pca_scores = []
                    for seed in range(NUM_SEEDS):
                        labels, reduced = run_pca_kmeans(scm.astype(float), n_clusters=2, n_components=2, random_state=seed)
                        pca_scores.append(compute_all_metrics(reduced, labels)["silhouette"])
                    scores_b = np.array(pca_scores)
                else:
                    continue

            # Ensure same length
            n = min(len(scores_a), len(scores_b))
            scores_a, scores_b = scores_a[:n], scores_b[:n]

            # Wilcoxon test
            test = wilcoxon_signed_rank_test(scores_a, scores_b, alternative='greater')
            r = compute_effect_size_r(test.get("statistic", 0), test.get("n", 0))
            delta, delta_interp = compute_cliffs_delta(scores_a, scores_b)
            ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(scores_a - scores_b)

            course_stats[str(file_idx)] = {
                "mean_a": float(np.mean(scores_a)),
                "mean_b": float(np.mean(scores_b)),
                "wilcoxon_stat": float(test.get("statistic", 0)) if not np.isnan(test.get("statistic", 0)) else None,
                "wilcoxon_p": float(test.get("p_value", 0)) if not np.isnan(test.get("p_value", 0)) else None,
                "effect_size_r": float(r),
                "cliffs_delta": float(delta),
                "cliffs_interp": delta_interp,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n": n,
            }

            sig = "***" if test.get("p_value", 1) < 0.001 else "**" if test.get("p_value", 1) < 0.01 else "*" if test.get("p_value", 1) < 0.05 else "ns"
            print(f"    Course {file_idx}: {method_a}={np.mean(scores_a):.4f} vs {method_b}={np.mean(scores_b):.4f} "
                  f"(Δ={np.mean(scores_a)-scores_b.mean():.4f}, p={test.get('p_value', 1):.6f} {sig}, r={r:.3f})")

        # Aggregate across courses
        all_a = np.array([course_stats[str(c)]["mean_a"] for c in FILE_INDICES if str(c) in course_stats])
        all_b = np.array([course_stats[str(c)]["mean_b"] for c in FILE_INDICES if str(c) in course_stats])

        if len(all_a) >= 3:
            agg_test = wilcoxon_signed_rank_test(all_a, all_b, alternative='greater')
            agg_r = compute_effect_size_r(agg_test.get("statistic", 0), agg_test.get("n", 0))
            agg_delta, agg_delta_interp = compute_cliffs_delta(all_a, all_b)
            agg_ci_mean, agg_ci_lower, agg_ci_upper = compute_bootstrap_ci(all_a - all_b)

            course_stats["aggregate"] = {
                "mean_a": float(np.mean(all_a)),
                "mean_b": float(np.mean(all_b)),
                "wilcoxon_p": float(agg_test.get("p_value", 0)),
                "effect_size_r": float(agg_r),
                "cliffs_delta": float(agg_delta),
                "cliffs_interp": agg_delta_interp,
                "ci_lower": float(agg_ci_lower),
                "ci_upper": float(agg_ci_upper),
                "n": len(all_a),
            }
            sig = "***" if agg_test.get("p_value", 1) < 0.001 else "ns"
            print(f"    AGGREGATE ({len(all_a)} courses): {method_a}={np.mean(all_a):.4f} vs {method_b}={np.mean(all_b):.4f} "
                  f"(Δ={np.mean(all_a-all_b):.4f}, p={agg_test.get('p_value', 1):.6f} {sig})")

        stat_results[comparison_name] = course_stats

    with open(os.path.join(LINUX_DIR, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2, cls=NumpyEncoder)

    return stat_results


# ============================================================
# SECTION H: Generate All Figures (PDF)
# ============================================================
def generate_figures(main_df, sensitivity_results, stability):
    """Generate publication-quality PDF figures."""
    print_header("SECTION H: Generating Figures")

    setup_pub_style()
    courses = sorted(main_df["course"].values)
    course_labels = [f"Course {int(c)}" for c in courses]

    # ── Figure 1: Main Silhouette Comparison (all 5 methods) ─────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = ["BoW+KMeans", "PCA+KMeans", "Spectral", "DeepWalk-SSP", "Node2Vec-SSP\n(p=1.0, q=1.0)"]
    method_colors = [COLORS['bow'], COLORS['pca_kmeans'], COLORS['spectral'], COLORS['deepwalk'], COLORS['node2vec']]
    x = np.arange(len(courses))
    width = 0.15

    for i, (method, color) in enumerate(zip(methods, method_colors)):
        col_map = {
            "BoW+KMeans": "bow_silhouette",
            "PCA+KMeans": "pca_silhouette",
            "Spectral": "spectral_silhouette",
            "DeepWalk-SSP\n(p=1.0, q=1.0)": "deepwalk_silhouette",
            "Node2Vec-SSP\n(p=1.0, q=1.0)": "n2v_p1.0_q1.0_silhouette",
        }
        vals = [main_df[main_df["course"] == c][col_map[method]].values[0] for c in courses]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=5.5)

    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score (↑ higher is better)')
    ax.set_title('Clustering Quality Comparison Across Six Courses')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "fig1_silhouette_comparison.pdf")
    print("  Saved: fig1_silhouette_comparison.pdf")

    # ── Figure 2: DBI Comparison ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    methods_dbi = ["BoW+KMeans", "PCA+KMeans", "Spectral", "DeepWalk-SSP", "Node2Vec-SSP\n(p=1.0, q=1.0)"]
    col_map_dbi = {
        "BoW+KMeans": "bow_davies_bouldin",
        "PCA+KMeans": "pca_davies_bouldin",
        "Spectral": "spectral_davies_bouldin",
        "DeepWalk-SSP": "deepwalk_davies_bouldin",
        "Node2Vec-SSP\n(p=1.0, q=1.0)": "n2v_p1.0_q1.0_davies_bouldin",
    }
    for i, (method, color) in enumerate(zip(methods_dbi, method_colors)):
        vals = [main_df[main_df["course"] == c][col_map_dbi[method]].values[0] for c in courses]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Course')
    ax.set_ylabel('Davies-Bouldin Index (↓ lower is better)')
    ax.set_title('Davies-Bouldin Index Across Six Courses')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "fig2_dbi_comparison.pdf")
    print("  Saved: fig2_dbi_comparison.pdf")

    # ── Figure 3: CH Index Comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    col_map_ch = {
        "BoW+KMeans": "bow_calinski_harabasz",
        "PCA+KMeans": "pca_calinski_harabasz",
        "Spectral": "spectral_calinski_harabasz",
        "DeepWalk-SSP": "deepwalk_calinski_harabasz",
        "Node2Vec-SSP\n(p=1.0, q=1.0)": "n2v_p1.0_q1.0_calinski_harabasz",
    }
    for i, (method, color) in enumerate(zip(methods_dbi, method_colors)):
        vals = [main_df[main_df["course"] == c][col_map_ch[method]].values[0] for c in courses]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Course')
    ax.set_ylabel('Calinski-Harabasz Index (↑ higher is better)')
    ax.set_title('Calinski-Harabasz Index Across Six Courses')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "fig3_ch_comparison.pdf")
    print("  Saved: fig3_ch_comparison.pdf")

    # ── Figure 4: Node2Vec Parameter Sensitivity Heatmap ─────────────────────
    p_values = sorted(set(r["p"] for r in sensitivity_results))
    q_values = sorted(set(r["q"] for r in sensitivity_results))

    # Average silhouette across courses
    heatmap_data = np.zeros((len(p_values), len(q_values)))
    for i, p_val in enumerate(p_values):
        for j, q_val in enumerate(q_values):
            matching = [r["silhouette"] for r in sensitivity_results if r["p"] == p_val and r["q"] == q_val]
            heatmap_data[i, j] = np.mean(matching) if matching else np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(q_values)))
    ax.set_xticklabels([f"{q:.2f}" for q in q_values])
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels([f"{p:.2f}" for p in q_values])
    ax.set_xlabel('q (BFS↔DFS parameter)')
    ax.set_ylabel('p (return parameter)')
    ax.set_title('Node2Vec Silhouette Score: Parameter Sensitivity\n(Average across 6 courses)')

    # Add text annotations
    for i in range(len(p_values)):
        for j in range(len(q_values)):
            if not np.isnan(heatmap_data[i, j]):
                ax.text(j, i, f'{heatmap_data[i,j]:.3f}', ha='center', va='center', fontsize=10,
                       color='white' if heatmap_data[i,j] > 0.5 else 'black')

    fig.colorbar(im, ax=ax, label='Silhouette Score')
    save_fig(fig, "fig4_node2vec_sensitivity.pdf")
    print("  Saved: fig4_node2vec_sensitivity.pdf")

    # ── Figure 5: Silhouette Score vs Embedding Dimension ────────────────────
    # Load the old sensitivity data if available
    try:
        with open(os.path.join(RESULTS_DIR, "exp_A_hyperparameter_sensitivity.json")) as f:
            old_sensitivity = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 5))
        dims = sorted(int(k) for k in old_sensitivity["embedding_dim"].keys())
        sils = [old_sensitivity["embedding_dim"][str(d)]["silhouette"] for d in dims]
        ax.plot(dims, sils, marker='o', linewidth=2, markersize=8, color=COLORS['deepwalk'])
        ax.set_xlabel('Embedding Dimension (d)')
        ax.set_ylabel('Mean Silhouette Score')
        ax.set_title('Impact of Embedding Dimension on Clustering Quality')
        ax.set_xticks(dims)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.axhline(y=0.50, color='green', linestyle='--', alpha=0.4, label='Good threshold')
        ax.legend(framealpha=0.9)
        save_fig(fig, "fig5_silhouette_vs_dim.pdf")
        print("  Saved: fig5_silhouette_vs_dim.pdf")
    except:
        print("  Skipped fig5 (no sensitivity data)")

    # ── Figure 6: Seed Stability Boxplot ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, (method_name, color) in enumerate([
        ("DeepWalk", COLORS['deepwalk']),
        ("Node2Vec(1,1)", COLORS['node2vec']),
        ("Node2Vec(0.5,1)", "#FF5722"),
    ]):
        ax = axes[ax_idx]
        box_data = []
        labels = []
        for c in courses:
            if method_name in stability and str(int(c)) in stability[method_name]:
                vals = stability[method_name][str(int(c))]["silhouette"]["values"]
                box_data.append(vals)
                labels.append(f"C{int(c)}")
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        ax.set_ylabel('Silhouette Score')
        ax.set_title(f'{method_name}\nSeed Stability (n={NUM_SEEDS})')
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig6_seed_stability.pdf")
    print("  Saved: fig6_seed_stability.pdf")

    # ── Figure 7: Runtime Comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_rt = ['BoW+KMeans', 'PCA+KMeans', 'Spectral', 'DeepWalk-SSP', 'Node2Vec-SSP']
    runtime_cols = ['bow_runtime_s', 'pca_runtime_s', 'spectral_runtime_s', 'deepwalk_runtime_s', 'n2v_p1.0_q1.0_runtime_s']
    for i, (method, col) in enumerate(zip(methods_rt, runtime_cols)):
        vals = main_df[col].values
        ax.bar(x + (i-2)*0.15, vals, 0.15, label=method, color=method_colors[i], alpha=0.85)
    ax.set_xlabel('Course')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "fig7_runtime_comparison.pdf")
    print("  Saved: fig7_runtime_comparison.pdf")

    # ── Figure 8: Node2Vec Configurations Comparison ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    n2v_configs = [
        ("N2V(1.0,1.0)", "n2v_p1.0_q1.0_silhouette", COLORS['node2vec']),
        ("N2V(0.5,1.0)", "n2v_p0.5_q1.0_silhouette", "#FF5722"),
        ("N2V(0.5,2.0)", "n2v_p0.5_q2.0_silhouette", "#9C27B0"),
        ("N2V(1.0,0.5)", "n2v_p1.0_q0.5_silhouette", "#FF9800"),
        ("DeepWalk", "deepwalk_silhouette", COLORS['deepwalk']),
    ]
    for i, (label, col, color) in enumerate(n2v_configs):
        vals = [main_df[main_df["course"] == c][col].values[0] for c in courses]
        ax.plot(x, vals, marker='o', label=label, color=color, linewidth=2, markersize=7)

    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score (↑)')
    ax.set_title('Node2Vec Configuration Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(course_labels)
    ax.legend(loc='best', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "fig8_node2vec_configs.pdf")
    print("  Saved: fig8_node2vec_configs.pdf")

    # ── Figure 9: Average Method Comparison (summary bar) ────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_methods = ['BoW\n+KMeans', 'PCA\n+KMeans', 'Spectral\nClustering', 'DeepWalk\n-SSP', 'Node2Vec\n-SSP (1.0,1.0)', 'Node2Vec\n-SSP (0.5,1.0)']
    avg_cols = ['bow_silhouette', 'pca_silhouette', 'spectral_silhouette',
                'deepwalk_silhouette', 'n2v_p1.0_q1.0_silhouette', 'n2v_p0.5_q1.0_silhouette']
    avg_colors = [COLORS['bow'], COLORS['pca_kmeans'], COLORS['spectral'],
                  COLORS['deepwalk'], COLORS['node2vec'], "#FF5722"]
    avg_vals = [main_df[col].mean() for col in avg_cols]
    avg_stds = [main_df[col].std() for col in avg_cols]

    bars = ax.bar(range(len(avg_methods)), avg_vals, yerr=avg_stds, capsize=4,
                  color=avg_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, avg_vals):
        h = bar.get_height()
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Silhouette Score (↑)')
    ax.set_title('Average Clustering Quality Across All Six Courses')
    ax.set_xticks(range(len(avg_methods)))
    ax.set_xticklabels(avg_methods, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0)
    save_fig(fig, "fig9_average_comparison.pdf")
    print("  Saved: fig9_average_comparison.pdf")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    total_start = time.time()

    print_header("DEEPWALK-SSP: COMPLETE LINUX RE-EXPERIMENT")
    print(f"  Output directory: {LINUX_DIR}")
    print(f"  Figures directory: {LINUX_FIG_DIR}")

    # A: Environment
    env_info = record_environment()

    # B: Graph analysis
    graph_stats = analyze_graphs()

    # C: Run all methods
    main_df = run_all_methods()

    # D: Node2Vec sensitivity
    sensitivity = run_node2vec_sensitivity()

    # E: Node2Vec validation
    validation = validate_node2vec_walks()

    # F: Stability analysis
    stability = run_stability_analysis()

    # G: Statistical analysis
    stat_results = run_statistical_analysis(main_df, stability)

    # H: Figures
    generate_figures(main_df, sensitivity, stability)

    total_time = time.time() - total_start
    print_header(f"RE-EXPERIMENT COMPLETE (total: {total_time:.1f}s / {total_time/60:.1f} min)")
    print(f"\n  All results saved to: {LINUX_DIR}")
    print(f"  All figures saved to: {LINUX_FIG_DIR}")
