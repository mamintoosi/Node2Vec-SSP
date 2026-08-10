# -*- coding: utf-8 -*-
"""
Add Two Baselines to DeepWalk-SSP Experiments
================================================
Implements:
  1. PCA + KMeans on the conventional Student-Course representation (d=2)
  2. Spectral Clustering on the student co-enrollment graph

Reuses the existing data, graph construction, and evaluation framework.
Results are saved to results/ for later paper integration.

Usage:
    python -m experiments.add_baselines
"""

import os
import sys
import time
import json
import warnings
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import (
    FILE_INDICES, DATA_DIR, RESULTS_DIR, FIGURES_DIR, COLORS,
    PUBLICATION_STYLE, DEFAULT_PARAMS
)
from experiments.core import read_class, create_graph_from_bow, set_seed
from experiments.evaluation import (
    compute_all_metrics, run_pca_kmeans, run_spectral, evaluate_clustering
)
from experiments.plotting import setup_style, save_fig


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n  --- {title} ---")


# ============================================================
# Baseline 1: PCA + KMeans on Student-Course Representation
# ============================================================
def experiment_pca_kmeans():
    """
    Apply PCA to reduce Student-Course matrix to d=2, then KMeans.
    This tests whether DeepWalk's improvement is simply due to
    dimensionality reduction.
    """
    print_header("BASELINE 1: PCA + KMeans on Student-Course Representation")

    n_components = DEFAULT_PARAMS["vector_size"]  # d=2
    n_clusters = DEFAULT_PARAMS["n_clusters"]
    seed = 0

    results = []

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, student_labels = read_class(filepath)
        if scm is None:
            print(f"  Course {file_idx}: FAILED to load")
            continue

        n_students = scm.shape[0]
        n_courses = scm.shape[1]

        print(f"\n  Course {file_idx}: {n_students} students, {n_courses} courses")
        print(f"    Student-Course matrix: {scm.shape}")

        # PCA + KMeans
        t0 = time.perf_counter()
        pca_labels, pca_reduced = run_pca_kmeans(
            scm.astype(float),
            n_clusters=n_clusters,
            n_components=n_components,
            random_state=seed,
        )
        t_pca = time.perf_counter() - t0

        # Evaluate clustering on PCA-reduced data
        metrics = compute_all_metrics(pca_reduced, pca_labels)

        print(f"    PCA({n_components}d) + KMeans: Silhouette={metrics['silhouette']:.3f}, "
              f"DBI={metrics['davies_bouldin']:.3f}, CH={metrics['calinski_harabasz']:.1f} "
              f"(time: {t_pca:.3f}s)")

        # Also run direct KMeans on original BoW for comparison
        bow_kmeans = evaluate_clustering(
            scm.astype(float), n_clusters=n_clusters, random_state=seed
        )
        bow_sil = bow_kmeans["kmeans"]["metrics"]["silhouette"]

        # Run DeepWalk for comparison
        from experiments.core import run_pipeline
        dw_result = run_pipeline(filepath, seed=seed)
        if dw_result["success"]:
            dw_eval = evaluate_clustering(
                dw_result["embeddings"], n_clusters=n_clusters, random_state=seed
            )
            dw_sil = dw_eval["kmeans"]["metrics"]["silhouette"]
        else:
            dw_sil = np.nan

        results.append({
            "course": file_idx,
            "n_students": n_students,
            "n_courses": n_courses,
            "pca_kmeans_silhouette": float(metrics["silhouette"]),
            "pca_kmeans_davies_bouldin": float(metrics["davies_bouldin"]),
            "pca_kmeans_calinski_harabasz": float(metrics["calinski_harabasz"]),
            "pca_kmeans_wcss": float(metrics["wcss"]),
            "bow_kmeans_silhouette": float(bow_sil),
            "deepwalk_silhouette": float(dw_sil),
            "pca_runtime_s": float(t_pca),
        })

        print(f"    Direct KMeans on BoW:  Silhouette={bow_sil:.3f}")
        print(f"    DeepWalk-SSP:          Silhouette={dw_sil:.3f}")
        delta_vs_bow = metrics["silhouette"] - bow_sil
        delta_vs_dw = dw_sil - metrics["silhouette"]
        print(f"    PCA+KMeans vs BoW:     {delta_vs_bow:+.3f}")
        print(f"    DeepWalk vs PCA+KMeans: {delta_vs_dw:+.3f}")

    # Summary
    print_subheader("PCA + KMeans Summary")
    df = pd.DataFrame(results)
    avg_pca = df["pca_kmeans_silhouette"].mean()
    avg_bow = df["bow_kmeans_silhouette"].mean()
    avg_dw = df["deepwalk_silhouette"].mean()
    print(f"  Average Silhouette (PCA+KMeans): {avg_pca:.3f}")
    print(f"  Average Silhouette (BoW+KMeans): {avg_bow:.3f}")
    print(f"  Average Silhouette (DeepWalk):   {avg_dw:.3f}")
    print(f"  PCA+KMeans improvement over BoW: {avg_pca - avg_bow:+.3f} "
          f"({(avg_pca - avg_bow) / avg_bow * 100:+.1f}%)")
    print(f"  DeepWalk improvement over PCA:   {avg_dw - avg_pca:+.3f} "
          f"({(avg_dw - avg_pca) / avg_pca * 100:+.1f}%)")

    return df


# ============================================================
# Baseline 2: Spectral Clustering on Co-enrollment Graph
# ============================================================
def experiment_spectral_clustering():
    """
    Apply Spectral Clustering directly to the student co-enrollment graph.
    This tests whether DeepWalk's learned embeddings provide value beyond
    what spectral methods can capture from the raw graph structure.
    """
    print_header("BASELINE 2: Spectral Clustering on Co-enrollment Graph")

    n_clusters = DEFAULT_PARAMS["n_clusters"]
    seed = 0

    results = []

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, student_labels = read_class(filepath)
        if scm is None:
            print(f"  Course {file_idx}: FAILED to load")
            continue

        n_students = scm.shape[0]

        # Build co-enrollment graph
        t0 = time.perf_counter()
        G = create_graph_from_bow(scm)
        t_graph = time.perf_counter() - t0

        # Convert graph to adjacency matrix for spectral clustering
        adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))

        print(f"\n  Course {file_idx}: {n_students} students, {G.number_of_edges()} edges")
        print(f"    Adjacency matrix: {adj_matrix.shape}")

        # Spectral Clustering
        t0 = time.perf_counter()
        spectral_labels = run_spectral(
            adj_matrix,
            n_clusters=n_clusters,
            random_state=seed,
        )
        t_spectral = time.perf_counter() - t0

        # Evaluate on the original feature space (BoW) for fair comparison
        # Spectral clustering produces labels; evaluate on BoW features
        metrics_bow = compute_all_metrics(scm.astype(float), spectral_labels)

        print(f"    Spectral Clustering: Silhouette (on BoW)={metrics_bow['silhouette']:.3f}, "
              f"DBI={metrics_bow['davies_bouldin']:.3f}, CH={metrics_bow['calinski_harabasz']:.1f} "
              f"(time: {t_spectral:.3f}s)")

        # Run DeepWalk for comparison
        from experiments.core import run_pipeline
        dw_result = run_pipeline(filepath, seed=seed)
        if dw_result["success"]:
            dw_eval = evaluate_clustering(
                dw_result["embeddings"], n_clusters=n_clusters, random_state=seed
            )
            dw_sil = dw_eval["kmeans"]["metrics"]["silhouette"]
        else:
            dw_sil = np.nan

        # Also run KMeans on BoW for comparison
        bow_eval = evaluate_clustering(scm.astype(float), n_clusters=n_clusters, random_state=seed)
        bow_sil = bow_eval["kmeans"]["metrics"]["silhouette"]

        results.append({
            "course": file_idx,
            "n_students": n_students,
            "n_edges": G.number_of_edges(),
            "spectral_silhouette_bow": float(metrics_bow["silhouette"]),
            "spectral_davies_bouldin": float(metrics_bow["davies_bouldin"]),
            "spectral_calinski_harabasz": float(metrics_bow["calinski_harabasz"]),
            "spectral_wcss": float(metrics_bow["wcss"]),
            "bow_kmeans_silhouette": float(bow_sil),
            "deepwalk_silhouette": float(dw_sil),
            "spectral_runtime_s": float(t_spectral),
        })

        print(f"    BoW + KMeans:           Silhouette={bow_sil:.3f}")
        print(f"    DeepWalk + KMeans:      Silhouette={dw_sil:.3f}")
        print(f"    Spectral vs BoW+KMeans: {metrics_bow['silhouette'] - bow_sil:+.3f}")
        print(f"    DeepWalk vs Spectral:   {dw_sil - metrics_bow['silhouette']:+.3f}")

    # Summary
    print_subheader("Spectral Clustering Summary")
    df = pd.DataFrame(results)
    avg_spec = df["spectral_silhouette_bow"].mean()
    avg_bow = df["bow_kmeans_silhouette"].mean()
    avg_dw = df["deepwalk_silhouette"].mean()
    print(f"  Average Silhouette (Spectral):   {avg_spec:.3f}")
    print(f"  Average Silhouette (BoW+KMeans): {avg_bow:.3f}")
    print(f"  Average Silhouette (DeepWalk):   {avg_dw:.3f}")
    print(f"  Spectral improvement over BoW:   {avg_spec - avg_bow:+.3f} "
          f"({(avg_spec - avg_bow) / avg_bow * 100:+.1f}%)")
    print(f"  DeepWalk improvement over Spectral: {avg_dw - avg_spec:+.3f} "
          f"({(avg_dw - avg_spec) / avg_spec * 100:+.1f}%)")

    return df


# ============================================================
# Generate Combined Comparison Figure
# ============================================================
def generate_comparison_figure(pca_df, spectral_df, filename="baseline_comparison.png"):
    """
    Bar chart comparing all baselines: BoW+KMeans, PCA+KMeans, Spectral, DeepWalk+KMeans.
    """
    setup_style("publication")

    courses = sorted(pca_df["course"].values)

    # Build data arrays
    bow_sils = [pca_df[pca_df["course"] == c]["bow_kmeans_silhouette"].values[0] for c in courses]
    pca_sils = [pca_df[pca_df["course"] == c]["pca_kmeans_silhouette"].values[0] for c in courses]
    spec_sils = [spectral_df[spectral_df["course"] == c]["spectral_silhouette_bow"].values[0] for c in courses]
    dw_sils = [pca_df[pca_df["course"] == c]["deepwalk_silhouette"].values[0] for c in courses]

    x = np.arange(len(courses))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - 1.5 * width, bow_sils, width, label='BoW + KMeans',
                   color=COLORS['bow'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x - 0.5 * width, pca_sils, width, label='PCA + KMeans',
                   color=COLORS['pca_kmeans'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + 0.5 * width, spec_sils, width, label='Spectral Clustering',
                   color=COLORS['spectral'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars4 = ax.bar(x + 1.5 * width, dw_sils, width, label='DeepWalk-SSP + KMeans',
                   color=COLORS['deepwalk'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score ($\\uparrow$ higher is better)')
    ax.set_title('Baseline Comparison: Clustering Quality Across Six Courses')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    return save_fig(fig, filename)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    total_start = time.time()

    # Run Baseline 1: PCA + KMeans
    pca_df = experiment_pca_kmeans()

    # Save results
    pca_df.to_excel(os.path.join(RESULTS_DIR, "baseline_pca_kmeans.xlsx"), index=False)
    pca_df.to_json(os.path.join(RESULTS_DIR, "baseline_pca_kmeans.json"),
                    orient="records", indent=2)
    print(f"\n  Saved: baseline_pca_kmeans.xlsx and .json")

    # Run Baseline 2: Spectral Clustering
    spectral_df = experiment_spectral_clustering()

    # Save results
    spectral_df.to_excel(os.path.join(RESULTS_DIR, "baseline_spectral.xlsx"), index=False)
    spectral_df.to_json(os.path.join(RESULTS_DIR, "baseline_spectral.json"),
                         orient="records", indent=2)
    print(f"\n  Saved: baseline_spectral.xlsx and .json")

    # Generate comparison figure
    print_header("Generating Comparison Figure")
    generate_comparison_figure(pca_df, spectral_df)
    print("  Saved: baseline_comparison.png")

    # ── Final Summary ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print_header("BASELINES COMPLETE")
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Combined summary table
    print_subheader("Combined Results (Silhouette Score, KMeans, seed=0)")
    print(f"\n  {'Course':<10} {'BoW+KM':<10} {'PCA+KM':<10} {'Spectral':<10} {'DeepWalk':<10}")
    print(f"  {'-'*50}")

    for c in FILE_INDICES:
        bow = pca_df[pca_df["course"] == c]["bow_kmeans_silhouette"].values[0]
        pca = pca_df[pca_df["course"] == c]["pca_kmeans_silhouette"].values[0]
        spec = spectral_df[spectral_df["course"] == c]["spectral_silhouette_bow"].values[0]
        dw = pca_df[pca_df["course"] == c]["deepwalk_silhouette"].values[0]
        print(f"  Course {c:<4} {bow:<10.3f} {pca:<10.3f} {spec:<10.3f} {dw:<10.3f}")

    print(f"  {'-'*50}")
    print(f"  {'Average':<10} "
          f"{pca_df['bow_kmeans_silhouette'].mean():<10.3f} "
          f"{pca_df['pca_kmeans_silhouette'].mean():<10.3f} "
          f"{spectral_df['spectral_silhouette_bow'].mean():<10.3f} "
          f"{pca_df['deepwalk_silhouette'].mean():<10.3f}")

    print(f"\n  Generated files in {RESULTS_DIR}:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("baseline_"):
            print(f"    {f}")

    print(f"\n  Figures in {FIGURES_DIR}:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if "baseline" in f:
            print(f"    {f}")

    print()
