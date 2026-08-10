# -*- coding: utf-8 -*-
"""
DeepWalk-SSP: Complete Experiment Suite
========================================
Runs all experiments (Steps 2, 4A-4F, 5, 7), statistical analysis,
improvements, and generates all tables, figures, and reports.

Usage:
    python -m experiments.run_experiments

All experiments use fixed random seeds (seed=0) for reproducibility.
Seed stability experiments use 20 seeds (0..19) to report mean ± std.
"""

import os
import sys
import time
import json
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"

# ── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.config import *
from experiments.core import (
    read_class, create_graph_from_bow, generate_random_walks,
    train_word2vec, run_pipeline, load_existing_results
)
from experiments.evaluation import (
    evaluate_clustering, compute_clustering_stability,
    reduce_pca, reduce_tsne, run_pca_kmeans
)
from experiments.plotting import (
    plot_silhouette_comparison_bar, plot_silhouette_vs_vectorsize,
    plot_clustering_methods_comparison, plot_embedding_visualization,
    plot_hyperparameter_sensitivity, plot_runtime_analysis,
    plot_seed_stability, plot_statistical_comparison,
    setup_style, save_fig
)
from experiments.stats import (
    wilcoxon_signed_rank_test, compute_effect_size_r,
    compute_cliffs_delta, compute_bootstrap_ci
)

import networkx as nx
from sklearn.metrics import silhouette_score


# ── Custom JSON encoder for NumPy types ──────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Helper functions ─────────────────────────────────────────────────────────
def set_global_seed(seed=0):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(0)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n  --- {title} ---")


# ============================================================
# STEP 2: Reproduce Existing Experiments
# ============================================================
def reproduce_existing():
    """Reproduce original experiments from the paper."""
    print_header("STEP 2: Reproducing Existing Experiments")

    # Load existing results
    df, graphs = load_existing_results(RESULTS_DIR)

    if df is None:
        print("  ERROR: No existing results found in results/df.xlsx")
        return None, None

    print(f"  Loaded {len(df)} rows from df.xlsx")
    print(f"  Loaded {len(graphs)} graphs from graphs.pkl")

    # Run pipeline for each file and vector size
    all_results = []
    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, labels = read_class(filepath)

        if scm is None:
            continue

        print(f"\n  Course {file_idx}: {scm.shape[0]} students, {scm.shape[1]} courses")

        for vs in sorted(df['vector_size'].unique()):
            result = run_pipeline(filepath, vector_size=int(vs), seed=0)

            if result["success"]:
                # Evaluate clustering
                eval_results = evaluate_clustering(
                    result["embeddings"], n_clusters=2, random_state=0
                )

                all_results.append({
                    'file_index': file_idx,
                    'vector_size': vs,
                    'num_students': result["num_students"],
                    'BoW.silhouette_score': float(np.nan),
                    'embeddings.silhouette_score': eval_results["kmeans"]["metrics"]["silhouette"],
                    'embeddings.davies_bouldin': eval_results["kmeans"]["metrics"]["davies_bouldin"],
                    'embeddings.calinski_harabasz': eval_results["kmeans"]["metrics"]["calinski_harabasz"],
                })

    if not all_results:
        print("  WARNING: No results generated")
        return df, graphs

    results_df = pd.DataFrame(all_results)

    # Compare with existing results
    print_subheader("Comparison with Existing Results")
    for file_idx in FILE_INDICES:
        existing = df[df['file_index'] == file_idx]
        new = results_df[results_df['file_index'] == file_idx]

        if len(existing) > 0 and len(new) > 0:
            # Get the row matching d=2 (vector_size=2)
            existing_row = existing[existing['vector_size'] == 2]
            new_row = new[new['vector_size'] == 2]

            if len(existing_row) > 0 and len(new_row) > 0:
                old_sil = existing_row['embeddings.silhouette_score'].values[0]
                new_sil = new_row['embeddings.silhouette_score'].values[0]
                bow_sil = existing_row['BoW.silhouette_score'].values[0]
                print(f"  Course {file_idx}: BoW={bow_sil:.3f}, DeepWalk(existing)={old_sil:.3f}, DeepWalk(reproduced)={new_sil:.3f}")

    # Generate reproduction figures
    print_subheader("Generating Figures")

    # Silhouette comparison bar chart
    plot_silhouette_comparison_bar(
        df, vector_size=2,
        filename="repro_silhouette_bar.png"
    )
    print("  Saved: repro_silhouette_bar.png")

    # Silhouette vs vector size
    plot_silhouette_vs_vectorsize(
        df, filename="repro_silhouette_vs_d.png"
    )
    print("  Saved: repro_silhouette_vs_d.png")

    # Summary statistics
    print_subheader("Summary Statistics (d=2)")
    d2_rows = df[df['vector_size'] == 2]
    if len(d2_rows) > 0:
        bow_mean = d2_rows['BoW.silhouette_score'].mean()
        dw_mean = d2_rows['embeddings.silhouette_score'].mean()
        improvement = ((dw_mean - bow_mean) / bow_mean * 100) if bow_mean != 0 else 0
        print(f"  BoW Silhouette (mean):      {bow_mean:.3f}")
        print(f"  DeepWalk Silhouette (mean): {dw_mean:.3f}")
        print(f"  Improvement:                {improvement:+.1f}%")

    # Save reproduced results
    results_df.to_excel(os.path.join(RESULTS_DIR, "df_reproduced.xlsx"), index=False)

    return df, graphs


# ============================================================
# STEP 4A: Hyperparameter Sensitivity Analysis
# ============================================================
def experiment_hyperparameter_sensitivity(df, all_graphs):
    """Analyze sensitivity to DeepWalk hyperparameters."""
    print_header("STEP 4A: Hyperparameter Sensitivity Analysis")

    results = {"embedding_dim": {}, "walk_length": {}, "num_walks": {}, "window_size": {}}

    # A.1: Embedding Dimension
    print_subheader("A.1: Embedding Dimension Sensitivity")
    for vs in VECTOR_SIZES:
        scores = []
        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            result = run_pipeline(filepath, vector_size=int(vs), seed=0)
            if result["success"]:
                eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                scores.append(eval_r["kmeans"]["metrics"]["silhouette"])
        mean_s = np.nanmean(scores) if scores else 0
        results["embedding_dim"][str(vs)] = {"silhouette": float(mean_s)}
        print(f"  d={vs:>2}: silhouette={mean_s:.3f}")

    # A.2: Walk Length
    print_subheader("A.2: Walk Length Sensitivity")
    for wl in WALK_LENGTHS:
        scores = []
        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            result = run_pipeline(filepath, walk_length=int(wl), seed=0)
            if result["success"]:
                eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                scores.append(eval_r["kmeans"]["metrics"]["silhouette"])
        mean_s = np.nanmean(scores) if scores else 0
        results["walk_length"][str(wl)] = {"silhouette": float(mean_s)}
        print(f"  t={wl:>2}: silhouette={mean_s:.3f}")

    # A.3: Number of Walks
    print_subheader("A.3: Number of Walks Sensitivity")
    for nw in NUM_WALKS_RANGE:
        scores = []
        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            result = run_pipeline(filepath, num_walks=int(nw), seed=0)
            if result["success"]:
                eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                scores.append(eval_r["kmeans"]["metrics"]["silhouette"])
        mean_s = np.nanmean(scores) if scores else 0
        results["num_walks"][str(nw)] = {"silhouette": float(mean_s)}
        print(f"  gamma={nw:>3}: silhouette={mean_s:.3f}")

    # A.4: Context Window Size
    print_subheader("A.4: Context Window Size Sensitivity")
    for ws in WINDOW_SIZES:
        scores = []
        for file_idx in FILE_INDICES:
            filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
            result = run_pipeline(filepath, window=int(ws), seed=0)
            if result["success"]:
                eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                scores.append(eval_r["kmeans"]["metrics"]["silhouette"])
        mean_s = np.nanmean(scores) if scores else 0
        results["window_size"][str(ws)] = {"silhouette": float(mean_s)}
        print(f"  w={ws:>2}: silhouette={mean_s:.3f}")

    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_A_hyperparameter_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Generate plots
    print_subheader("Generating Sensitivity Plots")
    for param_key, xlabel, title in [
        ("embedding_dim", "Embedding Dimension (d)", "Embedding Dimension Sensitivity"),
        ("walk_length", "Walk Length (t)", "Walk Length Sensitivity"),
        ("num_walks", "Number of Walks (gamma)", "Number of Walks Sensitivity"),
        ("window_size", "Context Window Size (w)", "Context Window Size Sensitivity"),
    ]:
        if results[param_key]:
            plot_hyperparameter_sensitivity(
                results[param_key], param_key, xlabel, title,
                filename=f"exp_A_sensitivity_{param_key}.png"
            )
            print(f"  Saved: exp_A_sensitivity_{param_key}.png")

    # Find best values
    print_subheader("Best Hyperparameters")
    for param_key, param_name in [
        ("embedding_dim", "d"), ("walk_length", "t"),
        ("num_walks", "gamma"), ("window_size", "w")
    ]:
        if results[param_key]:
            best = max(results[param_key].items(), key=lambda x: x[1]["silhouette"])
            print(f"  Best {param_name}: {best[0]} (silhouette={best[1]['silhouette']:.3f})")

    return results


# ============================================================
# STEP 4B: Runtime Analysis
# ============================================================
def experiment_runtime():
    """Measure runtime for each pipeline step."""
    print_header("STEP 4B: Runtime Analysis")

    runtime_results = {}

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        result = run_pipeline(filepath, seed=0)

        if result["success"]:
            timing = result["timing"]
            n_students = result["num_students"]
            n_edges = result["num_edges"]

            runtime_results[file_idx] = {
                "students": n_students,
                "edges": n_edges,
                "graph_construction": timing["graph_construction"],
                "random_walks": timing["random_walks"],
                "word2vec_training": timing["word2vec_training"],
                "total": timing["total"],
            }

            print(f"  Course {file_idx}: {n_students} students, {n_edges} edges")
            print(f"    Graph construction: {timing['graph_construction']:.3f}s")
            print(f"    Random walks:       {timing['random_walks']:.3f}s")
            print(f"    Word2Vec training:  {timing['word2vec_training']:.3f}s")
            print(f"    Total:              {timing['total']:.3f}s")

    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_B_runtime.json"), "w") as f:
        json.dump(runtime_results, f, indent=2, cls=NumpyEncoder)

    # Summary statistics
    print_subheader("Runtime Summary")
    totals = [v["total"] for v in runtime_results.values()]
    print(f"  Mean total runtime: {np.mean(totals):.3f}s")
    print(f"  Max total runtime:  {np.max(totals):.3f}s")

    # Generate plot
    plot_runtime_analysis(runtime_results, filename="exp_B_runtime_analysis.png")
    print("  Saved: exp_B_runtime_analysis.png")

    return runtime_results


# ============================================================
# STEP 4C: Random Seed Stability
# ============================================================
def experiment_seed_stability():
    """Evaluate embedding stability across random seeds."""
    print_header("STEP 4C: Random Seed Stability Analysis")

    stability_results = {}

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        print(f"\n  Course {file_idx}:")

        silhouettes = []
        dbis = []
        chs = []
        wcsss = []

        for seed in range(NUM_SEEDS):
            result = run_pipeline(filepath, seed=seed)
            if result["success"]:
                eval_r = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                m = eval_r["kmeans"]["metrics"]
                silhouettes.append(m["silhouette"])
                dbis.append(m["davies_bouldin"])
                chs.append(m["calinski_harabasz"])
                wcsss.append(m["wcss"])

        stability_results[str(file_idx)] = {
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
                "values": dbis,
            },
            "calinski_harabasz": {
                "mean": float(np.mean(chs)),
                "std": float(np.std(chs)),
                "values": chs,
            },
            "wcss": {
                "mean": float(np.mean(wcsss)),
                "std": float(np.std(wcsss)),
                "values": wcsss,
            },
        }

        sil_data = stability_results[str(file_idx)]["silhouette"]
        print(f"    Silhouette: {sil_data['mean']:.3f} +/- {sil_data['std']:.3f} "
              f"(min={sil_data['min']:.3f}, max={sil_data['max']:.3f})")

    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_C_seed_stability.json"), "w") as f:
        json.dump(stability_results, f, indent=2, cls=NumpyEncoder)

    # Summary
    print_subheader("Seed Stability Summary")
    all_means = [v["silhouette"]["mean"] for v in stability_results.values()]
    all_stds = [v["silhouette"]["std"] for v in stability_results.values()]
    print(f"  Average Silhouette across courses: {np.mean(all_means):.3f} +/- {np.mean(all_stds):.3f}")
    print(f"  Minimum silhouette (any seed, any course): {min(v['silhouette']['min'] for v in stability_results.values()):.3f}")

    return stability_results


# ============================================================
# STEP 4D: Clustering Stability
# ============================================================
def experiment_clustering_stability():
    """Measure clustering stability across methods and seeds."""
    print_header("STEP 4D: Clustering Stability Analysis")

    stability_results = {}

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        result = run_pipeline(filepath, seed=0)

        if not result["success"]:
            continue

        embeddings = result["embeddings"]
        print(f"\n  Course {file_idx}: {embeddings.shape[0]} students")

        course_stability = {}
        for method in ["kmeans", "gmm", "agglomerative"]:
            stab = compute_clustering_stability(
                embeddings, n_clusters=2, method=method,
                n_runs=NUM_SEEDS, base_seed=0
            )
            course_stability[method] = {
                "mean_ari": float(stab["mean_ari"]),
                "std_ari": float(stab["std_ari"]),
                "min_ari": float(stab["min_ari"]),
                "max_ari": float(stab["max_ari"]),
            }
            print(f"    {method:>15}: ARI = {stab['mean_ari']:.3f} +/- {stab['std_ari']:.3f}")

        stability_results[str(file_idx)] = course_stability

    # Save results
    with open(os.path.join(RESULTS_DIR, "exp_D_clustering_stability.json"), "w") as f:
        json.dump(stability_results, f, indent=2, cls=NumpyEncoder)

    # Summary
    print_subheader("Clustering Stability Summary")
    for method in ["kmeans", "gmm", "agglomerative"]:
        aris = [stability_results[str(fi)][method]["mean_ari"]
                for fi in FILE_INDICES if str(fi) in stability_results]
        print(f"  {method:>15} mean ARI: {np.mean(aris):.3f}")

    return stability_results


# ============================================================
# STEP 4E: Additional Clustering Metrics
# ============================================================
def experiment_additional_metrics():
    """Compute additional clustering quality metrics."""
    print_header("STEP 4E: Additional Clustering Metrics")

    all_results = []

    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, _ = read_class(filepath)
        result = run_pipeline(filepath, seed=0)

        if not result["success"] or scm is None:
            continue

        print(f"\n  Course {file_idx}:")

        # Evaluate BoW representation
        bow_eval = evaluate_clustering(scm, n_clusters=2, random_state=0)
        for method_name, method_data in bow_eval.items():
            if method_data["labels"] is not None:
                all_results.append({
                    "File Index": file_idx,
                    "Method": f"BoW {method_name}",
                    "Representation": "BoW",
                    "Silhouette": method_data["metrics"]["silhouette"],
                    "Davies-Bouldin": method_data["metrics"]["davies_bouldin"],
                    "Calinski-Harabasz": method_data["metrics"]["calinski_harabasz"],
                    "WCSS": method_data["metrics"]["wcss"],
                })

        # Evaluate DeepWalk embeddings
        dw_eval = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
        for method_name, method_data in dw_eval.items():
            if method_data["labels"] is not None:
                all_results.append({
                    "File Index": file_idx,
                    "Method": f"DeepWalk {method_name}",
                    "Representation": "DeepWalk",
                    "Silhouette": method_data["metrics"]["silhouette"],
                    "Davies-Bouldin": method_data["metrics"]["davies_bouldin"],
                    "Calinski-Harabasz": method_data["metrics"]["calinski_harabasz"],
                    "WCSS": method_data["metrics"]["wcss"],
                })

        # Print comparison for kmeans
        bow_s = bow_eval["kmeans"]["metrics"]["silhouette"]
        dw_s = dw_eval["kmeans"]["metrics"]["silhouette"]
        print(f"    KMeans: BoW={bow_s:.3f}, DeepWalk={dw_s:.3f}")

    df = pd.DataFrame(all_results)
    df.to_excel(os.path.join(RESULTS_DIR, "exp_E_additional_metrics.xlsx"), index=False)

    # Summary table
    print_subheader("Metrics Summary (KMeans, mean across courses)")
    kmeans_df = df[df['Method'].str.contains('kmeans')].copy()
    summary = kmeans_df.groupby('Representation')[['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']].mean()
    print(summary.round(3).to_string())

    return df


# ============================================================
# STEP 4F: Embedding Visualization (PCA + t-SNE)
# ============================================================
def experiment_visualization():
    """Generate PCA and t-SNE visualizations."""
    print_header("STEP 4F: Embedding Visualization")

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        result = run_pipeline(filepath, seed=0)

        if not result["success"]:
            print(f"  Course {i}: FAILED")
            continue

        embeddings = result["embeddings"]
        emb_labels = evaluate_clustering(embeddings, n_clusters=2, random_state=0)["kmeans"]["labels"]
        bow_labels = evaluate_clustering(scm, n_clusters=2, random_state=0)["kmeans"]["labels"]

        print(f"\n  Course {i}: {embeddings.shape[0]} students, {embeddings.shape[1]}D embeddings")

        # PCA of BoW
        bow_pca, bow_var = reduce_pca(scm, n_components=2)
        plot_embedding_visualization(
            scm, bow_labels,
            f"PCA of Student-Course Representation (Course {i})",
            f"exp_F_PCA_BoW_course{i}.png",
            reduced_2d=bow_pca,
        )

        # PCA of embeddings
        emb_pca, emb_var = reduce_pca(embeddings, n_components=2)
        plot_embedding_visualization(
            embeddings, emb_labels,
            f"PCA of DeepWalk-SSP Embeddings (Course {i})",
            f"exp_F_PCA_DeepWalk_course{i}.png",
            reduced_2d=emb_pca,
        )

        # t-SNE of BoW (with error handling for joblib/subprocess issues)
        try:
            bow_tsne = reduce_tsne(scm.astype(float), perplexity=min(20, scm.shape[0]-2), random_state=42)
            plot_embedding_visualization(
                scm, bow_labels,
                f"t-SNE of Student-Course Representation (Course {i})",
                f"exp_F_tSNE_BoW_course{i}.png",
                reduced_2d=bow_tsne,
            )
            print(f"    t-SNE BoW: OK")
        except Exception as e:
            print(f"    t-SNE BoW failed: {e}")

        # Direct 2D visualization of DeepWalk embeddings (no t-SNE needed)
        if embeddings.shape[1] == 2:
            plot_embedding_visualization(
                embeddings, emb_labels,
                f"DeepWalk-SSP Embeddings (Course {i})",
                f"exp_F_tSNE_DeepWalk_course{i}.png",
                reduced_2d=embeddings,
                axis_labels=('Embedding Dimension 1', 'Embedding Dimension 2'),
            )
            print(f"    DeepWalk 2D direct: OK")
        else:
            # Fallback: t-SNE only when d > 2
            try:
                emb_tsne = reduce_tsne(embeddings, perplexity=min(20, embeddings.shape[0]-2), random_state=42)
                plot_embedding_visualization(
                    embeddings, emb_labels,
                    f"t-SNE of DeepWalk-SSP Embeddings (Course {i})",
                    f"exp_F_tSNE_DeepWalk_course{i}.png",
                    reduced_2d=emb_tsne,
                )
                print(f"    t-SNE DeepWalk: OK")
            except Exception as e:
                print(f"    t-SNE DeepWalk failed: {e}")

    return True


# ============================================================
# STEP 5: Statistical Significance Analysis
# ============================================================
def experiment_statistical_analysis(seed_stability):
    """Perform Wilcoxon signed-rank tests comparing DeepWalk vs BoW."""
    print_header("STEP 5: Statistical Significance Analysis")

    stat_results = {}

    # Compute ACTUAL BoW silhouette scores (deterministic with fixed seed)
    bow_scores = {}
    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        bow_eval = evaluate_clustering(scm, n_clusters=2, random_state=0)
        bow_scores[i] = bow_eval["kmeans"]["metrics"]["silhouette"]
    print("  BoW silhouette scores:")
    for i in sorted(bow_scores.keys()):
        print(f"    Course {i}: {bow_scores[i]:.3f}")

    for i_str, course_data in seed_stability.items():
        i = int(i_str)
        dw_scores = np.array(course_data["silhouette"]["values"])
        bow_score = bow_scores[i]  # ACTUAL BoW silhouette (not the DeepWalk mean)

        # Wilcoxon signed-rank test: is DeepWalk significantly greater than BoW?
        test = wilcoxon_signed_rank_test(
            dw_scores, np.full_like(dw_scores, bow_score), alternative='greater'
        )
        r = compute_effect_size_r(test.get("statistic", 0), test.get("n", 0))
        delta, delta_interp = compute_cliffs_delta(dw_scores, np.full_like(dw_scores, bow_score))
        ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(dw_scores - bow_score)

        print(f"\n  Course {i}:")
        print(f"    DeepWalk: {np.mean(dw_scores):.3f} +/- {np.std(dw_scores):.3f}")
        print(f"    BoW:      {bow_score:.3f}")
        print(f"    Wilcoxon: stat={test['statistic']:.1f}, p={test['p_value']:.6f}")
        print(f"    Effect size r={r:.3f} ({'Large' if r>=0.5 else 'Medium' if r>=0.3 else 'Small'})")
        print(f"    Cliff's delta={delta:.3f} ({delta_interp})")
        print(f"    95% CI for diff: [{ci_lower:.3f}, {ci_upper:.3f}]")

        stat_results[str(i)] = {
            "deepwalk_mean": float(np.mean(dw_scores)),
            "deepwalk_std": float(np.std(dw_scores)),
            "bow_score": float(bow_score),
            "wilcoxon_stat": float(test["statistic"]),
            "wilcoxon_p": float(test["p_value"]),
            "effect_size_r": float(r),
            "cliffs_delta": float(delta),
            "cliffs_interp": delta_interp,
            "ci_mean_diff": float(ci_mean),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    # Aggregate across courses
    all_dw = [stat_results[str(i)]["deepwalk_mean"] for i in FILE_INDICES]
    all_bow = [stat_results[str(i)]["bow_score"] for i in FILE_INDICES]
    all_diff = [d - b for d, b in zip(all_dw, all_bow)]

    agg_test = wilcoxon_signed_rank_test(np.array(all_dw), np.array(all_bow), alternative='greater')
    agg_r = compute_effect_size_r(agg_test["statistic"], agg_test["n"])
    agg_delta, agg_delta_interp = compute_cliffs_delta(all_dw, all_bow)
    agg_ci_mean, agg_ci_lower, agg_ci_upper = compute_bootstrap_ci(np.array(all_diff))

    print(f"\n  AGGREGATE ({len(all_dw)} courses):")
    print(f"    Mean DeepWalk: {np.mean(all_dw):.3f}")
    print(f"    Mean BoW:      {np.mean(all_bow):.3f}")
    print(f"    Mean improvement: +{np.mean(all_diff):.3f}")
    print(f"    Wilcoxon: p={agg_test['p_value']:.6f}")
    print(f"    Effect size r={agg_r:.3f}")
    print(f"    Cliff's delta={agg_delta:.3f} ({agg_delta_interp})")
    print(f"    95% CI for mean improvement: [{agg_ci_lower:.3f}, {agg_ci_upper:.3f}]")

    stat_results["aggregate"] = {
        "mean_deepwalk": float(np.mean(all_dw)),
        "mean_bow": float(np.mean(all_bow)),
        "mean_improvement": float(np.mean(all_diff)),
        "wilcoxon_p": float(agg_test["p_value"]),
        "effect_size_r": float(agg_r),
        "cliffs_delta": float(agg_delta),
        "cliffs_interp": agg_delta_interp,
        "ci_lower": float(agg_ci_lower),
        "ci_upper": float(agg_ci_upper),
    }

    with open(os.path.join(RESULTS_DIR, "exp_statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2, cls=NumpyEncoder)

    return stat_results


# ============================================================
# STEP 5b: Statistical Significance — DeepWalk vs PCA+KMeans
# ============================================================
def experiment_statistical_analysis_pca(seed_stability):
    """Wilcoxon signed-rank test comparing DeepWalk-SSP vs PCA+KMeans."""
    print_header("STEP 5b: Statistical Significance — DeepWalk vs PCA+KMeans")

    # Load existing PCA+KMeans results
    pca_path = os.path.join(RESULTS_DIR, "baseline_pca_kmeans.json")
    if not os.path.exists(pca_path):
        print("  WARNING: baseline_pca_kmeans.json not found. Run Step 8 first.")
        return {}

    with open(pca_path) as f:
        pca_data = json.load(f)
    # Index by course
    pca_by_course = {int(c["course"]): c for c in pca_data}

    stat_results = {}

    for i in FILE_INDICES:
        if str(i) not in seed_stability:
            continue

        dw_scores = np.array(seed_stability[str(i)]["silhouette"]["values"])

        # PCA+KMeans: run with 20 seeds to get paired distribution
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)
        pca_scores = []
        for seed in range(NUM_SEEDS):
            labels, reduced = run_pca_kmeans(
                scm.astype(float), n_clusters=2, n_components=2,
                random_state=seed, n_init=10,
            )
            sil = silhouette_score(reduced, labels)
            pca_scores.append(sil)
        pca_scores = np.array(pca_scores)

        # Wilcoxon signed-rank test: DeepWalk > PCA
        test = wilcoxon_signed_rank_test(dw_scores, pca_scores, alternative='greater')
        r = compute_effect_size_r(test.get("statistic", 0), test.get("n", 0))
        delta, delta_interp = compute_cliffs_delta(dw_scores, pca_scores)
        diff = dw_scores - pca_scores
        ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(diff)

        print(f"\n  Course {i}:")
        print(f"    DeepWalk: {np.mean(dw_scores):.3f} +/- {np.std(dw_scores):.3f}")
        print(f"    PCA+KMeans: {np.mean(pca_scores):.3f} +/- {np.std(pca_scores):.3f}")
        print(f"    Wilcoxon: p={test.get('p_value', 0):.6f}, r={r:.3f}")
        print(f"    Cliff's delta={delta:.3f} ({delta_interp})")
        print(f"    95% CI for diff: [{ci_lower:.3f}, {ci_upper:.3f}]")

        stat_results[str(i)] = {
            "deepwalk_mean": float(np.mean(dw_scores)),
            "deepwalk_std": float(np.std(dw_scores)),
            "pca_mean": float(np.mean(pca_scores)),
            "pca_std": float(np.std(pca_scores)),
            "wilcoxon_stat": float(test.get("statistic", 0)),
            "wilcoxon_p": float(test.get("p_value", 0)),
            "effect_size_r": float(r),
            "cliffs_delta": float(delta),
            "cliffs_interp": delta_interp,
            "ci_mean_diff": float(ci_mean),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    # Aggregate
    all_dw = [stat_results[str(c)]["deepwalk_mean"] for c in FILE_INDICES if str(c) in stat_results]
    all_pca = [stat_results[str(c)]["pca_mean"] for c in FILE_INDICES if str(c) in stat_results]
    agg_test = wilcoxon_signed_rank_test(all_dw, all_pca, alternative='greater')
    agg_r = compute_effect_size_r(agg_test["statistic"], agg_test["n"])
    agg_delta, agg_delta_interp = compute_cliffs_delta(all_dw, all_pca)
    agg_diff = np.array(all_dw) - np.array(all_pca)
    agg_ci_mean, agg_ci_lower, agg_ci_upper = compute_bootstrap_ci(agg_diff)

    print(f"\n  AGGREGATE:")
    print(f"    Mean DeepWalk: {np.mean(all_dw):.3f}")
    print(f"    Mean PCA:      {np.mean(all_pca):.3f}")
    print(f"    Wilcoxon: p={agg_test['p_value']:.6f}, r={agg_r:.3f}")
    print(f"    95% CI for improvement: [{agg_ci_lower:.3f}, {agg_ci_upper:.3f}]")

    stat_results["aggregate"] = {
        "mean_deepwalk": float(np.mean(all_dw)),
        "mean_pca": float(np.mean(all_pca)),
        "mean_improvement": float(np.mean(agg_diff)),
        "wilcoxon_p": float(agg_test["p_value"]),
        "effect_size_r": float(agg_r),
        "cliffs_delta": float(agg_delta),
        "cliffs_interp": agg_delta_interp,
        "ci_lower": float(agg_ci_lower),
        "ci_upper": float(agg_ci_upper),
    }

    with open(os.path.join(RESULTS_DIR, "exp_statistical_analysis_pca.json"), "w") as f:
        json.dump(stat_results, f, indent=2, cls=NumpyEncoder)

    return stat_results


# ============================================================
# STEP 7: Practical Improvements
# ============================================================
def experiment_improvements():
    """Test practical pipeline modifications."""
    print_header("STEP 7: Practical Improvements")

    improvement_results = []

    for i in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{i}.txt")
        scm, _ = read_class(filepath)

        print(f"\n  Course {i} ({scm.shape[0]} students):")

        # Baseline
        base_result = run_pipeline(filepath, seed=0)
        if not base_result["success"]:
            continue
        base_metrics = evaluate_clustering(base_result["embeddings"], n_clusters=2, random_state=0)
        base_sil = base_metrics["kmeans"]["metrics"]["silhouette"]

        improvement_results.append({"Course": i, "Variant": "Baseline (standard)", "Silhouette": base_sil, "Type": "baseline"})
        print(f"    Baseline:          sil={base_sil:.3f}")

        # 1: Normalized edge weights
        G_norm = base_result["graph"].copy()
        max_w = max((d['weight'] for _, _, d in G_norm.edges(data=True)), default=1)
        for u, v, d in G_norm.edges(data=True):
            d['weight'] = d['weight'] / max_w
        walks_norm = generate_random_walks(G_norm, num_walks_per_node=80, walk_length=10, seed=0)
        w2v_norm = train_word2vec(walks_norm, vector_size=2, seed=0)
        if w2v_norm:
            m = evaluate_clustering(w2v_norm.wv.vectors, n_clusters=2, random_state=0)
            s = m["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Normalized edge weights", "Silhouette": s, "Type": "improvement"})
            print(f"    Normalized edges:  sil={s:.3f} ({'+' if s > base_sil else ''}{(s-base_sil):.3f})")

        # 2: More walks + longer walks
        r2 = run_pipeline(filepath, num_walks=160, walk_length=20, seed=0)
        if r2["success"]:
            s2 = evaluate_clustering(r2["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "More walks (160) + longer (20)", "Silhouette": s2, "Type": "improvement"})
            print(f"    More walks (160):  sil={s2:.3f} ({'+' if s2 > base_sil else ''}{(s2-base_sil):.3f})")

        # 3: d=1
        r3 = run_pipeline(filepath, vector_size=1, seed=0)
        if r3["success"]:
            s3 = evaluate_clustering(r3["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Vector size d=1", "Silhouette": s3, "Type": "improvement"})
            print(f"    Vector size d=1:   sil={s3:.3f} ({'+' if s3 > base_sil else ''}{(s3-base_sil):.3f})")

        # 4: Cosine similarity graph
        scm_f = scm.astype(float)
        norms = np.linalg.norm(scm_f, axis=1, keepdims=True)
        norms[norms == 0] = 1
        scm_n = scm_f / norms
        G_cos = nx.Graph()
        for k in range(scm_n.shape[0]):
            G_cos.add_node(k)
        for k1 in range(scm_n.shape[0]):
            for k2 in range(k1 + 1, scm_n.shape[0]):
                sim = float(np.dot(scm_n[k1], scm_n[k2]))
                if sim > 0.3:
                    G_cos.add_edge(k1, k2, weight=sim)
        walks_cos = generate_random_walks(G_cos, num_walks_per_node=80, walk_length=10, seed=0)
        w2v_cos = train_word2vec(walks_cos, vector_size=2, seed=0)
        if w2v_cos:
            s4 = evaluate_clustering(w2v_cos.wv.vectors, n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "Cosine similarity graph", "Silhouette": s4, "Type": "improvement"})
            print(f"    Cosine graph:     sil={s4:.3f} ({'+' if s4 > base_sil else ''}{(s4-base_sil):.3f})")

        # 5: 50 epochs
        r5 = run_pipeline(filepath, epochs=50, seed=0)
        if r5["success"]:
            s5 = evaluate_clustering(r5["embeddings"], n_clusters=2, random_state=0)["kmeans"]["metrics"]["silhouette"]
            improvement_results.append({"Course": i, "Variant": "50 epochs", "Silhouette": s5, "Type": "improvement"})
            print(f"    50 epochs:        sil={s5:.3f} ({'+' if s5 > base_sil else ''}{(s5-base_sil):.3f})")

    df = pd.DataFrame(improvement_results)
    df.to_excel(os.path.join(RESULTS_DIR, "exp_F_improvements.xlsx"), index=False)

    print(f"\n  Summary (Mean Silhouette):")
    summary = df.groupby("Variant")["Silhouette"].agg(["mean", "std"])
    print(summary.round(3).to_string())

    return df


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    total_start = time.time()

    print_header("DeepWalk-SSP: Complete Experiment Suite")
    print(f"  Output directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")
    print(f"  Courses: {FILE_INDICES}")
    print(f"  Seeds for stability: {NUM_SEEDS}")

    # Step 2: Reproduce existing experiments
    df, graphs = reproduce_existing()

    # Step 4A: Hyperparameter sensitivity
    hyper_results = experiment_hyperparameter_sensitivity(df, graphs)

    # Step 4B: Runtime analysis
    runtime_results = experiment_runtime()

    # Step 4C: Seed stability
    seed_results = experiment_seed_stability()

    # Step 4D: Clustering stability
    stab_results = experiment_clustering_stability()

    # Step 4E: Additional metrics
    metrics_df = experiment_additional_metrics()

    # Step 4F: Visualization
    experiment_visualization()

    # Step 5: Statistical analysis
    stat_results = experiment_statistical_analysis(seed_results)

    # Step 5b: Statistical analysis (DeepWalk vs PCA+KMeans)
    stat_pca_results = experiment_statistical_analysis_pca(seed_results)

    # Step 7: Practical improvements
    improvement_df = experiment_improvements()

    # Step 8: Baseline comparison (PCA+KMeans, Spectral Clustering)
    from experiments.add_baselines import (
        experiment_pca_kmeans, experiment_spectral_clustering,
        generate_comparison_figure
    )
    pca_baseline_df = experiment_pca_kmeans()
    pca_baseline_df.to_excel(os.path.join(RESULTS_DIR, "baseline_pca_kmeans.xlsx"), index=False)
    spectral_baseline_df = experiment_spectral_clustering()
    spectral_baseline_df.to_excel(os.path.join(RESULTS_DIR, "baseline_spectral.xlsx"), index=False)
    generate_comparison_figure(pca_baseline_df, spectral_baseline_df)

    # ── Final Summary ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print_header("COMPLETE: All Experiments Finished")
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Generated files in {RESULTS_DIR}:")

    results_files = sorted(os.listdir(RESULTS_DIR))
    for f in results_files:
        if f.startswith("exp_") or f.startswith("repro_") or f.startswith("baseline_"):
            print(f"    {f}")

    figures_files = sorted(os.listdir(FIGURES_DIR))
    if figures_files:
        print(f"\n  Generated figures in {FIGURES_DIR}:")
        for f in figures_files:
            if not f.endswith('.stt'):
                print(f"    {f}")

    # Key findings summary
    print_subheader("Key Findings")

    if hyper_results:
        best_d = max(hyper_results["embedding_dim"].items(), key=lambda x: x[1]["silhouette"])
        best_t = max(hyper_results["walk_length"].items(), key=lambda x: x[1]["silhouette"])
        best_g = max(hyper_results["num_walks"].items(), key=lambda x: x[1]["silhouette"])
        print(f"  Best embedding dimension d={best_d[0]} (silhouette={best_d[1]['silhouette']:.3f})")
        print(f"  Best walk length t={best_t[0]} (silhouette={best_t[1]['silhouette']:.3f})")
        print(f"  Best number of walks gamma={best_g[0]} (silhouette={best_g[1]['silhouette']:.3f})")

    if stat_results and "aggregate" in stat_results:
        agg = stat_results["aggregate"]
        print(f"  Aggregate improvement: +{agg['mean_improvement']:.3f}")
        print(f"  Wilcoxon p-value: {agg['wilcoxon_p']:.6f}")
        print(f"  Effect size r: {agg['effect_size_r']:.3f}")

    print(f"\n  To reproduce: python -m experiments.run_experiments")
    print()
