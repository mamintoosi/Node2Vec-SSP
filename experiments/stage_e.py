# -*- coding: utf-8 -*-
"""Stage E: Combine stability results, statistical analysis, final figures, report."""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon as scipy_wilcoxon
warnings.filterwarnings("ignore")

from experiments.shared import (OUT_DIR, FIG_DIR, FILE_INDICES, load_course_data,
    generate_unbiased_walks, generate_node2vec_walks, train_word2vec,
    cluster_kmeans, cluster_spectral, compute_metrics)
from sklearn.metrics import silhouette_score
import networkx as nx

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {"bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B",
          "deepwalk": "#2196F3", "node2vec": "#4CAF50"}


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return path


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
    if n < 5: return np.nan
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0: return 0.0
    z = (statistic - mean_w) / std_w
    return min(abs(z) / np.sqrt(n), 1.0)


def cliffs_delta(x, y):
    n, m = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n * m)


def holm_correction(p_values):
    """Holm (1979) step-down multiple-comparison correction.

    1. Sort raw p-values ascending.
    2. Multiply the i-th ordered p-value by (m - i) [0-based index].
    3. Enforce monotonicity via cumulative maximum (forward pass).
    4. Clamp to 1.0 and restore original order.
    """
    p = np.array(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.minimum(sorted_p * np.arange(m, 0, -1), 1.0)
    adjusted = np.maximum.accumulate(adjusted)          # enforce monotonicity
    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()


def main():
    t0 = time.time()
    print("=" * 70)
    print("  STAGE E: Statistical Analysis, Final Figures, Report")
    print("=" * 70)

    # Load all stability data
    partial_path = os.path.join(OUT_DIR, "stability_partial.json")
    with open(partial_path) as f:
        raw = json.load(f)

    # Organize by course
    per_course = {str(i): {"n2v": [], "dw": []} for i in FILE_INDICES}
    for key, data in raw.items():
        seed, course = key.split("_")
        per_course[course]["n2v"].append(data["n2v_silhouette"])
        per_course[course]["dw"].append(data["dw_silhouette"])

    # Stability stats
    stability_stats = {}
    for i in FILE_INDICES:
        c = str(i)
        n2v = np.array(per_course[c]["n2v"])
        dw = np.array(per_course[c]["dw"])
        stability_stats[c] = {
            "node2vec": {
                "silhouette": {"mean": float(np.mean(n2v)), "std": float(np.std(n2v)),
                               "min": float(np.min(n2v)), "max": float(np.max(n2v))},
            },
            "deepwalk": {
                "silhouette": {"mean": float(np.mean(dw)), "std": float(np.std(dw)),
                               "min": float(np.min(dw)), "max": float(np.max(dw))},
            },
            "n_seeds": len(n2v),
        }

    print(f"\n  Stability ({len(per_course['1']['n2v'])} seeds):")
    print(f"  {'Course':<10} {'Node2Vec':>20} {'DeepWalk':>20}")
    print("  " + "-" * 50)
    for i in FILE_INDICES:
        c = str(i)
        n2v = stability_stats[c]["node2vec"]["silhouette"]
        dw = stability_stats[c]["deepwalk"]["silhouette"]
        print(f"  Course {i:<4} {n2v['mean']:.3f}±{n2v['std']:.3f} ({n2v['min']:.3f}-{n2v['max']:.3f})  "
              f"{dw['mean']:.3f}±{dw['std']:.3f} ({dw['min']:.3f}-{dw['max']:.3f})")

    with open(os.path.join(OUT_DIR, "stability.json"), "w") as f:
        json.dump(stability_stats, f, indent=2)

    # Boxplot
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))
    n2v_data = [per_course[str(i)]["n2v"] for i in FILE_INDICES]
    dw_data = [per_course[str(i)]["dw"] for i in FILE_INDICES]
    positions_n2v = np.arange(1, 7) * 3 - 0.4
    positions_dw = np.arange(1, 7) * 3 + 0.4
    bp1 = ax.boxplot(n2v_data, positions=positions_n2v, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor=COLORS["node2vec"], alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    bp2 = ax.boxplot(dw_data, positions=positions_dw, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor=COLORS["deepwalk"], alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    ax.set_xticks(np.arange(1, 7) * 3)
    ax.set_xticklabels([f"Course {i}" for i in FILE_INDICES])
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Node2Vec vs DeepWalk Stability Across Seeds (Revised Graph)')
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=COLORS["node2vec"], alpha=0.7), Patch(facecolor=COLORS["deepwalk"], alpha=0.7)],
              ["Node2Vec (p=1,q=1)", "DeepWalk"], framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    save_fig(fig, "stability_boxplot")
    print("  Saved: stability_boxplot.pdf")

    # ═══ Statistical Analysis ═══
    print("\n--- Statistical Analysis ---")
    course_data = load_course_data()

    # Mean Node2Vec and DeepWalk silhouettes across seeds
    n2v_means = np.array([stability_stats[str(c)]["node2vec"]["silhouette"]["mean"] for c in FILE_INDICES])
    dw_means = np.array([stability_stats[str(c)]["deepwalk"]["silhouette"]["mean"] for c in FILE_INDICES])

    # Deterministic baselines
    bow_scores, pca_scores, spec_scores = [], [], []
    for i in FILE_INDICES:
        cd = course_data[i]
        m = cd["matrix"]
        bl = cluster_kmeans(m.astype(float), seed=0)
        bow_scores.append(float(silhouette_score(m.astype(float), bl)))
        from sklearn.decomposition import PCA
        reduced = PCA(n_components=2, random_state=0).fit_transform(m.astype(float))
        pl = cluster_kmeans(reduced, seed=0)
        pca_scores.append(float(silhouette_score(m.astype(float), pl)))
        G = cd["graph"]
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        sl = cluster_spectral(adj, seed=0)
        spec_scores.append(float(silhouette_score(m.astype(float), sl)))

    bow_scores, pca_scores, spec_scores = np.array(bow_scores), np.array(pca_scores), np.array(spec_scores)

    print(f"\n  Course-level Silhouette scores:")
    print(f"  {'Course':<10} {'Node2Vec':>10} {'DeepWalk':>10} {'BoW':>10} {'PCA':>10} {'Spectral':>10}")
    print("  " + "-" * 60)
    for ci, c in enumerate(FILE_INDICES):
        print(f"  Course {c:<4} {n2v_means[ci]:>10.3f} {dw_means[ci]:>10.3f} "
              f"{bow_scores[ci]:>10.3f} {pca_scores[ci]:>10.3f} {spec_scores[ci]:>10.3f}")
    print(f"  {'Average':<10} {np.mean(n2v_means):>10.3f} {np.mean(dw_means):>10.3f} "
          f"{np.mean(bow_scores):>10.3f} {np.mean(pca_scores):>10.3f} {np.mean(spec_scores):>10.3f}")

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

    corrected_p = holm_correction(raw_p)
    for cp, res in zip(corrected_p, stat_results):
        res["corrected_p"] = cp
        res["significant"] = cp < 0.05
        print(f"  {res['comparison']}: corrected p={cp:.4f} ({'*' if cp < 0.05 else 'ns'})")

    with open(os.path.join(OUT_DIR, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)

    # ═══ Final Figures ═══

    # Node2Vec vs DeepWalk per-course comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(FILE_INDICES))
    width = 0.35
    ax.bar(x - width/2, n2v_means, width, label='Node2Vec (p=1,q=1)', color=COLORS["node2vec"], alpha=0.85)
    ax.bar(x + width/2, dw_means, width, label='DeepWalk', color=COLORS["deepwalk"], alpha=0.85)
    for i_v, (n2v, dw) in enumerate(zip(n2v_means, dw_means)):
        ax.annotate(f'{n2v:.3f}', xy=(i_v-width/2, n2v), xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f'{dw:.3f}', xy=(i_v+width/2, dw), xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (↑)')
    ax.set_title('Node2Vec vs DeepWalk: Per-Course Comparison')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in FILE_INDICES])
    ax.legend(framealpha=0.9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    save_fig(fig, "node2vec_vs_deepwalk")
    print("  Saved: node2vec_vs_deepwalk.pdf")

    # All-methods silhouette comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    method_info = [("bow","BoW+KMeans",COLORS["bow"]), ("pca","PCA+KMeans",COLORS["pca"]),
                   ("spectral","Spectral",COLORS["spectral"]),
                   ("deepwalk","DeepWalk",COLORS["deepwalk"]), ("node2vec","Node2Vec(1,1)",COLORS["node2vec"])]
    all_scores = [bow_scores, pca_scores, spec_scores, dw_means, n2v_means]
    width = 0.15
    for i_m, ((mk, ml, mc), scores) in enumerate(zip(method_info, all_scores)):
        offset = (i_m - 2) * width
        bars = ax.bar(x + offset, scores, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=6)
    ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (↑)')
    ax.set_title('Method Comparison: Silhouette Score (Revised Graph)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in FILE_INDICES])
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9); ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    save_fig(fig, "method_comparison_silhouette")
    print("  Saved: method_comparison_silhouette.pdf")

    # ═══ Summary ═══
    total_time = time.time() - t0
    print(f"\n  Stage E complete. Time: {total_time:.1f}s")
    print(f"\n  All results saved to: {OUT_DIR}")
    print(f"  All figures saved to: {FIG_DIR}")

    # Load sensitivity for summary
    with open(os.path.join(OUT_DIR, "sensitivity.json")) as f:
        sens = json.load(f)
    pq = {}
    for d in sens:
        k = (d["p"], d["q"])
        if k not in pq: pq[k] = []
        pq[k].append(d["silhouette"])
    pq_means = {k: np.mean(v) for k, v in pq.items()}
    best = max(pq_means, key=pq_means.get)

    print(f"\n  KEY RESULTS:")
    print(f"  - Node2Vec avg Silhouette: {np.mean(n2v_means):.3f}")
    print(f"  - DeepWalk avg Silhouette: {np.mean(dw_means):.3f}")
    print(f"  - BoW avg Silhouette: {np.mean(bow_scores):.3f}")
    print(f"  - PCA avg Silhouette: {np.mean(pca_scores):.3f}")
    print(f"  - Spectral avg Silhouette: {np.mean(spec_scores):.3f}")
    print(f"  - Best Node2Vec config: p={best[0]}, q={best[1]} (Sil={pq_means[best]:.3f})")
    print(f"  - Sensitivity range: {min(pq_means.values()):.3f} to {max(pq_means.values()):.3f}")


if __name__ == "__main__":
    main()
