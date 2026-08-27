# Sections F+G+H: Stability, Statistics, Figures
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from experiments.config import FILE_INDICES, DATA_DIR, RESULTS_DIR, COLORS
from experiments.core import read_class, create_graph_from_bow, run_pipeline, run_node2vec_pipeline
from experiments.evaluation import compute_all_metrics, run_pca_kmeans, evaluate_clustering
from experiments.stats import (
    wilcoxon_signed_rank_test, compute_effect_size_r,
    compute_cliffs_delta, compute_bootstrap_ci
)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
LINUX_DIR = os.path.join(RESULTS_DIR, "linux_reexperiment")
LINUX_FIG_DIR = os.path.join(LINUX_DIR, "figures")
os.makedirs(LINUX_FIG_DIR, exist_ok=True)

with open(os.path.join(LINUX_DIR, "all_methods_results.json")) as f:
    all_results = json.load(f)
main_df = pd.DataFrame(all_results)
courses = sorted(main_df["course"].values)

# ============================================================
# SECTION F: Stability (10 seeds)
# ============================================================
print("=" * 70)
print("  SECTION F: Clustering Stability (10 seeds)")
print("=" * 70)
NUM_SEEDS = 5
stability = {}
for method_name, fn in [
    ("DeepWalk", lambda fp, s: run_pipeline(fp, seed=s)),
    ("Node2Vec(1,1)", lambda fp, s: run_node2vec_pipeline(fp, p=1.0, q=1.0, seed=s)),
    ("Node2Vec(0.5,1)", lambda fp, s: run_node2vec_pipeline(fp, p=0.5, q=1.0, seed=s)),
]:
    stability[method_name] = {}
    print(f"\n  Method: {method_name}")
    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        sils = []
        for seed in range(NUM_SEEDS):
            result = fn(filepath, seed)
            if result["success"]:
                ev = evaluate_clustering(result["embeddings"], n_clusters=2, random_state=0)
                sils.append(ev["kmeans"]["metrics"]["silhouette"])
        stability[method_name][str(file_idx)] = {
            "silhouette": {"mean": float(np.mean(sils)), "std": float(np.std(sils)),
                          "min": float(np.min(sils)), "max": float(np.max(sils)), "values": sils}
        }
        print(f"    Course {file_idx}: {np.mean(sils):.4f} +/- {np.std(sils):.4f} [{np.min(sils):.4f}, {np.max(sils):.4f}]")
with open(os.path.join(LINUX_DIR, "stability_analysis.json"), "w") as f:
    json.dump(stability, f, indent=2, cls=NumpyEncoder)
print("  Saved: stability_analysis.json")

# ============================================================
# SECTION G: Statistical Analysis
# ============================================================
print("\n" + "=" * 70)
print("  SECTION G: Statistical Analysis")
print("=" * 70)

stat_results = {}
for comp_name, meth_a, meth_b in [
    ("DeepWalk_vs_BoW", "DeepWalk", "BoW"),
    ("Node2Vec(1,1)_vs_DeepWalk", "Node2Vec(1,1)", "DeepWalk"),
    ("Node2Vec(0.5,1)_vs_DeepWalk", "Node2Vec(0.5,1)", "DeepWalk"),
    ("Node2Vec(0.5,1)_vs_BoW", "Node2Vec(0.5,1)", "BoW"),
    ("Node2Vec(1,1)_vs_PCA", "Node2Vec(1,1)", "PCA"),
    ("DeepWalk_vs_PCA", "DeepWalk", "PCA"),
]:
    course_stats = {}
    for file_idx in FILE_INDICES:
        filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
        scm, _ = read_class(filepath)
        if meth_a in stability and str(file_idx) in stability[meth_a]:
            scores_a = np.array(stability[meth_a][str(file_idx)]["silhouette"]["values"])
        elif meth_a == "BoW":
            bow_eval = evaluate_clustering(scm.astype(float), n_clusters=2, random_state=0)
            scores_a = np.full(NUM_SEEDS, bow_eval["kmeans"]["metrics"]["silhouette"])
        elif meth_a == "PCA":
            scores_a = np.array([compute_all_metrics(run_pca_kmeans(scm.astype(float), 2, 2, random_state=s)[1], run_pca_kmeans(scm.astype(float), 2, 2, random_state=s)[0])["silhouette"] for s in range(NUM_SEEDS)])
        else:
            continue
        if meth_b in stability and str(file_idx) in stability[meth_b]:
            scores_b = np.array(stability[meth_b][str(file_idx)]["silhouette"]["values"])
        elif meth_b == "BoW":
            bow_eval = evaluate_clustering(scm.astype(float), n_clusters=2, random_state=0)
            scores_b = np.full(len(scores_a), bow_eval["kmeans"]["metrics"]["silhouette"])
        elif meth_b == "PCA":
            def _pca_sil(seed):
                labels, reduced = run_pca_kmeans(scm.astype(float), 2, 2, random_state=seed)
                return compute_all_metrics(reduced, labels)["silhouette"]
            scores_b = np.array([_pca_sil(s) for s in range(len(scores_a))])
        else:
            continue
        n = min(len(scores_a), len(scores_b))
        scores_a, scores_b = scores_a[:n], scores_b[:n]
        test = wilcoxon_signed_rank_test(scores_a, scores_b, alternative='greater')
        r = compute_effect_size_r(test.get("statistic", 0), test.get("n", 0))
        delta, delta_interp = compute_cliffs_delta(scores_a, scores_b)
        ci_mean, ci_lower, ci_upper = compute_bootstrap_ci(scores_a - scores_b)
        course_stats[str(file_idx)] = {
            "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
            "wilcoxon_p": float(test.get("p_value", 0)) if not np.isnan(test.get("p_value", 0)) else None,
            "effect_size_r": float(r), "cliffs_delta": float(delta),
            "cliffs_interp": delta_interp, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
        }
    all_a = np.array([course_stats[str(c)]["mean_a"] for c in FILE_INDICES if str(c) in course_stats])
    all_b = np.array([course_stats[str(c)]["mean_b"] for c in FILE_INDICES if str(c) in course_stats])
    if len(all_a) >= 3:
        agg_test = wilcoxon_signed_rank_test(all_a, all_b, alternative='greater')
        agg_r = compute_effect_size_r(agg_test.get("statistic", 0), agg_test.get("n", 0))
        agg_delta, agg_delta_interp = compute_cliffs_delta(all_a, all_b)
        agg_ci_mean, agg_ci_lower, agg_ci_upper = compute_bootstrap_ci(all_a - all_b)
        course_stats["aggregate"] = {
            "mean_a": float(np.mean(all_a)), "mean_b": float(np.mean(all_b)),
            "wilcoxon_p": float(agg_test.get("p_value", 0)),
            "effect_size_r": float(agg_r), "cliffs_delta": float(agg_delta),
            "ci_lower": float(agg_ci_lower), "ci_upper": float(agg_ci_upper),
        }
        sig = "***" if agg_test.get("p_value", 1) < 0.001 else "**" if agg_test.get("p_value", 1) < 0.01 else "*" if agg_test.get("p_value", 1) < 0.05 else "ns"
        print(f"  {comp_name}: {np.mean(all_a):.4f} vs {np.mean(all_b):.4f} (Δ={np.mean(all_a-all_b):.4f}, p={agg_test.get('p_value', 1):.6f} {sig}, r={agg_r:.3f})")
    stat_results[comp_name] = course_stats
with open(os.path.join(LINUX_DIR, "statistical_analysis.json"), "w") as f:
    json.dump(stat_results, f, indent=2, cls=NumpyEncoder)
print("  Saved: statistical_analysis.json")

# ============================================================
# SECTION H: Generate ALL Figures (PDF)
# ============================================================
print("\n" + "=" * 70)
print("  SECTION H: Generating Figures")
print("=" * 70)

plt.rcParams.update({
    "figure.figsize": (7, 5), "figure.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "lines.linewidth": 1.5,
    "lines.markersize": 6, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

def save_fig(fig, fn, directory=LINUX_FIG_DIR, dpi=300):
    p = os.path.join(directory, fn)
    fig.savefig(p, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig); return p

course_labels = [f"Course {int(c)}" for c in courses]
x = np.arange(len(courses))

# Methods and colors for 6-way comparison
methods6 = ["BoW+KMeans", "PCA+KMeans", "Spectral", "DeepWalk-SSP", "Node2Vec-SSP\n(p=1.0, q=1.0)", "Node2Vec-SSP\n(p=0.5, q=1.0)"]
colors6 = [COLORS['bow'], COLORS['pca_kmeans'], COLORS['spectral'], COLORS['deepwalk'], COLORS['node2vec'], '#FF5722']
col_map_sil = {
    "BoW+KMeans": "bow_silhouette", "PCA+KMeans": "pca_silhouette", "Spectral": "spectral_silhouette",
    "DeepWalk-SSP": "deepwalk_silhouette",
    "Node2Vec-SSP\n(p=1.0, q=1.0)": "n2v_p1.0_q1.0_silhouette",
    "Node2Vec-SSP\n(p=0.5, q=1.0)": "n2v_p0.5_q1.0_silhouette",
}
width = 0.13

# Fig 1: Silhouette
fig, ax = plt.subplots(figsize=(12, 6))
for i, (m, c) in enumerate(zip(methods6, colors6)):
    vals = [main_df[main_df["course"]==cc][col_map_sil[m]].values[0] for cc in courses]
    offset = (i - 2.5) * width
    bars = ax.bar(x + offset, vals, width, label=m, color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=4.5)
ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (↑ higher is better)')
ax.set_title('Clustering Quality Comparison Across Six Courses')
ax.set_xticks(x); ax.set_xticklabels(course_labels)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7); ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
save_fig(fig, "fig1_silhouette_comparison.pdf")
print("  fig1_silhouette_comparison.pdf")
save_fig(fig, "fig1_silhouette_comparison.png", dpi=150)

# Fig 2: DBI
fig, ax = plt.subplots(figsize=(12, 6))
col_map_dbi = {k: v.replace("silhouette", "davies_bouldin") for k, v in col_map_sil.items()}
for i, (m, c) in enumerate(zip(methods6, colors6)):
    vals = [main_df[main_df["course"]==cc][col_map_dbi[m]].values[0] for cc in courses]
    offset = (i - 2.5) * width
    ax.bar(x + offset, vals, width, label=m, color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Course'); ax.set_ylabel('Davies-Bouldin Index (↓ lower is better)')
ax.set_title('Davies-Bouldin Index Across Six Courses')
ax.set_xticks(x); ax.set_xticklabels(course_labels)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
save_fig(fig, "fig2_dbi_comparison.pdf")
print("  fig2_dbi_comparison.pdf")

# Fig 3: CH
fig, ax = plt.subplots(figsize=(12, 6))
col_map_ch = {k: v.replace("silhouette", "calinski_harabasz") for k, v in col_map_sil.items()}
for i, (m, c) in enumerate(zip(methods6, colors6)):
    vals = [main_df[main_df["course"]==cc][col_map_ch[m]].values[0] for cc in courses]
    offset = (i - 2.5) * width
    ax.bar(x + offset, vals, width, label=m, color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Course'); ax.set_ylabel('Calinski-Harabasz Index (↑ higher is better)')
ax.set_title('Calinski-Harabasz Index Across Six Courses')
ax.set_xticks(x); ax.set_xticklabels(course_labels)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
save_fig(fig, "fig3_ch_comparison.pdf")
print("  fig3_ch_comparison.pdf")

# Fig 4: Sensitivity heatmap
with open(os.path.join(LINUX_DIR, "node2vec_sensitivity.json")) as f:
    sens = json.load(f)
p_vals_s = sorted(set(r["p"] for r in sens))
q_vals_s = sorted(set(r["q"] for r in sens))
hm = np.zeros((len(p_vals_s), len(q_vals_s)))
for i, pv in enumerate(p_vals_s):
    for j, qv in enumerate(q_vals_s):
        m = [r["silhouette"] for r in sens if r["p"]==pv and r["q"]==qv]
        hm[i, j] = np.mean(m) if m else np.nan
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(hm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(q_vals_s))); ax.set_xticklabels([f"{q:.2f}" for q in q_vals_s])
ax.set_yticks(range(len(p_vals_s))); ax.set_yticklabels([f"{p:.2f}" for p in p_vals_s])
ax.set_xlabel('q (BFS↔DFS parameter)'); ax.set_ylabel('p (return parameter)')
ax.set_title('Node2Vec Silhouette Score: Parameter Sensitivity\n(Average across 6 courses)')
for i in range(len(p_vals_s)):
    for j in range(len(q_vals_s)):
        if not np.isnan(hm[i,j]):
            ax.text(j, i, f'{hm[i,j]:.3f}', ha='center', va='center', fontsize=12, color='white' if hm[i,j]>0.6 else 'black')
fig.colorbar(im, ax=ax, label='Silhouette Score')
save_fig(fig, "fig4_node2vec_sensitivity.pdf")
print("  fig4_node2vec_sensitivity.pdf")

# Fig 5: Seed stability
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (mn, co) in enumerate([("DeepWalk", COLORS['deepwalk']), ("Node2Vec(1,1)", COLORS['node2vec']), ("Node2Vec(0.5,1)", '#FF5722')]):
    ax = axes[idx]; bd = []; lb = []
    for c in courses:
        if mn in stability and str(int(c)) in stability[mn]:
            bd.append(stability[mn][str(int(c))]["silhouette"]["values"])
            lb.append(f"C{int(c)}")
    bp = ax.boxplot(bd, tick_labels=lb, patch_artist=True)
    for p in bp['boxes']: p.set_facecolor(co); p.set_alpha(0.5); p.set_edgecolor(co)
    ax.set_ylabel('Silhouette Score'); ax.set_title(f'{mn}\nSeed Stability (n={NUM_SEEDS})')
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig, "fig5_seed_stability.pdf")
print("  fig5_seed_stability.pdf")

# Fig 6: Runtime
fig, ax = plt.subplots(figsize=(10, 5))
rt_cols = ['bow_runtime_s', 'pca_runtime_s', 'spectral_runtime_s', 'deepwalk_runtime_s', 'n2v_p1.0_q1.0_runtime_s']
rt_labels = ['BoW', 'PCA', 'Spectral', 'DeepWalk', 'Node2Vec']
for i, (lb, col, co) in enumerate(zip(rt_labels, rt_cols, colors6[:5])):
    vals = main_df[col].values
    ax.bar(x + (i-2)*0.15, vals, 0.15, label=lb, color=co, alpha=0.85)
ax.set_xlabel('Course'); ax.set_ylabel('Runtime (seconds)')
ax.set_title('Runtime Comparison Across Methods')
ax.set_xticks(x); ax.set_xticklabels(course_labels)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
save_fig(fig, "fig6_runtime_comparison.pdf")
print("  fig6_runtime_comparison.pdf")

# Fig 7: Node2Vec configs line
fig, ax = plt.subplots(figsize=(10, 6))
cfgs = [("N2V(1.0,1.0)", "n2v_p1.0_q1.0_silhouette", COLORS['node2vec']),
        ("N2V(0.5,1.0)", "n2v_p0.5_q1.0_silhouette", '#FF5722'),
        ("N2V(0.5,2.0)", "n2v_p0.5_q2.0_silhouette", '#9C27B0'),
        ("N2V(1.0,0.5)", "n2v_p1.0_q0.5_silhouette", '#FF9800'),
        ("DeepWalk", "deepwalk_silhouette", COLORS['deepwalk'])]
for lb, col, co in cfgs:
    vals = [main_df[main_df["course"]==cc][col].values[0] for cc in courses]
    ax.plot(x, vals, marker='o', label=lb, color=co, linewidth=2, markersize=7)
ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (↑)')
ax.set_title('Node2Vec Configuration Comparison')
ax.set_xticks(x); ax.set_xticklabels(course_labels)
ax.legend(loc='best', framealpha=0.9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
save_fig(fig, "fig7_node2vec_configs.pdf")
print("  fig7_node2vec_configs.pdf")

# Fig 8: Average summary
fig, ax = plt.subplots(figsize=(10, 5))
avg_lbs = ['BoW\n+KMeans', 'PCA\n+KMeans', 'Spectral\nClustering', 'DeepWalk\n-SSP', 'Node2Vec\n-SSP(1,1)', 'Node2Vec\n-SSP(0.5,1)']
avg_cols = ['bow_silhouette', 'pca_silhouette', 'spectral_silhouette', 'deepwalk_silhouette', 'n2v_p1.0_q1.0_silhouette', 'n2v_p0.5_q1.0_silhouette']
avgs = [main_df[c].mean() for c in avg_cols]
stds = [main_df[c].std() for c in avg_cols]
bars = ax.bar(range(len(avg_lbs)), avgs, yerr=stds, capsize=4, color=colors6, alpha=0.85, edgecolor='white')
for b, v in zip(bars, avgs):
    ax.annotate(f'{v:.3f}', xy=(b.get_x()+b.get_width()/2, b.get_height()), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Mean Silhouette Score (↑)')
ax.set_title('Average Clustering Quality Across All Six Courses')
ax.set_xticks(range(len(avg_lbs))); ax.set_xticklabels(avg_lbs, fontsize=8)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True); ax.set_ylim(0, 1.0)
save_fig(fig, "fig8_average_comparison.pdf")
print("  fig8_average_comparison.pdf")

# Fig 9: All metrics grouped
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
mc = [("Silhouette Score (↑)", ["bow_silhouette","pca_silhouette","spectral_silhouette","deepwalk_silhouette","n2v_p1.0_q1.0_silhouette","n2v_p0.5_q1.0_silhouette"]),
      ("Davies-Bouldin Index (↓)", ["bow_davies_bouldin","pca_davies_bouldin","spectral_davies_bouldin","deepwalk_davies_bouldin","n2v_p1.0_q1.0_davies_bouldin","n2v_p0.5_q1.0_davies_bouldin"]),
      ("Calinski-Harabasz (↑)", ["bow_calinski_harabasz","pca_calinski_harabasz","spectral_calinski_harabasz","deepwalk_calinski_harabasz","n2v_p1.0_q1.0_calinski_harabasz","n2v_p0.5_q1.0_calinski_harabasz"])]
sl = ['BoW', 'PCA', 'Spectral', 'DeepWalk', 'N2V(1,1)', 'N2V(0.5,1)']
for ax, (title, cols) in zip(axes, mc):
    vals = [main_df[c].mean() for c in cols]
    bars = ax.bar(range(len(vals)), vals, color=colors6, alpha=0.85, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.2f}', xy=(b.get_x()+b.get_width()/2, b.get_height()), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(sl))); ax.set_xticklabels(sl, fontsize=8)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
fig.suptitle('All Metrics: Average Across Six Courses', fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "fig9_all_metrics_comparison.pdf")
print("  fig9_all_metrics_comparison.pdf")

print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE")
print("=" * 70)
