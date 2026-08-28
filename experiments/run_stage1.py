# -*- coding: utf-8 -*-
"""
Stage 1: Graph analysis, all methods (seed=0), and Node2Vec sensitivity.
"""

import os, sys, time, json, warnings, random, platform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REEXPERIMENT_DIR = os.path.join(RESULTS_DIR, "revised_reexperiment")
FIGURES_DIR = os.path.join(REEXPERIMENT_DIR, "figures")
os.makedirs(REEXPERIMENT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FILE_INDICES = [1, 2, 3, 4, 5, 6]

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {"bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B", "node2vec": "#4CAF50"}

# ── Utilities ────────────────────────────────────────────────────────────────
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
            if len(bv) < m: bv = bv.ljust(m, '0')
            matrix[j] = [int(c) for c in bv[:m]]
    return matrix, labels

def find_universal_columns(matrix):
    col_sums = matrix.sum(axis=0)
    return np.where(col_sums == matrix.shape[0])[0].tolist()

def create_graph(matrix):
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n): G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            w = int(np.sum(matrix[i] * matrix[j]))
            if w > 0: G.add_edge(i, j, weight=w)
    return G

def graph_stats(G):
    n, e = G.number_of_nodes(), G.number_of_edges()
    max_e = n * (n-1) // 2
    degs = [d for _, d in G.degree()]
    ws = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    return {
        "n_nodes": n, "n_edges": e, "max_edges": max_e,
        "density": e / max_e if max_e > 0 else 0,
        "n_components": nx.number_connected_components(G),
        "avg_degree": float(np.mean(degs)) if degs else 0,
        "std_degree": float(np.std(degs)) if degs else 0,
        "min_degree": int(np.min(degs)) if degs else 0,
        "max_degree": int(np.max(degs)) if degs else 0,
        "avg_weight": float(np.mean(ws)) if ws else 0,
        "std_weight": float(np.std(ws)) if ws else 0,
    }

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
    return SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                              random_state=seed, assign_labels='kmeans').fit_predict(adj)

def cluster_pca_kmeans(data, n_clusters=2, n_components=2, seed=0):
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(data)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(reduced)
    return labels, reduced

def generate_node2vec_walks(G, num_walks=80, walk_length=10, p=1.0, q=1.0, seed=0):
    rng = np.random.RandomState(seed)
    walks = []
    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            prev_node, current_node = None, node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current_node))
                if not neighbors: break
                if prev_node is None:
                    next_node = neighbors[rng.randint(len(neighbors))]
                else:
                    prev_set = set(G.neighbors(prev_node))
                    weights = []
                    for z in neighbors:
                        if z == prev_node: alpha = 1.0 / p
                        elif z in prev_set: alpha = 1.0
                        else: alpha = 1.0 / q
                        weights.append(alpha * G[current_node][z].get('weight', 1))
                    total = sum(weights)
                    probs = np.array(weights) / total
                    r = rng.random()
                    cum, next_node = 0.0, neighbors[-1]
                    for idx, prob in enumerate(probs):
                        cum += prob
                        if r <= cum: next_node = neighbors[idx]; break
                walk.append(next_node)
                prev_node, current_node = current_node, next_node
            walks.append(walk)
    return walks

def train_word2vec(walks, vector_size=2, window=5, epochs=30, seed=0, workers=1):
    model = Word2Vec(walks, vector_size=vector_size, window=window, hs=1, sg=1,
                     workers=workers, seed=seed, min_count=1, sample=0)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs, report_delay=0)
    return model.wv.vectors

def run_node2vec_full(matrix, G, p=1.0, q=1.0, n_clusters=2, seed=0, **kwargs):
    t0 = time.perf_counter()
    walks = generate_node2vec_walks(G, p=p, q=q, seed=seed)
    t_walks = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb = train_word2vec(walks, seed=seed)
    t_w2v = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = cluster_kmeans(emb, n_clusters=n_clusters, seed=seed)
    t_cluster = time.perf_counter() - t0
    m_emb = compute_metrics(emb, labels)
    m_bow = compute_metrics(matrix.astype(float), labels)
    return {
        "silhouette_emb": m_emb["silhouette"], "dbi_emb": m_emb["dbi"], "ch_emb": m_emb["ch"],
        "silhouette_bow": m_bow["silhouette"], "dbi_bow": m_bow["dbi"], "ch_bow": m_bow["ch"],
        "t_walks": t_walks, "t_w2v": t_w2v, "t_cluster": t_cluster,
        "t_total": t_walks + t_w2v + t_cluster,
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("  STAGE 1: Graph Analysis + All Methods + Sensitivity")
    print("=" * 70)

    # ═══ PART 1: Graph Structure Analysis ═══
    print("\n--- PART 1: Graph Structure Analysis ---")
    old_stats, new_stats, univ_info = {}, {}, {}
    for i in FILE_INDICES:
        matrix, _ = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        G_old = create_graph(matrix)
        old_stats[str(i)] = graph_stats(G_old)
        uc = find_universal_columns(matrix)
        matrix_new = np.delete(matrix, uc, axis=1)
        G_new = create_graph(matrix_new)
        new_stats[str(i)] = graph_stats(G_new)
        univ_info[str(i)] = {
            "n_students": matrix.shape[0], "n_courses_orig": matrix.shape[1],
            "universal_columns": uc, "n_removed": len(uc),
            "n_courses_remaining": matrix_new.shape[1],
        }
        print(f"  Course {i}: {matrix.shape[0]} students, {matrix.shape[1]}→{matrix_new.shape[1]} courses, "
              f"density {old_stats[str(i)]['density']:.4f}→{new_stats[str(i)]['density']:.4f}")

    with open(os.path.join(REEXPERIMENT_DIR, "graph_analysis.json"), "w") as f:
        json.dump({"universal_info": univ_info, "old": old_stats, "new": new_stats}, f, indent=2)

    # Figure: density comparison
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    courses = [str(i) for i in range(1, 7)]
    x = np.arange(6)
    w = 0.35
    ax.bar(x - w/2, [old_stats[c]["density"] for c in courses], w, label='Original', color='#E53935', alpha=0.8)
    ax.bar(x + w/2, [new_stats[c]["density"] for c in courses], w, label='Revised (universal removed)', color='#43A047', alpha=0.8)
    ax.set_xlabel('Course'); ax.set_ylabel('Graph Density')
    ax.set_title('Effect of Removing Universal Courses on Graph Density')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(); ax.set_ylim(0, 1.1); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "graph_density_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: graph_density_comparison.pdf")

    # ═══ PART 2: All Methods (seed=0) ═══
    print("\n--- PART 2: All Methods (seed=0) ---")
    all_results = []
    runtime_data = []

    for i in FILE_INDICES:
        matrix_orig, _ = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        uc = find_universal_columns(matrix_orig)
        matrix = np.delete(matrix_orig, uc, axis=1)
        G = create_graph(matrix)
        n = matrix.shape[0]
        print(f"\n  Course {i}: {n} students, {matrix.shape[1]} courses, {G.number_of_edges()} edges")

        # BoW
        t0 = time.perf_counter()
        bow_l = cluster_kmeans(matrix.astype(float), seed=0)
        t_bow = time.perf_counter() - t0
        bow_m = compute_metrics(matrix.astype(float), bow_l)
        print(f"    BoW+KMeans:    Sil={bow_m['silhouette']:.3f}, DBI={bow_m['dbi']:.3f}, CH={bow_m['ch']:.1f}")
        all_results.append({"course": i, "method": "bow", "sil": bow_m["silhouette"], "dbi": bow_m["dbi"], "ch": bow_m["ch"]})
        runtime_data.append({"course": i, "method": "BoW+KMeans", "t_total": t_bow})

        # PCA
        t0 = time.perf_counter()
        pca_l, pca_r = cluster_pca_kmeans(matrix.astype(float), seed=0)
        t_pca = time.perf_counter() - t0
        pca_m = compute_metrics(matrix.astype(float), pca_l)
        print(f"    PCA+KMeans:    Sil={pca_m['silhouette']:.3f}, DBI={pca_m['dbi']:.3f}, CH={pca_m['ch']:.1f}")
        all_results.append({"course": i, "method": "pca", "sil": pca_m["silhouette"], "dbi": pca_m["dbi"], "ch": pca_m["ch"]})
        runtime_data.append({"course": i, "method": "PCA+KMeans", "t_total": t_pca})

        # Spectral
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        t0 = time.perf_counter()
        spec_l = cluster_spectral(adj, seed=0)
        t_spec = time.perf_counter() - t0
        spec_m = compute_metrics(matrix.astype(float), spec_l)
        print(f"    Spectral:      Sil={spec_m['silhouette']:.3f}, DBI={spec_m['dbi']:.3f}, CH={spec_m['ch']:.1f}")
        all_results.append({"course": i, "method": "spectral", "sil": spec_m["silhouette"], "dbi": spec_m["dbi"], "ch": spec_m["ch"]})
        runtime_data.append({"course": i, "method": "Spectral", "t_total": t_spec})

        # Node2Vec(p=1, q=1)
        n2v = run_node2vec_full(matrix, G, p=1.0, q=1.0, seed=0)
        print(f"    Node2Vec(1,1): Sil={n2v['silhouette_emb']:.3f}, DBI={n2v['dbi_emb']:.3f}, CH={n2v['ch_emb']:.1f}")
        all_results.append({"course": i, "method": "node2vec", "sil": n2v["silhouette_emb"],
                           "dbi": n2v["dbi_emb"], "ch": n2v["ch_emb"]})
        runtime_data.append({"course": i, "method": "Node2Vec", "t_total": n2v["t_total"]})

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(os.path.join(REEXPERIMENT_DIR, "all_methods.xlsx"), index=False)
    results_df.to_json(os.path.join(REEXPERIMENT_DIR, "all_methods.json"), orient="records", indent=2)

    # Summary
    print("\n  SUMMARY (Silhouette Score, seed=0)")
    print(f"  {'Method':<20}", end="")
    for i in FILE_INDICES: print(f" {'C'+str(i):>6}", end="")
    print(f" {'Avg':>8}")
    print("  " + "-" * 60)
    for mname in ["bow", "pca", "spectral", "node2vec"]:
        sub = results_df[results_df["method"] == mname]
        label = {"bow": "BoW+KMeans", "pca": "PCA+KMeans", "spectral": "Spectral", "node2vec": "Node2Vec(1,1)"}[mname]
        print(f"  {label:<20}", end="")
        for i in FILE_INDICES:
            row = sub[sub["course"] == i]
            print(f" {row['sil'].values[0]:>6.3f}", end="")
        print(f" {sub['sil'].mean():>8.3f}")

    # ═══ PART 3: Node2Vec Sensitivity ═══
    print("\n--- PART 3: Node2Vec Parameter Sensitivity ---")
    p_vals, q_vals = [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]
    sens_results = []
    for p in p_vals:
        for q in q_vals:
            sils = []
            for i in FILE_INDICES:
                m_orig, _ = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
                uc = find_universal_columns(m_orig)
                m = np.delete(m_orig, uc, axis=1)
                G = create_graph(m)
                res = run_node2vec_full(m, G, p=p, q=q, seed=0)
                sils.append(res["silhouette_emb"])
                sens_results.append({"p": p, "q": q, "course": i,
                    "silhouette": res["silhouette_emb"], "dbi": res["dbi_emb"], "ch": res["ch_emb"]})
            print(f"    p={p:.1f}, q={q:.1f}: avg Sil = {np.mean(sils):.3f}")

    sens_df = pd.DataFrame(sens_results)
    sens_df.to_excel(os.path.join(REEXPERIMENT_DIR, "sensitivity.xlsx"), index=False)
    sens_df.to_json(os.path.join(REEXPERIMENT_DIR, "sensitivity.json"), orient="records", indent=2)

    # Heatmap
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (met, title, cmap) in enumerate([
        ("silhouette", "Silhouette (↑)", "YlGn"),
        ("dbi", "Davies-Bouldin (↓)", "YlGn_r"),
        ("ch", "Calinski-Harabasz (↑)", "YlGn"),
    ]):
        pv = sens_df.groupby(["p", "q"])[met].mean().reset_index().pivot(index="p", columns="q", values=met)
        sns.heatmap(pv, annot=True, fmt=".3f", cmap=cmap, ax=axes[idx], cbar_kws={"shrink": 0.8})
        axes[idx].set_title(title); axes[idx].set_xlabel("q"); axes[idx].set_ylabel("p")
    fig.suptitle("Node2Vec Parameter Sensitivity (averaged over 6 courses)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sensitivity_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: sensitivity_heatmap.pdf")

    # Method comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))
    courses = sorted(results_df["course"].unique())
    methods_info = [("bow", "BoW+KMeans", COLORS["bow"]), ("pca", "PCA+KMeans", COLORS["pca"]),
                    ("spectral", "Spectral", COLORS["spectral"]), ("node2vec", "Node2Vec(1,1)", COLORS["node2vec"])]
    x = np.arange(len(courses)); width = 0.18
    for i_m, (mk, ml, mc) in enumerate(methods_info):
        vals = [results_df[(results_df["course"] == c) & (results_df["method"] == mk)]["sil"].values[0] for c in courses]
        offset = (i_m - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=6)
    ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (↑)')
    ax.set_title('Method Comparison: Clustering Quality (Revised Graph Construction)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(loc='upper left', framealpha=0.9); ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "method_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: method_comparison.pdf")

    # Runtime figure
    rt_df = pd.DataFrame(runtime_data)
    rt_df.to_json(os.path.join(REEXPERIMENT_DIR, "runtime.json"), orient="records", indent=2)
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_rt = ["BoW+KMeans", "PCA+KMeans", "Spectral", "Node2Vec"]
    colors_rt = [COLORS["bow"], COLORS["pca"], COLORS["spectral"], COLORS["node2vec"]]
    for i_m, (ml, mc) in enumerate(zip(methods_rt, colors_rt)):
        vals = [rt_df[(rt_df["course"] == c) & (rt_df["method"] == ml)]["t_total"].values[0] for c in courses]
        offset = (i_m - 1.5) * 0.18
        ax.bar(x + offset, vals, 0.18, label=ml, color=mc, alpha=0.85)
    ax.set_xlabel('Course'); ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime Comparison'); ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses]); ax.legend(); ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "runtime_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: runtime_comparison.pdf")

    print(f"\n  Stage 1 complete. Time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
