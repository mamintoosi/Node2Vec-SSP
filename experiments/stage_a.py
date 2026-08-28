# -*- coding: utf-8 -*-
"""
Stage A: Preprocessing, graph analysis, all methods (seed=0), and figures.
"""
import os, sys, time, json, warnings, platform
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
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "final_reexperiment")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
FILE_INDICES = [1, 2, 3, 4, 5, 6]

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {"bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B",
          "deepwalk": "#2196F3", "node2vec": "#4CAF50"}

def read_class(fp):
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        labels = []
        for j in range(n):
            parts = f.readline().split()
            labels.append(parts[1])
            bv = parts[2]
            if len(bv) < m: bv = bv.ljust(m, '0')
            mat[j] = [int(c) for c in bv[:m]]
    return mat, labels

def find_universal_columns(matrix):
    return np.where(matrix.sum(axis=0) == matrix.shape[0])[0].tolist()

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

def generate_unbiased_walks(G, num_walks=80, walk_length=10, seed=0):
    rng = np.random.RandomState(seed)
    walks = []
    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            cur = node
            for _ in range(walk_length - 1):
                nbs = list(G.neighbors(cur))
                if not nbs: break
                cur = nbs[rng.randint(len(nbs))]
                walk.append(cur)
            walks.append(walk)
    return walks

def generate_node2vec_walks(G, num_walks=80, walk_length=10, p=1.0, q=1.0, seed=0):
    rng = np.random.RandomState(seed)
    walks = []
    for node in sorted(G.nodes()):
        for _ in range(num_walks):
            walk = [node]
            prev, cur = None, node
            for _ in range(walk_length - 1):
                nbs = list(G.neighbors(cur))
                if not nbs: break
                if prev is None:
                    nxt = nbs[rng.randint(len(nbs))]
                else:
                    ps = set(G.neighbors(prev))
                    ws = []
                    for z in nbs:
                        a = 1.0/p if z==prev else (1.0 if z in ps else 1.0/q)
                        ws.append(a * G[cur][z].get('weight', 1))
                    t = sum(ws)
                    pr = np.array(ws)/t
                    r = rng.random()
                    c, nxt = 0.0, nbs[-1]
                    for idx, p_ in enumerate(pr):
                        c += p_
                        if r <= c: nxt = nbs[idx]; break
                walk.append(nxt)
                prev, cur = cur, nxt
            walks.append(walk)
    return walks

def train_word2vec(walks, vector_size=2, window=5, epochs=30, seed=0, workers=1):
    model = Word2Vec(walks, vector_size=vector_size, window=window, hs=1, sg=1,
                     workers=workers, seed=seed, min_count=1, sample=0)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs, report_delay=0)
    return model.wv.vectors


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return path


def main():
    t0 = time.time()
    print("=" * 70)
    print("  STAGE A: Preprocessing, All Methods (seed=0), Figures")
    print("=" * 70)

    # Environment
    import platform
    env = {"python": platform.python_version()}
    for pkg in ["numpy", "scipy", "sklearn", "gensim", "networkx", "pandas", "matplotlib"]:
        try:
            env[pkg] = __import__(pkg).__version__
        except: pass
    env["platform"] = platform.platform()
    with open(os.path.join(OUT_DIR, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(f"  Python {env['python']}, gensim {env.get('gensim','?')}")

    # ═══ PART 1: Preprocessing ═══
    print("\n--- PART 1: Preprocessing ---")
    course_data = {}
    old_stats, new_stats = {}, {}
    prep_info = {}

    for i in FILE_INDICES:
        m_orig, labels = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        uc = find_universal_columns(m_orig)
        m_filt = np.delete(m_orig, uc, axis=1)
        G_old = create_graph(m_orig)
        G_new = create_graph(m_filt)
        old_stats[str(i)] = graph_stats(G_old)
        new_stats[str(i)] = graph_stats(G_new)
        course_data[i] = {"matrix": m_filt, "labels": labels, "graph": G_new}
        prep_info[str(i)] = {
            "n_students": m_orig.shape[0], "n_courses_orig": m_orig.shape[1],
            "n_courses_filtered": m_filt.shape[1], "n_removed": len(uc),
            "removed_columns": uc, "target_course_removed": 0 in uc,
        }
        print(f"  Course {i}: {m_orig.shape[0]} students, {m_orig.shape[1]}→{m_filt.shape[1]} courses, "
              f"density {old_stats[str(i)]['density']:.4f}→{new_stats[str(i)]['density']:.4f}, "
              f"complete={new_stats[str(i)]['is_complete']}, components={new_stats[str(i)]['n_components']}")

    with open(os.path.join(OUT_DIR, "preprocessing_info.json"), "w") as f:
        json.dump(prep_info, f, indent=2)
    with open(os.path.join(OUT_DIR, "graph_analysis.json"), "w") as f:
        json.dump({"old": old_stats, "new": new_stats}, f, indent=2)

    # Figure: graph density
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(6); w = 0.35
    courses = [str(i) for i in range(1, 7)]
    ax.bar(x-w/2, [old_stats[c]["density"] for c in courses], w, label='Original', color='#E53935', alpha=0.8)
    ax.bar(x+w/2, [new_stats[c]["density"] for c in courses], w, label='Revised', color='#43A047', alpha=0.8)
    ax.set_xlabel('Course'); ax.set_ylabel('Graph Density')
    ax.set_title('Effect of Removing Constant Courses on Graph Density')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in range(1,7)])
    ax.legend(); ax.set_ylim(0, 1.15); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    save_fig(fig, "graph_density_comparison")
    print("  Saved: graph_density_comparison.pdf")

    # ═══ PART 2: All Methods (seed=0) ═══
    print("\n--- PART 2: All Methods (seed=0) ---")
    all_results = []
    runtime_data = []

    for i in FILE_INDICES:
        cd = course_data[i]
        m, G, n = cd["matrix"], cd["graph"], cd["matrix"].shape[0]
        print(f"\n  Course {i}: {n} students, {m.shape[1]} courses, {G.number_of_edges()} edges")

        # BoW
        t_s = time.perf_counter()
        bl = cluster_kmeans(m.astype(float), seed=0)
        t_bow = time.perf_counter() - t_s
        bm = compute_metrics(m.astype(float), bl)
        print(f"    BoW+KMeans:    Sil={bm['silhouette']:.3f}, DBI={bm['dbi']:.3f}, CH={bm['ch']:.1f}")
        all_results.append({"course": i, "method": "bow",
            "silhouette_emb": bm["silhouette"], "dbi_emb": bm["dbi"], "ch_emb": bm["ch"],
            "silhouette_bow": bm["silhouette"], "dbi_bow": bm["dbi"], "ch_bow": bm["ch"]})
        runtime_data.append({"course": i, "method": "BoW+KMeans", "t_total": t_bow})

        # PCA
        t_s = time.perf_counter()
        pl, pr = cluster_pca_kmeans(m.astype(float), seed=0)
        t_pca = time.perf_counter() - t_s
        pm_bow = compute_metrics(m.astype(float), pl)
        pm_pca = compute_metrics(pr, pl)
        print(f"    PCA+KMeans:    Sil={pm_bow['silhouette']:.3f}, DBI={pm_bow['dbi']:.3f}, CH={pm_bow['ch']:.1f}")
        all_results.append({"course": i, "method": "pca",
            "silhouette_emb": pm_pca["silhouette"], "dbi_emb": pm_pca["dbi"], "ch_emb": pm_pca["ch"],
            "silhouette_bow": pm_bow["silhouette"], "dbi_bow": pm_bow["dbi"], "ch_bow": pm_bow["ch"]})
        runtime_data.append({"course": i, "method": "PCA+KMeans", "t_total": t_pca})

        # Spectral
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        t_s = time.perf_counter()
        sl = cluster_spectral(adj, seed=0)
        t_spec = time.perf_counter() - t_s
        sm_bow = compute_metrics(m.astype(float), sl)
        sm_adj = compute_metrics(adj, sl)
        print(f"    Spectral:      Sil={sm_bow['silhouette']:.3f}, DBI={sm_bow['dbi']:.3f}, CH={sm_bow['ch']:.1f}")
        all_results.append({"course": i, "method": "spectral",
            "silhouette_emb": sm_adj["silhouette"], "dbi_emb": sm_adj["dbi"], "ch_emb": sm_adj["ch"],
            "silhouette_bow": sm_bow["silhouette"], "dbi_bow": sm_bow["dbi"], "ch_bow": sm_bow["ch"]})
        runtime_data.append({"course": i, "method": "Spectral", "t_total": t_spec})

        # DeepWalk
        t_s = time.perf_counter()
        dw_walks = generate_unbiased_walks(G, seed=0)
        dw_emb = train_word2vec(dw_walks, seed=0)
        dw_labels = cluster_kmeans(dw_emb, seed=0)
        t_dw = time.perf_counter() - t_s
        dw_m_emb = compute_metrics(dw_emb, dw_labels)
        dw_m_bow = compute_metrics(m.astype(float), dw_labels)
        print(f"    DeepWalk:      Sil={dw_m_emb['silhouette']:.3f}, DBI={dw_m_emb['dbi']:.3f}, CH={dw_m_emb['ch']:.1f}")
        all_results.append({"course": i, "method": "deepwalk",
            "silhouette_emb": dw_m_emb["silhouette"], "dbi_emb": dw_m_emb["dbi"], "ch_emb": dw_m_emb["ch"],
            "silhouette_bow": dw_m_bow["silhouette"], "dbi_bow": dw_m_bow["dbi"], "ch_bow": dw_m_bow["ch"]})
        runtime_data.append({"course": i, "method": "DeepWalk", "t_total": t_dw})

        # Node2Vec(p=1,q=1)
        t_s = time.perf_counter()
        n2v_walks = generate_node2vec_walks(G, p=1.0, q=1.0, seed=0)
        n2v_emb = train_word2vec(n2v_walks, seed=0)
        n2v_labels = cluster_kmeans(n2v_emb, seed=0)
        t_n2v = time.perf_counter() - t_s
        n2v_m_emb = compute_metrics(n2v_emb, n2v_labels)
        n2v_m_bow = compute_metrics(m.astype(float), n2v_labels)
        print(f"    Node2Vec(1,1): Sil={n2v_m_emb['silhouette']:.3f}, DBI={n2v_m_emb['dbi']:.3f}, CH={n2v_m_emb['ch']:.1f}")
        all_results.append({"course": i, "method": "node2vec",
            "silhouette_emb": n2v_m_emb["silhouette"], "dbi_emb": n2v_m_emb["dbi"], "ch_emb": n2v_m_emb["ch"],
            "silhouette_bow": n2v_m_bow["silhouette"], "dbi_bow": n2v_m_bow["dbi"], "ch_bow": n2v_m_bow["ch"]})
        runtime_data.append({"course": i, "method": "Node2Vec(1,1)", "t_total": t_n2v})

    pd.DataFrame(all_results).to_json(os.path.join(OUT_DIR, "all_methods.json"), orient="records", indent=2)
    pd.DataFrame(all_results).to_excel(os.path.join(OUT_DIR, "all_methods.xlsx"), index=False)
    pd.DataFrame(runtime_data).to_json(os.path.join(OUT_DIR, "runtime.json"), orient="records", indent=2)

    # Summary
    print("\n  SUMMARY (Silhouette Score, seed=0)")
    print(f"  {'Method':<20}", end="")
    for i in FILE_INDICES: print(f" {'C'+str(i):>6}", end="")
    print(f" {'Avg':>8}")
    print("  " + "-" * 60)
    for mk, ml in [("bow","BoW+KMeans"), ("pca","PCA+KMeans"), ("spectral","Spectral"),
                    ("deepwalk","DeepWalk"), ("node2vec","Node2Vec(1,1)")]:
        sub = [d for d in all_results if d["method"] == mk]
        print(f"  {ml:<20}", end="")
        vals = []
        for i in FILE_INDICES:
            row = [d for d in sub if d["course"] == i]
            v = row[0]["silhouette_bow"] if mk in ("bow","pca","spectral") else row[0]["silhouette_emb"]
            vals.append(v)
            print(f" {v:>6.3f}", end="")
        print(f" {np.mean(vals):>8.3f}")

    # Method comparison figure (3 panels: Sil, DBI, CH)
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    courses_list = sorted(set(d["course"] for d in all_results))
    x = np.arange(len(courses_list))
    width = 0.15
    method_info = [("bow","BoW+KMeans",COLORS["bow"]), ("pca","PCA+KMeans",COLORS["pca"]),
                   ("spectral","Spectral",COLORS["spectral"]),
                   ("deepwalk","DeepWalk",COLORS["deepwalk"]), ("node2vec","Node2Vec(1,1)",COLORS["node2vec"])]
    for ax_idx, (met, title, _) in enumerate([
        ("silhouette", "Silhouette Score (↑)", None),
        ("dbi", "Davies-Bouldin Index (↓)", None),
        ("ch", "Calinski-Harabasz Index (↑)", None)]):
        ax = axes[ax_idx]
        for i_m, (mk, ml, mc) in enumerate(method_info):
            vals = []
            for c in courses_list:
                row = [d for d in all_results if d["course"] == c and d["method"] == mk]
                v = row[0].get(f"{met}_emb", row[0].get(f"{met}_bow", 0)) if row else 0
                vals.append(v)
            offset = (i_m - 2) * width
            bars = ax.bar(x + offset, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h) and h != 0:
                    fmt = f'{h:.1f}' if met == "ch" else f'{h:.3f}'
                    ax.annotate(fmt, xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3),
                                textcoords="offset points", ha='center', va='bottom', fontsize=5.5)
        ax.set_xlabel('Course'); ax.set_ylabel(title)
        ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in courses_list])
        if ax_idx == 0: ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle('Method Comparison: All Metrics (Revised Graph Construction)', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "method_comparison_all_metrics")
    print("  Saved: method_comparison_all_metrics.pdf")

    # Runtime figure
    fig, ax = plt.subplots(figsize=(9, 5))
    for i_m, (ml, mc) in enumerate([("BoW+KMeans",COLORS["bow"]), ("PCA+KMeans",COLORS["pca"]),
                                      ("Spectral",COLORS["spectral"]), ("DeepWalk",COLORS["deepwalk"]),
                                      ("Node2Vec(1,1)",COLORS["node2vec"])]):
        vals = [d["t_total"] for d in runtime_data if d["method"] == ml]
        offset = (i_m - 2) * 0.15
        ax.bar(x + offset, vals, 0.15, label=ml, color=mc, alpha=0.85)
    ax.set_xlabel('Course'); ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime Comparison'); ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses_list])
    ax.legend(framealpha=0.9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    save_fig(fig, "runtime_comparison")
    print("  Saved: runtime_comparison.pdf")

    # Save course_data for next stages
    save_data = {}
    for i in FILE_INDICES:
        cd = course_data[i]
        save_data[str(i)] = {
            "matrix": cd["matrix"].tolist(),
            "n_students": cd["matrix"].shape[0],
            "n_courses": cd["matrix"].shape[1],
        }
    with open(os.path.join(OUT_DIR, "course_data_cache.json"), "w") as f:
        json.dump(save_data, f)

    print(f"\n  Stage A complete. Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
