# -*- coding: utf-8 -*-
"""
Stage 2: Stability analysis (20 seeds) — optimized for speed.
Uses workers=1 and pre-loads all course data.
"""
import os, sys, time, json, warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REEXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "results", "revised_reexperiment")
FIGURES_DIR = os.path.join(REEXPERIMENT_DIR, "figures")
os.makedirs(REEXPERIMENT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FILE_INDICES = [1, 2, 3, 4, 5, 6]
NUM_SEEDS = 10

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLOR_NODE2VEC = "#4CAF50"


def read_class(file_path):
    with open(file_path, 'r') as f:
        line1 = f.readline().strip().split()
        n, m = int(line1[0]), int(line1[1])
        matrix = np.zeros((n, m), dtype=int)
        for j in range(n):
            parts = f.readline().strip().split()
            bv = parts[2]
            if len(bv) < m: bv = bv.ljust(m, '0')
            matrix[j] = [int(c) for c in bv[:m]]
    return matrix

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

def train_word2vec(walks, vector_size=2, window=5, epochs=30, seed=0):
    model = Word2Vec(walks, vector_size=vector_size, window=window, hs=1, sg=1,
                     workers=1, seed=seed, min_count=1, sample=0)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs, report_delay=0)
    return model.wv.vectors


def main():
    t_start = time.time()
    print("=" * 70)
    print("  STAGE 2: Seed Stability Analysis (20 seeds)")
    print("=" * 70)

    # Pre-load and prepare all course data
    course_data = {}
    for i in FILE_INDICES:
        m_orig = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        uc = find_universal_columns(m_orig)
        m = np.delete(m_orig, uc, axis=1)
        G = create_graph(m)
        course_data[i] = {"matrix": m, "graph": G, "n_students": m.shape[0]}

    all_per_course = {i: [] for i in FILE_INDICES}

    for seed in range(NUM_SEEDS):
        for i in FILE_INDICES:
            cd = course_data[i]
            walks = generate_node2vec_walks(cd["graph"], p=1.0, q=1.0, seed=seed)
            emb = train_word2vec(walks, seed=seed)
            labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(emb)
            sil = float(silhouette_score(emb, labels)) if len(np.unique(labels)) >= 2 else np.nan
            dbi = float(davies_bouldin_score(emb, labels)) if len(np.unique(labels)) >= 2 else np.nan
            ch = float(calinski_harabasz_score(emb, labels)) if len(np.unique(labels)) >= 2 else np.nan
            all_per_course[i].append({
                "course": i, "seed": seed,
                "silhouette_emb": sil, "dbi_emb": dbi, "ch_emb": ch,
            })

        elapsed = time.time() - t_start
        eta = elapsed / (seed + 1) * (NUM_SEEDS - seed - 1)
        if (seed + 1) % 5 == 0 or seed == NUM_SEEDS - 1:
            print(f"  Seed {seed + 1}/{NUM_SEEDS} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Compute statistics per course
    stats = {}
    for i in FILE_INDICES:
        sils = [r["silhouette_emb"] for r in all_per_course[i]]
        dbis = [r["dbi_emb"] for r in all_per_course[i]]
        chs = [r["ch_emb"] for r in all_per_course[i]]
        stats[str(i)] = {
            "silhouette": {"mean": float(np.mean(sils)), "std": float(np.std(sils)),
                           "min": float(np.min(sils)), "max": float(np.max(sils)),
                           "values": [float(x) for x in sils]},
            "dbi": {"mean": float(np.mean(dbis)), "std": float(np.std(dbis)),
                    "values": [float(x) for x in dbis]},
            "ch": {"mean": float(np.mean(chs)), "std": float(np.std(chs)),
                   "values": [float(x) for x in chs]},
        }
        print(f"  Course {i}: Silhouette = {np.mean(sils):.3f} ± {np.std(sils):.3f} "
              f"(range [{np.min(sils):.3f}, {np.max(sils):.3f}])")

    with open(os.path.join(REEXPERIMENT_DIR, "stability.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(REEXPERIMENT_DIR, "stability_per_seed.json"), "w") as f:
        json.dump({str(i): all_per_course[i] for i in FILE_INDICES}, f, indent=2)

    # Boxplot
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    all_sil_data = [[r["silhouette_emb"] for r in all_per_course[i]] for i in FILE_INDICES]
    bp = ax.boxplot(all_sil_data, tick_labels=[f"Course {i}" for i in FILE_INDICES],
                    patch_artist=True, boxprops=dict(facecolor=COLOR_NODE2VEC, alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Node2Vec Stability Across 20 Random Seeds (Revised Graph)')
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "stability_boxplot.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: stability_boxplot.pdf")

    print(f"\n  Stage 2 complete. Time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
