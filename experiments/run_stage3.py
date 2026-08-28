# -*- coding: utf-8 -*-
"""
Stage 3: Statistical Analysis, Walk Validation, and Final Summary.
"""

import os, sys, time, json, warnings, random, platform
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon as scipy_wilcoxon

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REEXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "results", "revised_reexperiment")

FILE_INDICES = [1, 2, 3, 4, 5, 6]


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


def main():
    print("=" * 70)
    print("  STAGE 3: Statistical Analysis + Walk Validation")
    print("=" * 70)

    # Load stability data
    with open(os.path.join(REEXPERIMENT_DIR, "stability.json")) as f:
        stability_stats = json.load(f)

    # Load sensitivity data
    sens_df = pd.read_excel(os.path.join(REEXPERIMENT_DIR, "node2vec_sensitivity.xlsx"))

    # ═══ PART A: Walk Validation ═══
    print("\n--- Walk Validation: Node2Vec(p=1,q=1) uniform transitions ---")
    m_orig = read_class(os.path.join(DATA_DIR, "1.txt"))
    uc = find_universal_columns(m_orig)
    m = np.delete(m_orig, uc, axis=1)
    G = create_graph(m)

    node = 0
    neighbors = list(G.neighbors(node))
    n_nb = len(neighbors)
    expected = 1.0 / n_nb

    walks = generate_node2vec_walks(G, num_walks=1000, walk_length=2, p=1.0, q=1.0, seed=42)
    first_steps = [w[1] for w in walks if w[0] == node]
    counts = Counter(first_steps)
    empirical = {k: v / len(first_steps) for k, v in counts.items()}
    max_dev = max(abs(empirical.get(nb, 0) - expected) for nb in neighbors)

    print(f"  Node {node}: {n_nb} neighbors")
    print(f"  Expected probability: {expected:.4f}")
    print(f"  Max deviation: {max_dev:.4f}")
    print(f"  Walks analyzed: {len(first_steps)}")

    valid = max_dev < 0.05
    print(f"  {'✓ PASSED' if valid else '✗ WARNING'}: Node2Vec(p=1,q=1) {'approximately' if valid else 'does not'} produce uniform transitions")

    with open(os.path.join(REEXPERIMENT_DIR, "walk_validation.json"), "w") as f:
        json.dump({"node": node, "n_neighbors": n_nb, "expected_prob": expected,
                   "max_deviation": max_dev, "n_walks": len(first_steps), "valid": valid,
                   "empirical_probs": {str(k): v for k, v in empirical.items()}}, f, indent=2)

    # ═══ PART B: Statistical Analysis ═══
    print("\n--- Statistical Analysis ---")

    # Node2Vec mean silhouettes across seeds (from stability analysis)
    n2v_means = np.array([stability_stats[str(c)]["silhouette"]["mean"] for c in FILE_INDICES])

    # Deterministic baselines with revised graphs
    bow_scores, pca_scores, spec_scores = [], [], []
    for i in FILE_INDICES:
        m_orig = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        uc = find_universal_columns(m_orig)
        m = np.delete(m_orig, uc, axis=1)

        bow_l = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(m.astype(float))
        bow_scores.append(float(silhouette_score(m.astype(float), bow_l)))

        reduced = PCA(n_components=2, random_state=0).fit_transform(m.astype(float))
        pca_l = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        pca_scores.append(float(silhouette_score(m.astype(float), pca_l)))

        G = create_graph(m)
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        spec_l = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0,
                                    assign_labels='kmeans').fit_predict(adj)
        spec_scores.append(float(silhouette_score(m.astype(float), spec_l)))

    bow_scores = np.array(bow_scores)
    pca_scores = np.array(pca_scores)
    spec_scores = np.array(spec_scores)

    print(f"\n  Course-level scores:")
    print(f"  {'Course':<10} {'Node2Vec':>10} {'BoW':>10} {'PCA':>10} {'Spectral':>10}")
    print("  " + "-" * 50)
    for ci, c in enumerate(FILE_INDICES):
        print(f"  Course {c:<4} {n2v_means[ci]:>10.3f} {bow_scores[ci]:>10.3f} "
              f"{pca_scores[ci]:>10.3f} {spec_scores[ci]:>10.3f}")
    print(f"  {'Average':<10} {np.mean(n2v_means):>10.3f} {np.mean(bow_scores):>10.3f} "
          f"{np.mean(pca_scores):>10.3f} {np.mean(spec_scores):>10.3f}")

    # Pairwise Wilcoxon tests
    comparisons = [
        ("Node2Vec vs BoW+KMeans", n2v_means, bow_scores),
        ("Node2Vec vs PCA+KMeans", n2v_means, pca_scores),
        ("Node2Vec vs Spectral", n2v_means, spec_scores),
    ]

    stat_results = []
    raw_p = []

    for name, x, y in comparisons:
        try:
            stat, p = scipy_wilcoxon(x, y, alternative='two-sided')
            stat_val, p_val = float(stat), float(p)
        except ValueError:
            stat_val, p_val = float('nan'), float('nan')

        # Effect size r
        n = len(x)
        r = np.nan
        if n >= 5 and not np.isnan(stat_val):
            mean_w = n * (n + 1) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            if std_w > 0:
                z = (stat_val - mean_w) / std_w
                r = min(abs(z) / np.sqrt(n), 1.0)

        # Cliff's delta
        dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
        delta = dom / (n * n)

        raw_p.append(p_val)
        res = {
            "comparison": name,
            "n2v_mean": float(np.mean(x)),
            "other_mean": float(np.mean(y)),
            "mean_diff": float(np.mean(x - y)),
            "wilcoxon_stat": stat_val,
            "wilcoxon_p": p_val,
            "effect_size_r": r,
            "cliffs_delta": delta,
        }
        stat_results.append(res)

        print(f"\n  {name}:")
        print(f"    N2V: {np.mean(x):.3f}, Other: {np.mean(y):.3f}, diff: {np.mean(x-y):+.3f}")
        print(f"    Wilcoxon: stat={stat_val:.1f}, p={p_val:.6f}")
        print(f"    Effect size r={r:.3f}, Cliff's delta={delta:.3f}")

    # Holm correction
    def holm_correction(p_values):
        p = np.array(p_values)
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

    corrected_p = holm_correction(raw_p)
    for i_r, (cp, res) in enumerate(zip(corrected_p, stat_results)):
        res["corrected_p"] = cp
        res["significant"] = cp < 0.05
        print(f"\n  {res['comparison']}:")
        print(f"    Raw p = {res['wilcoxon_p']:.6f}, Holm-corrected p = {cp:.6f}")
        print(f"    {'Significant' if cp < 0.05 else 'Not significant'} at α=0.05")

    with open(os.path.join(REEXPERIMENT_DIR, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)

    # ═══ PART C: Sensitivity Summary ═══
    print("\n--- Node2Vec Sensitivity Summary ---")
    pq_means = sens_df.groupby(["p", "q"])["silhouette_emb"].mean()
    print(f"  {'p':>4} {'q':>4} {'Avg Silhouette':>15}")
    print("  " + "-" * 25)
    for (p, q), val in pq_means.items():
        print(f"  {p:>4.1f} {q:>4.1f} {val:>15.3f}")

    best = pq_means.idxmax()
    print(f"\n  Best: p={best[0]}, q={best[1]}, avg Silhouette={pq_means.max():.3f}")
    print(f"  Range: {pq_means.min():.3f} to {pq_means.max():.3f}")
    print(f"  Sensitivity (q effect at p=1): {pq_means[(1.0, 0.5)] - pq_means[(1.0, 2.0)]:+.4f}")

    # ═══ PART D: Environment & Final Summary ═══
    print("\n--- Environment ---")
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy; env["numpy"] = numpy.__version__
        import scipy; env["scipy"] = scipy.__version__
        import sklearn; env["sklearn"] = sklearn.__version__
        import gensim; env["gensim"] = gensim.__version__
        import networkx; env["networkx"] = networkx.__version__
        import pandas; env["pandas"] = pandas.__version__
        import matplotlib; env["matplotlib"] = matplotlib.__version__
    except: pass

    for k, v in env.items():
        print(f"  {k}: {v}")

    with open(os.path.join(REEXPERIMENT_DIR, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    # Save sensitivity summary
    with open(os.path.join(REEXPERIMENT_DIR, "sensitivity_summary.json"), "w") as f:
        json.dump({
            "pq_means": {f"p{p}_q{q}": float(v) for (p, q), v in pq_means.items()},
            "best_config": {"p": float(best[0]), "q": float(best[1]), "silhouette": float(pq_means.max())},
        }, f, indent=2)

    print("\n  Stage 3 complete.")


if __name__ == "__main__":
    main()
