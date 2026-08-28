# -*- coding: utf-8 -*-
"""Shared functions for experiment stages."""
import os, sys, json
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "final_reexperiment")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
FILE_INDICES = [1, 2, 3, 4, 5, 6]


def read_class(fp):
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        for j in range(n):
            parts = f.readline().split()
            bv = parts[2]
            if len(bv) < m: bv = bv.ljust(m, '0')
            mat[j] = [int(c) for c in bv[:m]]
    return mat


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


def load_course_data():
    """Load all course data with universal columns removed."""
    course_data = {}
    for i in FILE_INDICES:
        m = read_class(os.path.join(DATA_DIR, f"{i}.txt"))
        uc = find_universal_columns(m)
        m_filt = np.delete(m, uc, axis=1)
        G = create_graph(m_filt)
        course_data[i] = {"matrix": m_filt, "graph": G, "n_students": m.shape[0]}
    return course_data
