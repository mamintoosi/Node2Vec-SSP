#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inter-section Course Conflict Potential (shared-course overlap) analysis.

Computes a task-oriented sectioning metric from EXISTING artifacts:
  - Node2Vec cluster labels: results/grid_search/labels_<config>_course<i>.json
    (primary p=1.0,q=0.5,d=2; quality-optimal p=2.0,q=1.0,d=1; neutral p=1.0,q=1.0,d=2; seed 42)
  - BoW / PCA / Spectral labels: NOT archived anywhere -> recomputed here with the
    EXACT protocol of experiments/final_d1_d2_experiments.py (seed 42, KMeans
    n_init=10, PCA n_components=2, spectral = KMeans on the weighted adjacency
    rows) and saved as new label artifacts for future reuse.

Metric (per course instance, target course = the single universal column):
  s_ij = number of shared non-target courses of students i, j
       = (M_f M_f^T)_ij on the constant-feature-filtered matrix  [== graph edge weight]
  A: C_A = pair-weighted mean of s_ij over within-section pairs (both sections pooled)
  B: C_B = pair-weighted fraction of within-section pairs with s_ij >= 1
  Reference: the same statistics over ALL pairs = expectation under a random split.
  Gain: C_A / E_random[A]  (size-independent, >1 means sections group students
        with more shared courses than a random split would).

No Node2Vec/walks/Word2Vec is re-run. Outputs are written ONLY to
results/conflict_potential/.
"""
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
GRID = os.path.join(ROOT, "results", "grid_search")
OUT = os.path.join(ROOT, "results", "conflict_potential")
COURSES = [1, 2, 3, 4, 5, 6]
SEED = 42  # SEED of the archived experiments (experiments/final_d1_d2_experiments.py)

N2V_CONFIGS = {
    "n2v_primary": "labels_p1.0_q0.5_d2_course{i}.json",
    "n2v_quality": "labels_p2.0_q1.0_d1_course{i}.json",
    "n2v_neutral": "labels_p1.0_q1.0_d2_course{i}.json",
}
METHOD_NAMES = {
    "bow": "BoW + KMeans",
    "pca": "PCA + KMeans",
    "spec": "Spectral",
    "n2v_primary": "Node2Vec (p=1.0, q=0.5, d=2)",
    "n2v_quality": "Node2Vec (p=2.0, q=1.0, d=1)",
    "n2v_neutral": "Node2Vec (p=1.0, q=1.0, d=2)",
}


def read_class(fp):
    """Identical to experiments/final_d1_d2_experiments.read_class."""
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        ids = []
        for j in range(n):
            p = f.readline().split()
            ids.append(p[1])
            bv = p[2]
            if len(bv) < m:
                bv = bv.ljust(m, "0")
            mat[j] = [int(c) for c in bv[:m]]
    return mat, ids


def find_univ(m):
    """Identical to experiments/final_d1_d2_experiments.find_univ."""
    return np.where(m.sum(axis=0) == m.shape[0])[0].tolist()


def km(data, seed=0):
    """Identical to experiments/final_d1_d2_experiments.km."""
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(data)


# ---------------------------------------------------------------- metric ----
def pair_stats(S, labels):
    """Within-section pair statistics for a 2-section partition.

    S: n x n matrix of shared non-target courses (s_ij).
    Returns dict with section-wise and pooled (pair-weighted) statistics.
    """
    n = len(labels)
    idx = np.arange(n)
    same = labels[:, None] == labels[None, :]
    iu, ju = np.triu_indices(n, 1)
    mask = same[iu, ju]
    w_pairs = S[iu[mask], ju[mask]]
    n_pairs = int(mask.sum())

    sections = {}
    for c in (0, 1):
        sel = idx[labels == c]
        if len(sel) < 2:
            sections[c] = {"size": int(len(sel)), "n_pairs": 0,
                           "C_A": None, "C_B": None}
            continue
        su, juu = np.triu_indices(len(sel), 1)
        sp = S[sel[su], sel[juu]]
        sections[c] = {
            "size": int(len(sel)),
            "n_pairs": int(len(sp)),
            "C_A": float(sp.mean()),
            "C_B": float((sp >= 1).mean()),
        }

    all_pairs = S[iu, ju]
    return {
        "n_within_pairs": n_pairs,
        "n_all_pairs": int(len(all_pairs)),
        "sections": sections,
        "C_A": float(w_pairs.mean()) if n_pairs else None,
        "C_B": float((w_pairs >= 1).mean()) if n_pairs else None,
        # random-split expectation (exchangeability): statistics over ALL pairs
        "random_C_A": float(all_pairs.mean()),
        "random_C_B": float((all_pairs >= 1).mean()),
        "gain_A": (float(w_pairs.mean()) / float(all_pairs.mean())
                   if n_pairs and all_pairs.mean() > 0 else None),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"metadata": {
        "date": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "definition": {
            "s_ij": "number of shared non-target courses = (Mf Mf^T)_ij on the "
                    "constant-feature-filtered matrix (identical to the co-enrollment "
                    "graph edge weight)",
            "C_A": "pair-weighted mean of s_ij over within-section pairs (both sections pooled)",
            "C_B": "pair-weighted fraction of within-section pairs with s_ij >= 1",
            "random_*": "same statistics over all pairs = expectation under a random split",
            "gain_A": "C_A / random_C_A",
        },
    }, "courses": {}, "validation": {}}

    for i in COURSES:
        mat, ids = read_class(os.path.join(DATA, f"{i}.txt"))
        uc = find_univ(mat)
        assert len(uc) == 1, f"course {i}: expected exactly 1 constant (target) column, got {uc}"
        mf = np.delete(mat, uc, axis=1).astype(float)  # pipeline preprocessing
        S = mf @ mf.T
        np.fill_diagonal(S, 0)

        # --- labels -------------------------------------------------------
        labels_by_method = {}

        # baselines: recompute with the exact archived protocol, then SAVE labels
        l = km(mf, seed=SEED)
        labels_by_method["bow"] = l
        sil = float(silhouette_score(mf, l))
        results["validation"].setdefault("bow", {})[str(i)] = sil
        _save_labels("bow", i, ids, l, sil)

        red = PCA(n_components=2, random_state=SEED).fit_transform(mf)
        l = km(red, seed=SEED)
        labels_by_method["pca"] = l
        sil = float(silhouette_score(red, l))
        results["validation"].setdefault("pca", {})[str(i)] = sil
        _save_labels("pca", i, ids, l, sil)

        # spectral baseline of the pipeline: real SpectralClustering on the
        # weighted co-enrollment adjacency (experiments/run_all.py, spec())
        adj = S.copy()  # == nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        l = SpectralClustering(n_clusters=2, affinity="precomputed",
                               random_state=SEED, assign_labels="kmeans").fit_predict(adj)
        labels_by_method["spec"] = l
        sil = float(silhouette_score(mf, l))  # pipeline scores spec on mf
        results["validation"].setdefault("spec", {})[str(i)] = sil
        _save_labels("spec", i, ids, l, sil)

        # Node2Vec configs: labels loaded from the archived grid-search artifacts
        for key, pattern in N2V_CONFIGS.items():
            fp = os.path.join(GRID, pattern.format(i=i))
            with open(fp) as f:
                g = json.load(f)
            assert g["seed"] == SEED and g["course"] == i
            assert list(g["student_labels"]) == list(ids), f"student order mismatch {key} course {i}"
            labels_by_method[key] = np.array(g["cluster_labels"])
            results["validation"].setdefault(key, {})[str(i)] = g["silhouette"]

        # --- metric -------------------------------------------------------
        results["courses"][str(i)] = {}
        for key, labels in labels_by_method.items():
            results["courses"][str(i)][key] = pair_stats(S, np.asarray(labels))

        print(f"course {i}: done")

    # ------------------------------------------------------------ summary ---
    print("\n=== Conflict potential C_A (mean shared non-target courses, within-section pairs) ===")
    print("course | " + " | ".join(f"{METHOD_NAMES[k]:>28s}" for k in METHOD_NAMES))
    for i in COURSES:
        row = [results["courses"][str(i)][k]["C_A"] for k in METHOD_NAMES]
        print(f"     {i} | " + " | ".join(f"{v:28.3f}" for v in row))
    means = {k: np.mean([results["courses"][str(c)][k]["C_A"] for c in COURSES])
             for k in METHOD_NAMES}
    print(f"  mean  | " + " | ".join(f"{means[k]:28.3f}" for k in METHOD_NAMES))
    print(f"  median| " + " | ".join(
        f"{np.median([results['courses'][str(c)][k]['C_A'] for c in COURSES]):28.3f}"
        for k in METHOD_NAMES))

    print("\n=== C_B (fraction of within-section pairs sharing >=1 non-target course) ===")
    for i in COURSES:
        row = [results["courses"][str(i)][k]["C_B"] for k in METHOD_NAMES]
        print(f"     {i} | " + " | ".join(f"{v:28.3f}" for v in row))
    print(f"  mean  | " + " | ".join(
        f"{np.mean([results['courses'][str(c)][k]['C_B'] for c in COURSES]):28.3f}"
        for k in METHOD_NAMES))

    print("\n=== gain_A (C_A relative to random-split expectation; >1 = above random) ===")
    for i in COURSES:
        row = [results["courses"][str(i)][k]["gain_A"] for k in METHOD_NAMES]
        print(f"     {i} | " + " | ".join(f"{v:28.3f}" for v in row))
    print(f"  mean  | " + " | ".join(
        f"{np.mean([results['courses'][str(c)][k]['gain_A'] for c in COURSES]):28.3f}"
        for k in METHOD_NAMES))

    # win counts of the primary config vs each other method on C_A
    print("\n=== per-course C_A win count of n2v_primary (out of 6) ===")
    for other in METHOD_NAMES:
        if other == "n2v_primary":
            continue
        wins = sum(results["courses"][str(c)]["n2v_primary"]["C_A"]
                   > results["courses"][str(c)][other]["C_A"] for c in COURSES)
        rel = (means["n2v_primary"] - means[other]) / means[other] * 100
        print(f"  vs {METHOD_NAMES[other]:<30s}: {wins}/6   mean diff {rel:+.1f}%")

    with open(os.path.join(OUT, "conflict_potential.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT, 'conflict_potential.json')}")
    print(f"Saved baseline label files: labels_bow/pca/spec_course<i>.json in {OUT}")


def _save_labels(method, course, ids, labels, sil):
    """Persist recomputed baseline labels so future analyses need no reruns."""
    with open(os.path.join(OUT, f"labels_{method}_course{course}.json"), "w") as f:
        json.dump({
            "method": method, "course": course, "seed": SEED,
            "student_index": list(range(len(ids))),
            "student_labels": list(ids),
            "cluster_labels": [int(x) for x in labels],
            "silhouette": sil,
        }, f, indent=2)


if __name__ == "__main__":
    main()
