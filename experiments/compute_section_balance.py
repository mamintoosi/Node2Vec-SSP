# -*- coding: utf-8 -*-
"""
Compute Section Balance Values
==============================
Computes section balance metric B for all 6 courses using Node2Vec pipeline.
B = min(|S1|, |S2|) / max(|S1|, |S2|)
"""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans

# Add experiments directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FILE_PATHS, FILE_INDICES, DEFAULT_PARAMS
from core import read_class, create_graph_from_bow, generate_node2vec_walks, train_word2vec


def compute_section_balance(labels, n_clusters=2):
    """
    Compute section balance metric B.
    
    B = min(|S1|, |S2|) / max(|S1|, |S2|)
    
    Args:
        labels: Cluster labels for each student.
        n_clusters: Number of clusters/sections.
    
    Returns:
        Section balance value B (1.0 = perfectly balanced).
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if len(unique_labels) < 2:
        return 1.0  # Only one section
    
    # Sort counts
    counts = np.sort(counts)
    
    # For binary case: B = min / max
    if n_clusters == 2:
        return counts[0] / counts[1]
    else:
        # For k > 2: analogous ratio (min / max)
        return counts[0] / counts[-1]


def run_node2vec_and_cluster(file_path, n_clusters=2, seed=0):
    """
    Run Node2Vec pipeline and cluster students.
    
    Returns:
        Tuple of (section_balance, labels, silhouette_score)
    """
    from sklearn.metrics import silhouette_score
    
    # Load data
    student_course_matrix, student_labels = read_class(file_path)
    if student_course_matrix is None:
        return None, None, None
    
    # Preprocess: remove constant features
    constant_cols = np.all(student_course_matrix == student_course_matrix[0], axis=0)
    student_course_matrix = student_course_matrix[:, ~constant_cols]
    
    # Build graph
    G = create_graph_from_bow(student_course_matrix)
    
    # Generate Node2Vec walks
    walks = generate_node2vec_walks(
        G,
        num_walks_per_node=DEFAULT_PARAMS['num_walks'],
        walk_length=DEFAULT_PARAMS['walk_length'],
        p=1.0,  # Default parameters
        q=1.0,
        seed=seed
    )
    
    # Train Word2Vec
    wv_model = train_word2vec(
        walks,
        vector_size=DEFAULT_PARAMS['vector_size'],
        window=DEFAULT_PARAMS['window'],
        epochs=DEFAULT_PARAMS['epochs'],
        seed=seed
    )
    
    if wv_model is None:
        return None, None, None
    
    embeddings = wv_model.wv.vectors
    
    # Cluster with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Compute metrics
    balance = compute_section_balance(labels, n_clusters)
    silhouette = silhouette_score(embeddings, labels)
    
    return balance, labels, silhouette


def main():
    print("=" * 70)
    print("Section Balance Analysis for Student Sectioning")
    print("=" * 70)
    print()
    print("Metric: B = min(|S1|, |S2|) / max(|S1|, |S2|)")
    print("B = 1.0 means perfectly balanced sections")
    print()
    print("Using Node2Vec with default parameters (p=1.0, q=1.0)")
    print("-" * 70)
    print()
    
    results = []
    
    for idx, file_path in zip(FILE_INDICES, FILE_PATHS):
        balance, labels, silhouette = run_node2vec_and_cluster(file_path, n_clusters=2, seed=0)
        
        if balance is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            results.append({
                'course': idx,
                'balance': balance,
                'section_sizes': counts.tolist(),
                'silhouette': silhouette
            })
            print(f"Course {idx}: B = {balance:.3f} | Sections: {counts[0]}, {counts[1]} | Silhouette: {silhouette:.3f}")
        else:
            print(f"Course {idx}: Failed to compute")
    
    print()
    print("-" * 70)
    
    if results:
        avg_balance = np.mean([r['balance'] for r in results])
        min_balance = min([r['balance'] for r in results])
        max_balance = max([r['balance'] for r in results])
        
        print(f"Average Section Balance: B = {avg_balance:.3f}")
        print(f"Range: {min_balance:.3f} - {max_balance:.3f}")
        print()
        
        # Format for paper
        print("=" * 70)
        print("FOR PAPER (LaTeX format):")
        print("=" * 70)
        print()
        print("Section sizes per course:")
        for r in results:
            print(f"  Course {r['course']}: {r['section_sizes'][0]}, {r['section_sizes'][1]} students")
        print()
        print(f"Section balance metric B ranges from {min_balance:.3f} to {max_balance:.3f}")
        print(f"(average {avg_balance:.3f}), indicating {'well-balanced' if avg_balance > 0.8 else 'moderately balanced'} sections across all courses.")
    
    return results


if __name__ == "__main__":
    main()
