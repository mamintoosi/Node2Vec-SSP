# -*- coding: utf-8 -*-
"""
Core Pipeline Functions for DeepWalk-SSP
==========================================
Functions for data loading, graph construction, random walks, and embedding.
"""

import os
import time
import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


def set_seed(seed=0):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_class(file_path):
    """
    Reads a student-course data file and returns the BoW matrix and student labels.
    
    File format:
    - Line 1: num_students num_courses
    - Subsequent lines: index student_id binary_vector
    
    Args:
        file_path: Path to the data file.
        
    Returns:
        Tuple of (student_course_matrix, student_labels) or (None, None) on error.
    """
    try:
        with open(file_path, 'r') as f:
            line1 = f.readline().strip().split()
            num_students = int(line1[0])
            num_courses = int(line1[1])

            student_course_matrix = np.zeros((num_students, num_courses), dtype=int)
            student_labels = []

            for j in range(num_students):
                line = f.readline().strip().split()
                student_id = line[1]
                student_labels.append(student_id)
                course_vector_str = line[2]

                if len(course_vector_str) != num_courses:
                    course_vector_str = course_vector_str.ljust(num_courses, '0')

                course_vector = np.array([int(c) for c in course_vector_str])
                student_course_matrix[j, :] = course_vector

        return student_course_matrix, student_labels

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")
        return None, None


def create_graph_from_bow(student_course_matrix):
    """
    Creates a co-enrollment NetworkX graph from a BoW matrix.
    
    Nodes represent students; edges connect students sharing at least one course.
    Edge weight = number of shared courses.
    
    Args:
        student_course_matrix: NumPy array (num_students x num_courses).
        
    Returns:
        NetworkX Graph or None if input is invalid.
    """
    if not isinstance(student_course_matrix, np.ndarray) or student_course_matrix.ndim != 2:
        return None

    G = nx.Graph()
    num_nodes = student_course_matrix.shape[0]

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            common_courses = int(np.sum(student_course_matrix[i] * student_course_matrix[j]))
            if common_courses > 0:
                G.add_edge(i, j, weight=common_courses)

    return G


def generate_random_walks(G, num_walks_per_node=80, walk_length=10, seed=None):
    """
    Generates random walks from a graph.
    
    Args:
        G: NetworkX graph.
        num_walks_per_node: Number of walks per starting node.
        walk_length: Length of each walk.
        seed: Random seed for reproducibility.
        
    Returns:
        List of walks (each walk is a list of node IDs).
    """
    if not isinstance(G, nx.Graph):
        return []

    rng = np.random.RandomState(seed)
    walks = []

    def random_walk(start_node, walk_length):
        walk = [start_node]
        current_node = start_node
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                next_node = neighbors[rng.randint(len(neighbors))]
                walk.append(next_node)
                current_node = next_node
            else:
                break
        return walk

    # Sort nodes for deterministic walk order
    for node in sorted(G.nodes()):
        for _ in range(num_walks_per_node):
            walks.append(random_walk(node, walk_length))

    return walks


def train_word2vec(walks, vector_size=2, window=5, hs=1, sg=1, 
                   workers=1, seed=None, epochs=30):
    """
    Trains a Word2Vec model on random walks.
    
    Args:
        walks: List of walks (each walk is a list of node IDs).
        vector_size: Embedding dimension.
        window: Context window size.
        hs: Use hierarchical softmax (1=yes).
        sg: Use skip-gram (1=yes, 0=CBOW).
        workers: Number of worker threads.
        seed: Random seed.
        epochs: Number of training epochs.
        
    Returns:
        Trained gensim Word2Vec model or None on error.
    """
    if not walks:
        return None

    try:
        wv_model = Word2Vec(
            walks, 
            hs=hs, 
            sg=sg, 
            vector_size=vector_size, 
            window=window, 
            workers=workers, 
            seed=seed,
            min_count=1,
            sample=0  # Disable subsampling for reproducibility
        )
        wv_model.train(
            walks, 
            total_examples=wv_model.corpus_count, 
            epochs=epochs,
            report_delay=0
        )
        return wv_model
    except Exception as e:
        print(f"Word2Vec training error: {e}")
        return None


def generate_node2vec_walks(G, num_walks_per_node=80, walk_length=10,
                             p=1.0, q=1.0, seed=None):
    """
    Generates biased random walks following the Node2Vec algorithm.

    Unlike DeepWalk which uses uniform transition probabilities, Node2Vec
    biases the walk using two parameters:
    - p: Return parameter controlling likelihood of returning to the previous node.
    - q: In-out parameter controlling search behavior (BFS vs DFS).

    Args:
        G: NetworkX graph.
        num_walks_per_node: Number of walks per starting node.
        walk_length: Length of each walk.
        p: Return parameter (higher = less likely to return).
        q: In-out parameter (lower = more BFS-like, higher = more DFS-like).
        seed: Random seed for reproducibility.

    Returns:
        List of walks (each walk is a list of node IDs).
    """
    if not isinstance(G, nx.Graph):
        return []

    rng = np.random.RandomState(seed)
    walks = []

    # Precompute alias tables for each node's neighbors
    # alias_tables[node] = (accept, alias) arrays for sampling
    alias_tables = {}
    for node in sorted(G.nodes()):
        neighbors = list(G.neighbors(node))
        if not neighbors:
            alias_tables[node] = None
            continue
        # Compute unnormalized transition probabilities from node to its neighbors
        # when the previous node is prev
        alias_tables[node] = neighbors  # Store neighbors list; actual alias table
        # is built per step based on prev_node

    def node2vec_walk(start_node, walk_length):
        walk = [start_node]
        prev_node = None
        current_node = start_node

        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break

            if prev_node is None:
                # First step: uniform over neighbors (same as DeepWalk)
                next_node = neighbors[rng.randint(len(neighbors))]
            else:
                # Compute biased transition probabilities
                # For each neighbor z of current_node:
                #   alpha = 1/q if z == prev_node
                #   alpha = 1   if z is a common neighbor of current and prev
                #   alpha = 1/p otherwise
                # Note: In unweighted graphs, all edges have the same weight.
                # We compute the probability proportional to alpha * edge_weight.

                prev_set = set(G.neighbors(prev_node)) if prev_node is not None else set()
                weights = []
                for z in neighbors:
                    if z == prev_node:
                        alpha = 1.0 / p
                    elif z in prev_set:
                        alpha = 1.0
                    else:
                        alpha = 1.0 / q
                    # Multiply by edge weight for weighted graphs
                    edge_weight = G[current_node][z].get('weight', 1)
                    weights.append(alpha * edge_weight)

                # Normalize to probabilities
                total = sum(weights)
                probs = np.array(weights) / total

                # Sample next node
                r = rng.random()
                cumulative = 0.0
                next_node = neighbors[-1]  # fallback
                for idx, prob in enumerate(probs):
                    cumulative += prob
                    if r <= cumulative:
                        next_node = neighbors[idx]
                        break

            walk.append(next_node)
            prev_node = current_node
            current_node = next_node

        return walk

    # Sort nodes for deterministic walk order (same as DeepWalk)
    for node in sorted(G.nodes()):
        for _ in range(num_walks_per_node):
            walks.append(node2vec_walk(node, walk_length))

    return walks


def run_node2vec_pipeline(file_path, vector_size=2, walk_length=10, num_walks=80,
                          window=5, epochs=30, p=1.0, q=1.0, n_clusters=2,
                          seed=None, verbose=False):
    """
    Complete Node2Vec pipeline: load data → build graph → biased walks → embeddings.

    Args:
        file_path: Path to the data file.
        vector_size: Embedding dimension.
        walk_length: Length of each random walk.
        num_walks: Number of walks per node.
        window: Context window size for Word2Vec.
        epochs: Word2Vec training epochs.
        p: Return parameter for Node2Vec.
        q: In-out parameter for Node2Vec.
        n_clusters: Number of clusters for sectioning.
        seed: Random seed.
        verbose: If True, print progress.

    Returns:
        Dictionary with all results including timing information.
    """
    if seed is not None:
        set_seed(seed)

    result = {
        "file_path": file_path,
        "file_index": os.path.splitext(os.path.basename(file_path))[0],
        "success": False,
        "method": "Node2Vec",
        "p": p,
        "q": q,
    }

    timing = {}

    # Step 1: Read data
    student_course_matrix, student_labels = read_class(file_path)
    if student_course_matrix is None:
        return result

    result["student_labels"] = student_labels
    result["num_students"] = student_course_matrix.shape[0]
    result["num_courses"] = student_course_matrix.shape[1]

    # Step 2: Build graph (identical to DeepWalk)
    t0 = time.perf_counter()
    G = create_graph_from_bow(student_course_matrix)
    timing["graph_construction"] = time.perf_counter() - t0
    result["graph"] = G
    result["num_edges"] = G.number_of_edges()
    result["avg_degree"] = np.mean([d for _, d in G.degree()])

    # Step 3: Generate Node2Vec biased random walks
    t0 = time.perf_counter()
    walks = generate_node2vec_walks(G, num_walks_per_node=num_walks,
                                    walk_length=walk_length, p=p, q=q, seed=seed)
    timing["random_walks"] = time.perf_counter() - t0
    result["num_walks"] = len(walks)

    # Step 4: Train Word2Vec (identical to DeepWalk)
    t0 = time.perf_counter()
    wv_model = train_word2vec(walks, vector_size=vector_size, window=window,
                               epochs=epochs, seed=seed)
    timing["word2vec_training"] = time.perf_counter() - t0

    if wv_model is None:
        return result

    embeddings = wv_model.wv.vectors
    result["embeddings"] = embeddings
    result["vector_size"] = vector_size

    timing["total"] = sum(timing.values())
    result["timing"] = timing
    result["success"] = True

    return result


def run_pipeline(file_path, vector_size=2, walk_length=10, num_walks=80,
                 window=5, epochs=30, n_clusters=2, seed=None, verbose=False):
    """
    Complete DeepWalk pipeline: load data → build graph → walks → embeddings → cluster.
    
    Returns:
        Dictionary with all results including timing information.
    """
    # Set global seed for reproducibility
    if seed is not None:
        set_seed(seed)
    
    result = {
        "file_path": file_path,
        "file_index": os.path.splitext(os.path.basename(file_path))[0],
        "success": False,
    }
    
    timing = {}
    
    # Step 1: Read data
    student_course_matrix, student_labels = read_class(file_path)
    if student_course_matrix is None:
        return result
    
    result["student_labels"] = student_labels
    result["num_students"] = student_course_matrix.shape[0]
    result["num_courses"] = student_course_matrix.shape[1]
    
    # Step 2: Build graph
    t0 = time.perf_counter()
    G = create_graph_from_bow(student_course_matrix)
    timing["graph_construction"] = time.perf_counter() - t0
    result["graph"] = G
    result["num_edges"] = G.number_of_edges()
    result["avg_degree"] = np.mean([d for _, d in G.degree()])
    
    # Step 3: Generate random walks
    t0 = time.perf_counter()
    walks = generate_random_walks(G, num_walks_per_node=num_walks, 
                                   walk_length=walk_length, seed=seed)
    timing["random_walks"] = time.perf_counter() - t0
    result["num_walks"] = len(walks)
    
    # Step 4: Train Word2Vec
    t0 = time.perf_counter()
    wv_model = train_word2vec(walks, vector_size=vector_size, window=window,
                               epochs=epochs, seed=seed)
    timing["word2vec_training"] = time.perf_counter() - t0
    
    if wv_model is None:
        return result
    
    embeddings = wv_model.wv.vectors
    result["embeddings"] = embeddings
    result["vector_size"] = vector_size

    timing["total"] = sum(timing.values())
    result["timing"] = timing
    result["success"] = True

    return result


def load_existing_results(results_dir):
    """Load previously saved df.xlsx and graphs.pkl."""
    import pandas as pd
    import pickle
    
    df_path = os.path.join(results_dir, "df.xlsx")
    graphs_path = os.path.join(results_dir, "graphs.pkl")
    
    df = pd.read_excel(df_path) if os.path.exists(df_path) else None
    
    graphs = {}
    if os.path.exists(graphs_path):
        with open(graphs_path, 'rb') as f:
            data = pickle.load(f)
            graphs = data.get('all_graphs', {})
    
    return df, graphs
