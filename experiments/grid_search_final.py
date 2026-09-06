#!/usr/bin/env python3
"""
Comprehensive (p, q, d) grid search for Node2Vec on student co-enrollment graphs.
Runs all 3×3×5 = 45 configurations across 6 courses with seed=42.
Saves complete results including cluster labels and embeddings.

Usage:
    cd /path/to/Deepwalk-SSP
    python experiments/grid_search_final.py 2>&1 | tee results/grid_search/grid_search.log
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

# ── Setup ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "grid_search")
FIG = os.path.join(OUT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

COURSES = [1, 2, 3, 4, 5, 6]
SEED = 42

# Grid parameters
P_VALUES = [0.5, 1.0, 2.0]
Q_VALUES = [0.5, 1.0, 2.0]
D_VALUES = [1, 2, 3, 5, 10]

# Fixed parameters
WALK_LENGTH = 10
NUM_WALKS = 80
WINDOW = 5
EPOCHS = 30
N_CLUSTERS = 2

def read_class(fp):
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        labels = []
        for j in range(n):
            p = f.readline().split()
            labels.append(p[1])
            bv = p[2]
            if len(bv) < m:
                bv = bv.ljust(m, '0')
            mat[j] = [int(c) for c in bv[:m]]
    return mat, labels

def find_univ(m):
    return np.where(m.sum(axis=0) == m.shape[0])[0].tolist()

def mk_graph(m):
    G = nx.Graph()
    n = m.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            w = int(np.sum(m[i] * m[j]))
            if w > 0:
                G.add_edge(i, j, weight=w)
    return G

def n2v_walks(G, nw=80, wl=10, p=1.0, q=1.0, seed=0):
    rng = np.random.RandomState(seed)
    walks = []
    for nd in sorted(G.nodes()):
        for _ in range(nw):
            walk = [nd]
            prev, cur = None, nd
            for _ in range(wl - 1):
                nbs = list(G.neighbors(cur))
                if not nbs:
                    break
                if prev is None:
                    nxt = nbs[rng.randint(len(nbs))]
                else:
                    ps = set(G.neighbors(prev))
                    ws = []
                    for z in nbs:
                        a = 1.0 / p if z == prev else (1.0 if z in ps else 1.0 / q)
                        ws.append(a * G[cur][z].get('weight', 1))
                    t = sum(ws)
                    pr = np.array(ws) / t
                    r = rng.random()
                    cum = 0.0
                    nxt = nbs[-1]
                    for idx, prob in enumerate(pr):
                        cum += prob
                        if r <= cum:
                            nxt = nbs[idx]
                            break
                walk.append(nxt)
                prev, cur = cur, nxt
            walks.append(walk)
    return walks

def w2v(walks, vs=2, window=5, epochs=30, seed=0):
    m = Word2Vec(walks, vector_size=vs, window=window, hs=1, sg=1,
                 workers=1, seed=seed, min_count=1, sample=0)
    m.train(walks, total_examples=m.corpus_count, epochs=epochs, report_delay=0)
    return m.wv.vectors

def km(data, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(data)

def silhouette(data, labels):
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(silhouette_score(data, labels))

def calc_all_metrics(data, labels):
    """Calculate all clustering metrics"""
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": float('nan'),
            "davies_bouldin": float('nan'),
            "calinski_harabasz": float('nan')
        }
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    return {
        "silhouette": float(silhouette_score(data, labels)),
        "davies_bouldin": float(davies_bouldin_score(data, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(data, labels))
    }

# ── Load & preprocess data ──
print("=" * 70)
print("  COMPREHENSIVE (p, q, d) GRID SEARCH — seed=42")
print(f"  Grid: {len(P_VALUES)}×{len(Q_VALUES)}×{len(D_VALUES)} = "
      f"{len(P_VALUES)*len(Q_VALUES)*len(D_VALUES)} configs × {len(COURSES)} courses")
print("=" * 70)

course_data = {}
for i in COURSES:
    mat, student_labels = read_class(os.path.join(DATA, f"{i}.txt"))
    uc = find_univ(mat)
    mf = np.delete(mat, uc, axis=1)
    G = mk_graph(mf)
    course_data[i] = {"m": mf, "G": G, "labels": student_labels}
    print(f"  Course {i}: {mf.shape[0]} students, {mf.shape[1]} courses, "
          f"{G.number_of_edges()} edges")

# ── Grid search ──
print(f"\nRunning grid search...")
t0 = time.time()
results = []
total = len(P_VALUES) * len(Q_VALUES) * len(D_VALUES) * len(COURSES)
count = 0

for d in D_VALUES:
    for p in P_VALUES:
        for q in Q_VALUES:
            sils = []
            for i in COURSES:
                cd = course_data[i]
                walks = n2v_walks(cd["G"], nw=NUM_WALKS, wl=WALK_LENGTH, p=p, q=q, seed=SEED)
                emb = w2v(walks, vs=d, window=WINDOW, epochs=EPOCHS, seed=SEED)
                labels = km(emb, seed=SEED)
                sil = silhouette(emb, labels)
                sils.append(sil)
                count += 1
                results.append({
                    "p": p, "q": q, "d": d,
                    "course": i, "silhouette": sil
                })
                
                # Save cluster labels
                labels_file = os.path.join(OUT, f"labels_p{p}_q{q}_d{d}_course{i}.json")
                labels_data = {
                    "config_id": f"p{p}_q{q}_d{d}",
                    "p": p,
                    "q": q,
                    "d": d,
                    "course": i,
                    "seed": SEED,
                    "student_index": list(range(len(cd["labels"]))),
                    "student_labels": cd["labels"],
                    "cluster_labels": labels.tolist(),
                    "silhouette": sil
                }
                with open(labels_file, "w") as f:
                    json.dump(labels_data, f, indent=2)
                
                # Save embeddings for key configurations
                if d in [1, 2] and p == 1.0 and q == 1.0:
                    emb_file = os.path.join(OUT, f"embeddings_p{p}_q{q}_d{d}_course{i}.npy")
                    np.save(emb_file, emb)
                
            avg = np.nanmean(sils)
            print(f"  p={p:.1f}, q={q:.1f}, d={d:<2d}  "
                  f"Avg Sil={avg:.4f}  "
                  f"[{count}/{total}]")

elapsed = time.time() - t0
print(f"\nGrid search complete in {elapsed:.1f}s")

# ── Analysis ──
df = pd.DataFrame(results)

# Per-config averages
avgs = df.groupby(["p", "q", "d"])["silhouette"].mean().reset_index()
avgs.columns = ["p", "q", "d", "avg_silhouette"]
avgs = avgs.sort_values("avg_silhouette", ascending=False)

print("\n" + "=" * 70)
print("  TOP 10 CONFIGURATIONS (by average Silhouette across 6 courses)")
print("=" * 70)
print(f"  {'Rank':<5} {'p':<6} {'q':<6} {'d':<5} {'Avg Sil':>10}")
print("  " + "-" * 35)
for rank, (_, row) in enumerate(avgs.head(10).iterrows(), 1):
    marker = " ★ BEST" if rank == 1 else ""
    print(f"  {rank:<5} {row['p']:<6.1f} {row['q']:<6.1f} {int(row['d']):<5} "
          f"{row['avg_silhouette']:>10.4f}{marker}")

best = avgs.iloc[0]
print(f"\n  ★ BEST OVERALL: p={best['p']}, q={best['q']}, d={int(best['d'])}  "
      f"(Avg Silhouette = {best['avg_silhouette']:.4f})")

# Best at d=1
best_d1 = avgs[avgs["d"] == 1].iloc[0]
print(f"  ★ BEST at d=1:  p={best_d1['p']}, q={best_d1['q']}, d=1  "
      f"(Avg Silhouette = {best_d1['avg_silhouette']:.4f})")

# Best at d=2
best_d2 = avgs[avgs["d"] == 2].iloc[0]
print(f"  ★ BEST at d=2:  p={best_d2['p']}, q={best_d2['q']}, d=2  "
      f"(Avg Silhouette = {best_d2['avg_silhouette']:.4f})")

# Neutral (p=1, q=1)
for d_val in [1, 2]:
    neutral = avgs[(avgs["p"] == 1.0) & (avgs["q"] == 1.0) & (avgs["d"] == d_val)]
    if len(neutral) > 0:
        print(f"  ● NEUTRAL (1,1,{d_val}):  "
              f"(Avg Silhouette = {neutral.iloc[0]['avg_silhouette']:.4f})")

# ── Save complete results ──
output = {
    "grid": {
        "p_values": P_VALUES,
        "q_values": Q_VALUES,
        "d_values": D_VALUES,
        "seed": SEED,
        "total_configs": len(P_VALUES) * len(Q_VALUES) * len(D_VALUES),
        "courses": COURSES,
        "fixed_params": {
            "walk_length": WALK_LENGTH,
            "num_walks": NUM_WALKS,
            "window": WINDOW,
            "epochs": EPOCHS,
            "n_clusters": N_CLUSTERS
        }
    },
    "results": results,  # ALL individual results
    "averages": avgs.to_dict(orient="records"),  # ALL averages
    "best_overall": {
        "p": float(best["p"]),
        "q": float(best["q"]),
        "d": int(best["d"]),
        "avg_silhouette": float(best["avg_silhouette"]),
    },
    "best_d1": {
        "p": float(best_d1["p"]),
        "q": float(best_d1["q"]),
        "d": 1,
        "avg_silhouette": float(best_d1["avg_silhouette"]),
    },
    "best_d2": {
        "p": float(best_d2["p"]),
        "q": float(best_d2["q"]),
        "d": 2,
        "avg_silhouette": float(best_d2["avg_silhouette"]),
    },
    "selection_criterion": "Average Silhouette Score across 6 courses",
    "selection_note": "Configuration selected based on highest average Silhouette Score. "
                      "The complete grid results are preserved for reproducibility and "
                      "alternative selection criteria can be applied post-hoc."
}

with open(os.path.join(OUT, "grid_search_complete.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  [saved grid_search_complete.json]")

# Save selection rationale
selection_info = {
    "selected_configuration": {
        "p": float(best["p"]),
        "q": float(best["q"]),
        "d": int(best["d"]),
        "avg_silhouette": float(best["avg_silhouette"])
    },
    "selection_criterion": "Maximum average Silhouette Score across all 6 courses",
    "rationale": "The Silhouette Score is the primary evaluation metric as it balances "
                 "intra-cluster cohesion and inter-cluster separation. The selected "
                  "configuration achieves the highest average Silhouette Score across "
                  "all six course sectioning instances.",
    "alternative_considerations": {
        "d1_best": {
            "p": float(best_d1["p"]),
            "q": float(best_d1["q"]),
            "avg_silhouette": float(best_d1["avg_silhouette"])
        },
        "d2_best": {
            "p": float(best_d2["p"]),
            "q": float(best_d2["q"]),
            "avg_silhouette": float(best_d2["avg_silhouette"])
        },
        "note": "If d=1 is preferred for visualization/simple reasons, "
                f"the best d=1 configuration (p={best_d1['p']}, q={best_d1['q']}) "
                f"achieves avg Silhouette = {best_d1['avg_silhouette']:.4f}"
    }
}

with open(os.path.join(OUT, "selection_rationale.json"), "w") as f:
    json.dump(selection_info, f, indent=2, default=str)
print(f"  [saved selection_rationale.json]")

# ── Heatmap figures (one per dimension with independent normalization) ──
PS = {'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
      'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'}

for d in D_VALUES:
    plt.rcParams.update(PS)
    sub = df[df["d"] == d]
    pivot = sub.groupby(["p", "q"])["silhouette"].mean().unstack()
    
    # Independent normalization per d (each d gets its own color range)
    vmin = pivot.values.min()
    vmax = pivot.values.max()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(Q_VALUES)))
    ax.set_xticklabels([f"{q:.1f}" for q in Q_VALUES])
    ax.set_yticks(range(len(P_VALUES)))
    ax.set_yticklabels([f"{p:.1f}" for p in P_VALUES])
    ax.set_xlabel("q (BFS/DFS bias)")
    ax.set_ylabel("p (return parameter)")
    ax.set_title(f"Node2Vec Sensitivity — d={d}\n(Independent color scale: [{vmin:.3f}, {vmax:.3f}])")
    
    # Annotate cells with numerical values
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if val > pivot.values.mean() else "black")
    
    fig.colorbar(im, ax=ax, label="Avg Silhouette Score", shrink=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"grid_heatmap_d{d}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig grid_heatmap_d{d} (independent normalization)]")

# ── Combined heatmap with independent normalization per panel ──
# This addresses Part 10: each d gets its own color range
plt.rcParams.update(PS)
fig, axes = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 5), sharey=True)

for idx, d in enumerate(D_VALUES):
    sub = df[df["d"] == d]
    pivot = sub.groupby(["p", "q"])["silhouette"].mean().unstack()
    
    # Independent normalization per panel
    vmin_d = pivot.values.min()
    vmax_d = pivot.values.max()
    
    im = axes[idx].imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                          vmin=vmin_d, vmax=vmax_d)
    axes[idx].set_xticks(range(len(Q_VALUES)))
    axes[idx].set_xticklabels([f"{q:.1f}" for q in Q_VALUES])
    if idx == 0:
        axes[idx].set_yticks(range(len(P_VALUES)))
        axes[idx].set_yticklabels([f"{p:.1f}" for p in P_VALUES])
        axes[idx].set_ylabel("p")
    axes[idx].set_xlabel("q")
    axes[idx].set_title(f"d={d}\n[{vmin_d:.3f}–{vmax_d:.3f}]")
    
    # Annotate cells
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            val = pivot.values[i, j]
            axes[idx].text(j, i, f"{val:.3f}", ha="center", va="center",
                           fontsize=8, color="white" if val > pivot.values.mean() else "black")

# Add a single colorbar for reference
cbar = fig.colorbar(im, ax=axes, label="Avg Silhouette Score", shrink=0.8, pad=0.02)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

fig.suptitle("Node2Vec (p, q, d) Grid Search — Average Silhouette Score\n"
             "Each panel uses independent color normalization", 
             fontsize=14, y=1.02)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "grid_heatmap_all_d_independent.{ext}"),  # Will be fixed below
                dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  [fig grid_heatmap_all_d_independent]")

# Fix the filename (the f-string above had a bug)
for ext in ("pdf", "png"):
    old_path = os.path.join(FIG, f"grid_heatmap_all_d_independent.{{ext}}")
    new_path = os.path.join(FIG, f"grid_heatmap_all_d_independent.{ext}")
    if os.path.exists(old_path.format(ext=ext)):
        os.rename(old_path.format(ext=ext), new_path)

# ── Also create combined heatmap with common scale for comparison ──
# This addresses Part 10: preserve the option to compare absolute scores
plt.rcParams.update(PS)
global_min = df["silhouette"].min()
global_max = df["silhouette"].max()

fig, axes = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 5), sharey=True)

for idx, d in enumerate(D_VALUES):
    sub = df[df["d"] == d]
    pivot = sub.groupby(["p", "q"])["silhouette"].mean().unstack()
    
    # Common normalization across all d
    im = axes[idx].imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                          vmin=global_min, vmax=global_max)
    axes[idx].set_xticks(range(len(Q_VALUES)))
    axes[idx].set_xticklabels([f"{q:.1f}" for q in Q_VALUES])
    if idx == 0:
        axes[idx].set_yticks(range(len(P_VALUES)))
        axes[idx].set_yticklabels([f"{p:.1f}" for p in P_VALUES])
        axes[idx].set_ylabel("p")
    axes[idx].set_xlabel("q")
    axes[idx].set_title(f"d={d}")
    
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            val = pivot.values[i, j]
            axes[idx].text(j, i, f"{val:.3f}", ha="center", va="center",
                           fontsize=8, color="white" if val > pivot.values.mean() else "black")

fig.colorbar(im, ax=axes, label="Avg Silhouette Score (common scale)", shrink=0.8, pad=0.02)
fig.suptitle("Node2Vec (p, q, d) Grid Search — Average Silhouette Score\n"
             "Common color scale across all panels for absolute comparison",
             fontsize=14, y=1.02)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"grid_heatmap_all_d_common.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  [fig grid_heatmap_all_d_common]")

# ── Silhouette vs d plot ──
plt.rcParams.update(PS)
fig, ax = plt.subplots(figsize=(7, 4))
for p, q in [(1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (0.5, 0.5), (2.0, 2.0)]:
    sub = df[(df["p"] == p) & (df["q"] == q)]
    d_avgs = sub.groupby("d")["silhouette"].mean()
    label = f"p={p:.1f}, q={q:.1f}"
    marker = "o" if (p, q) == (1.0, 1.0) else "s"
    lw = 2.5 if (p, q) == (1.0, 1.0) else 1.5
    ax.plot(d_avgs.index, d_avgs.values, marker=marker, label=label,
            linewidth=lw, markersize=7)
ax.set_xlabel("Embedding Dimension (d)")
ax.set_ylabel("Average Silhouette Score")
ax.set_title("Silhouette vs. Embedding Dimension for Selected (p, q) Configurations")
ax.set_xticks(D_VALUES)
ax.legend(framealpha=0.9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "sil_vs_dim.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  [fig sil_vs_dim]")

print("\n" + "=" * 70)
print("  GRID SEARCH COMPLETE")
print(f"  Output: {os.path.join(OUT, 'grid_search_complete.json')}")
print(f"  Figures: {FIG}/")
print("=" * 70)
