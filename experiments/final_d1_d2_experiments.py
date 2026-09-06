#!/usr/bin/env python3
"""
Final d=1 and d=2 experiments for Node2Vec student sectioning.
Runs the main comparison with both d=1 and d=2, saving all required outputs.

Usage:
    cd /path/to/Deepwalk-SSP
    python experiments/final_d1_d2_experiments.py 2>&1 | tee results/final_d1_d2.log
"""

import os
import sys
import time
import json
import warnings
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Setup ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
OUT_BASE = os.path.join(ROOT, "results")

# Create output directories
OUT_D1 = os.path.join(OUT_BASE, "final_d1")
OUT_D2 = os.path.join(OUT_BASE, "final_d2")
FIG_D1 = os.path.join(OUT_D1, "figures")
FIG_D2 = os.path.join(OUT_D2, "figures")
os.makedirs(OUT_D1, exist_ok=True)
os.makedirs(OUT_D2, exist_ok=True)
os.makedirs(FIG_D1, exist_ok=True)
os.makedirs(FIG_D2, exist_ok=True)

from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from scipy.stats import wilcoxon as sp_wilcoxon

COURSES = [1, 2, 3, 4, 5, 6]
SEED = 42
N_SEEDS = 20

# Node2Vec parameters (same for d=1 and d=2, only d differs)
P = 1.0
Q = 1.0
WALK_LENGTH = 10
NUM_WALKS = 80
WINDOW = 5
EPOCHS = 30
N_CLUSTERS = 2

def ss(d, fn, out_dir):
    """Save JSON to specified output directory"""
    with open(os.path.join(out_dir, fn), "w") as f:
        json.dump(d, f, indent=2, default=str)
    print(f"  [saved {out_dir}/{fn}]")

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
            w = int(np.sum(m[i]*m[j]))
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
            for _ in range(wl-1):
                nbs = list(G.neighbors(cur))
                if not nbs:
                    break
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
    return m.wv.vectors, m.wv.index_to_key

def km(data, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(data)

def calc_m(data, labels):
    if len(np.unique(labels)) < 2:
        return {"sil": np.nan, "dbi": np.nan, "ch": np.nan}
    return {
        "sil": float(silhouette_score(data, labels)),
        "dbi": float(davies_bouldin_score(data, labels)),
        "ch": float(calinski_harabasz_score(data, labels))
    }

def run_experiment(vs, out_dir, fig_dir):
    """
    Run complete experiment for a given embedding dimension.
    
    Args:
        vs: vector_size (embedding dimension) - 1 or 2
        out_dir: output directory for JSON results
        fig_dir: output directory for figures
        
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*70}")
    print(f"  RUNNING EXPERIMENTS WITH d={vs}")
    print(f"{'='*70}")
    
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Load data
    print("\n[1/6] Loading and preprocessing data...")
    cd = {}
    for i in COURSES:
        mat, student_labels = read_class(os.path.join(DATA, f"{i}.txt"))
        uc = find_univ(mat)
        mf = np.delete(mat, uc, axis=1)
        G = mk_graph(mf)
        cd[i] = {"m": mf, "G": G, "n": mf.shape[0], "labels": student_labels}
        print(f"  Course {i}: {mf.shape[0]} students, {mf.shape[1]} courses, {G.number_of_edges()} edges")
    
    results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown",
            "python_version": sys.version,
            "seed": SEED,
            "embedding_dimension": vs,
            "node2vec_params": {
                "p": P,
                "q": Q,
                "walk_length": WALK_LENGTH,
                "num_walks": NUM_WALKS,
                "window": WINDOW,
                "epochs": EPOCHS,
                "n_clusters": N_CLUSTERS
            },
            "output_directory": out_dir
        },
        "all_methods": [],
        "runtime": [],
        "sensitivity_pq": [],
        "stability": {},
        "ari_stability": {},
        "course_data": {}
    }
    
    # Main comparison - 5 methods
    print("\n[2/6] Main comparison (5 methods × 6 courses)...")
    METHS = ["bow", "pca", "spec", "n11", "n105"]
    LABELS_MAP = {
        "bow": "BoW+KMeans",
        "pca": "PCA+KMeans",
        "spec": "Spectral",
        "n11": f"Node2Vec(p={P},q={Q})",
        "n105": f"Node2Vec(p={P},q={Q-0.5})"
    }
    PQ = {"n11": (P, Q), "n105": (P, Q-0.5)}
    
    for i in COURSES:
        m, G = cd[i]["m"], cd[i]["G"]
        labels_list = cd[i]["labels"]
        print(f"\n  Course {i}:")
        
        for mk in METHS:
            t0 = time.perf_counter()
            
            if mk == "bow":
                l = km(m.astype(float), seed=SEED)
                me = calc_m(m.astype(float), l)
                emb = None
            elif mk == "pca":
                pca = PCA(n_components=2, random_state=SEED)
                red = pca.fit_transform(m.astype(float))
                l = km(red, seed=SEED)
                me = calc_m(red, l)
                emb = red
            elif mk == "spec":
                adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
                l = km(adj, seed=SEED)
                me = calc_m(m.astype(float), l)
                emb = None
            else:
                p, q = PQ[mk]
                t_walk = time.perf_counter()
                w = n2v_walks(G, nw=NUM_WALKS, wl=WALK_LENGTH, p=p, q=q, seed=SEED)
                t_walk_end = time.perf_counter()
                t_emb = time.perf_counter()
                emb, idx = w2v(w, vs=vs, window=WINDOW, epochs=EPOCHS, seed=SEED)
                t_emb_end = time.perf_counter()
                t_clust = time.perf_counter()
                l = km(emb, seed=SEED)
                t_clust_end = time.perf_counter()
                
                me = calc_m(emb, l)
                tt = (t_walk_end - t_walk) + (t_emb_end - t_emb) + (t_clust_end - t_clust)
                
                # SAVE EMBEDDINGS
                emb_file = os.path.join(out_dir, f"embeddings_course{i}_{mk}.npy")
                np.save(emb_file, emb)
                print(f"    [saved embeddings: {emb_file}]")
                
                # SAVE CLUSTER LABELS
                labels_file = os.path.join(out_dir, f"labels_course{i}_{mk}.json")
                labels_data = {
                    "course": i,
                    "method": mk,
                    "d": vs,
                    "p": p,
                    "q": q,
                    "seed": SEED,
                    "student_index": list(range(len(labels_list))),
                    "student_labels": labels_list,
                    "cluster_labels": l.tolist()
                }
                with open(labels_file, "w") as f:
                    json.dump(labels_data, f, indent=2)
                print(f"    [saved cluster labels: {labels_file}]")
            
            tt = time.perf_counter() - t0
            results["all_methods"].append({
                "course": i,
                "method": mk,
                "sil": me["sil"],
                "dbi": me["dbi"],
                "ch": me["ch"]
            })
            results["runtime"].append({
                "course": i,
                "method": LABELS_MAP[mk],
                "time_seconds": tt
            })
            print(f"    {LABELS_MAP[mk]:<25} Sil={me['sil']:.3f} DBI={me['dbi']:.3f} CH={me['ch']:.1f} ({tt:.2f}s)")
    
    ss(results["all_methods"], "all_methods.json", out_dir)
    ss(results["runtime"], "runtime.json", out_dir)
    
    # Summary table
    print("\n  SUMMARY TABLE:")
    print(f"  {'Method':<25}", end="")
    for i in COURSES:
        print(f" {'C'+str(i):>6}", end="")
    print(f" {'Avg':>8}")
    print("  " + "-"*70)
    for mk in METHS:
        print(f"  {LABELS_MAP[mk]:<25}", end="")
        vs_list = []
        for i in COURSES:
            v = [d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]==mk][0]
            print(f" {v:>6.3f}", end="")
            vs_list.append(v)
        print(f" {np.mean(vs_list):>8.3f}")
    
    # Sensitivity p,q
    print("\n[3/6] Node2Vec (p,q) sensitivity...")
    sens = []
    for p in [0.5, 1.0, 2.0]:
        for q in [0.5, 1.0, 2.0]:
            ss_list = []
            for i in COURSES:
                r = run_n2v(cd[i]["m"], cd[i]["G"], p=p, q=q, seed=SEED, vs=vs)
                ss_list.append(r["sil"])
                sens.append({
                    "p": p, "q": q, "course": i,
                    "sil": r["sil"], "dbi": r["dbi"], "ch": r["ch"],
                    "d": vs
                })
            print(f"  p={p}, q={q}: Sil={np.mean(ss_list):.3f}")
    results["sensitivity_pq"] = sens
    ss(results["sensitivity_pq"], "sensitivity_pq.json", out_dir)
    
    # Stability (20 seeds)
    print("\n[4/6] Stability analysis (20 seeds × 6 courses)...")
    stab = {}
    for i in COURSES:
        sres = []
        for seed in range(N_SEEDS):
            w = n2v_walks(cd[i]["G"], nw=NUM_WALKS, wl=WALK_LENGTH, p=P, q=Q, seed=seed)
            emb, _ = w2v(w, vs=vs, window=WINDOW, epochs=EPOCHS, seed=seed)
            l = km(emb, seed=seed)
            me = calc_m(emb, l)
            sres.append({"seed": seed, "sil": me["sil"], "dbi": me["dbi"], "ch": me["ch"]})
            
            # Save labels for each seed
            labels_file = os.path.join(out_dir, f"labels_course{i}_n11_seed{seed}.json")
            with open(labels_file, "w") as f:
                json.dump({
                    "course": i,
                    "method": "n11",
                    "d": vs,
                    "p": P,
                    "q": Q,
                    "seed": seed,
                    "student_index": list(range(len(cd[i]["labels"]))),
                    "student_labels": cd[i]["labels"],
                    "cluster_labels": l.tolist()
                }, f, indent=2)
        sils = [r["sil"] for r in sres]
        stab[str(i)] = sres
        print(f"  Course {i}: Sil={np.mean(sils):.3f} ± {np.std(sils):.3f} (min={np.min(sils):.3f}, max={np.max(sils):.3f})")
    results["stability"] = stab
    ss(results["stability"], "stability.json", out_dir)
    
    # ARI stability
    print("\n  ARI stability (KMeans on fixed embeddings, 20 seeds):")
    ari = {}
    for i in COURSES:
        w = n2v_walks(cd[i]["G"], nw=NUM_WALKS, wl=WALK_LENGTH, p=P, q=Q, seed=SEED)
        emb, _ = w2v(w, vs=vs, window=WINDOW, epochs=EPOCHS, seed=SEED)
        labs = [km(emb, seed=s) for s in range(N_SEEDS)]
        ari_vals = [adjusted_rand_score(labs[a], labs[b])
                    for a in range(len(labs)) for b in range(a+1, len(labs))]
        ari[str(i)] = {"mean": float(np.mean(ari_vals)), "std": float(np.std(ari_vals))}
        print(f"  Course {i}: ARI={np.mean(ari_vals):.3f} ± {np.std(ari_vals):.3f}")
    results["ari_stability"] = ari
    ss(results["ari_stability"], "ari_stability.json", out_dir)
    
    # Statistical analysis
    print("\n[5/6] Statistical analysis...")
    n11 = np.array([[d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]=="n11"][0]
                    for i in COURSES])
    n105 = np.array([[d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]=="n105"][0]
                     for i in COURSES])
    bow = np.array([[d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]=="bow"][0]
                    for i in COURSES])
    pca = np.array([[d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]=="pca"][0]
                    for i in COURSES])
    spec = np.array([[d["sil"] for d in results["all_methods"] if d["course"]==i and d["method"]=="spec"][0]
                     for i in COURSES])
    
    print(f"\n  {'Course':<10} {'N2V(p,q)':>10} {'N2V(p,q-0.5)':>12} {'BoW':>10} {'PCA':>10} {'Spectral':>10}")
    print("  " + "-"*60)
    for ci, c in enumerate(COURSES):
        print(f"  Course {c:<4} {n11[ci]:>10.3f} {n105[ci]:>12.3f} {bow[ci]:>10.3f} {pca[ci]:>10.3f} {spec[ci]:>10.3f}")
    print(f"  {'Average':<10} {np.mean(n11):>10.3f} {np.mean(n105):>12.3f} {np.mean(bow):>10.3f} {np.mean(pca):>10.3f} {np.mean(spec):>10.3f}")
    
    comps = [
        (f"Node2Vec(p={P},q={Q}) vs BoW", n11, bow),
        (f"Node2Vec(p={P},q={Q}) vs PCA", n11, pca),
        (f"Node2Vec(p={P},q={Q}) vs Spectral", n11, spec),
        (f"Node2Vec(p={P},q={Q-0.5}) vs BoW", n105, bow),
        (f"Node2Vec(p={P},q={Q-0.5}) vs PCA", n105, pca),
        (f"Node2Vec(p={P},q={Q}) vs Node2Vec(p={P},q={Q-0.5})", n11, n105),
    ]
    stat = []
    for nm, x, y in comps:
        try:
            st, pv = sp_wilcoxon(x, y, alternative='two-sided')
            st, pv = float(st), float(pv)
        except:
            st, pv = float('nan'), float('nan')
        n = len(x)
        dom = sum(1 for a in x for b in y if a>b) - sum(1 for a in x for b in y if a<b)
        delta = dom/(n*n)
        stat.append({
            "name": nm,
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
            "diff": float(np.mean(x-y)),
            "W": st, "p": pv,
            "delta": delta, "n": n
        })
        print(f"\n  {nm}:")
        print(f"    Mean: {np.mean(x):.3f} vs {np.mean(y):.3f} (diff: {np.mean(x-y):+.3f})")
        print(f"    W={st}, p={pv:.4f}, delta={delta:.3f}")
    results["statistical_analysis"] = stat
    ss(results["statistical_analysis"], "statistical_analysis.json", out_dir)
    
    # Generate figures
    print("\n[6/6] Generating figures...")
    generate_figures(results, fig_dir, vs)
    
    return results


def run_n2v(mat, G, p=1.0, q=1.0, seed=0, vs=2):
    """Run Node2Vec pipeline and return metrics"""
    w = n2v_walks(G, nw=NUM_WALKS, wl=WALK_LENGTH, p=p, q=q, seed=seed)
    emb, _ = w2v(w, vs=vs, window=WINDOW, epochs=EPOCHS, seed=seed)
    l = km(emb, seed=seed)
    me = calc_m(emb, l)
    return {"sil": me["sil"], "dbi": me["dbi"], "ch": me["ch"]}


def generate_figures(results, fig_dir, vs):
    """Generate all figures for the experiment"""
    PS = {
        "figure.figsize": (7, 5),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    }
    
    COLORS = {
        "bow": "#FF9800",
        "pca": "#795548",
        "spec": "#607D8B",
        "n11": "#2196F3",
        "n105": "#4CAF50"
    }
    
    METHS = ["bow", "pca", "spec", "n11", "n105"]
    MNAME = {
        "bow": "BoW +\nKMeans",
        "pca": "PCA +\nKMeans",
        "spec": "Spectral",
        "n11": f"Node2Vec\n(p={P}, q={Q})",
        "n105": f"Node2Vec\n(p={P}, q={Q-0.5})"
    }
    
    def sf(fig, name):
        fig.savefig(os.path.join(fig_dir, f"{name}.pdf"), format='pdf')
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), format='png')
        plt.close(fig)
        print(f"  [fig {name}]")
    
    # Baseline comparison
    plt.rcParams.update(PS)
    cs = sorted(set(d["course"] for d in results["all_methods"]))
    x = np.arange(len(cs))
    w = 0.15
    off = np.arange(5) - 2
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mk in enumerate(METHS):
        vals = [[d["sil"] for d in results["all_methods"] if d["course"]==c and d["method"]==mk][0] for c in cs]
        bars = ax.bar(x + off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
                      edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, vals):
            if v > 0.3:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sf(fig, 'baseline_comparison')
    
    # All metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, ylabel) in zip(axes, [
        ("sil", "Silhouette Score (↑)"),
        ("dbi", "Davies-Bouldin Index (↓)"),
        ("ch", "Calinski-Harabasz Index (↑)")
    ]):
        for i, mk in enumerate(METHS):
            vals = [[d[col] for d in results["all_methods"] if d["course"]==c and d["method"]==mk][0] for c in cs]
            ax.bar(x + off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
                   edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Course')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in cs], fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    fig.tight_layout()
    sf(fig, 'method_comparison_all_metrics')
    
    # Individual metric figures
    for metric, ylabel, fname in [
        ("sil", "Silhouette Score (higher is better)", "silhouette_score_comparison"),
        ("dbi", "Davies-Bouldin Index (lower is better)", "DBI_comparison"),
        ("ch", "Calinski-Harabasz Index (higher is better)", "CHI_comparison")
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, mk in enumerate(METHS):
            vals = [[d[metric] for d in results["all_methods"] if d["course"]==c and d["method"]==mk][0] for c in cs]
            ax.bar(x + off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
                   edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Course')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Course {i}' for i in cs])
        ax.legend(loc='upper left', ncol=2, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        sf(fig, fname)
    
    # Stability boxplot
    plt.rcParams.update(PS)
    cs = sorted(results["stability"].keys(), key=int)
    data = [[r["sil"] for r in results["stability"][c]] for c in cs]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=[f"Course {c}" for c in cs], patch_artist=True,
                    boxprops=dict(facecolor=COLORS["n11"], alpha=0.7),
                    medianprops=dict(color='black', lw=2))
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'Node2Vec Stability Across 20 Random Seeds (d={vs})')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    sf(fig, 'stability_boxplot')
    
    # Heatmap for p,q sensitivity
    plt.rcParams.update(PS)
    df = pd.DataFrame(results["sensitivity_pq"])
    pv = df.groupby(['p', 'q'])['sil'].mean().reset_index().pivot(index='p', columns='q', values='sil')
    pv = pv.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pv.values, cmap='YlOrRd', aspect='auto',
                   vmin=pv.values.min() - 0.002, vmax=pv.values.max() + 0.002)
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f'{c:.1f}' for c in pv.columns])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f'{r:.1f}' for r in pv.index])
    ax.set_xlabel('q (in-out)')
    ax.set_ylabel('p (return)')
    ax.set_title(f'Node2Vec Parameter Sensitivity (d={vs})')
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            tc = 'white' if v > pv.values.mean() else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10, color=tc, fontweight='bold')
    fig.colorbar(im, label='Silhouette Score', shrink=0.8)
    fig.tight_layout()
    sf(fig, 'sensitivity_heatmap')
    
    print(f"\n  Figures saved to: {fig_dir}")


def main():
    """Run both d=1 and d=2 experiments"""
    print("="*70)
    print("  FINAL d=1 AND d=2 EXPERIMENTS")
    print("  Node2Vec for Student Sectioning")
    print("="*70)
    print(f"\nRepository: {ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Seed: {SEED}")
    print(f"CPU cores: 2 (taskset 0,1)")
    print(f"\nNode2Vec parameters:")
    print(f"  p = {P}")
    print(f"  q = {Q}")
    print(f"  walk_length = {WALK_LENGTH}")
    print(f"  num_walks = {NUM_WALKS}")
    print(f"  window = {WINDOW}")
    print(f"  epochs = {EPOCHS}")
    print(f"  n_clusters = {N_CLUSTERS}")
    print(f"\nOutput directories:")
    print(f"  d=1: {OUT_D1}")
    print(f"  d=2: {OUT_D2}")
    
    start_time = time.time()
    
    # Run d=1 experiments
    results_d1 = run_experiment(1, OUT_D1, FIG_D1)
    
    # Run d=2 experiments
    results_d2 = run_experiment(2, OUT_D2, FIG_D2)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n  Output directories:")
    print(f"    d=1: {OUT_D1}")
    print(f"    d=2: {OUT_D2}")
    print(f"\n  Files saved:")
    print(f"    - all_methods.json (Silhouette, DBI, CH for all methods)")
    print(f"    - runtime.json (execution times)")
    print(f"    - sensitivity_pq.json (p,q sensitivity)")
    print(f"    - stability.json (20-seed stability)")
    print(f"    - ari_stability.json (ARI stability)")
    print(f"    - statistical_analysis.json (Wilcoxon tests)")
    print(f"    - embeddings_course*.npy (Node2Vec embeddings)")
    print(f"    - labels_course*.json (cluster assignments)")
    print(f"    - figures/*.pdf, *.png (publication figures)")
    print(f"\n  Key results:")
    print(f"  d=1:")
    d1_methods = results_d1["all_methods"]
    for mk in ["n11", "n105"]:
        vals = [d["sil"] for d in d1_methods if d["method"]==mk]
        print(f"    Node2Vec (p={P},q={Q}): {np.mean(vals):.4f}")
    print(f"  d=2:")
    d2_methods = results_d2["all_methods"]
    for mk in ["n11", "n105"]:
        vals = [d["sil"] for d in d2_methods if d["method"]==mk]
        print(f"    Node2Vec (p={P},q={Q}): {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
