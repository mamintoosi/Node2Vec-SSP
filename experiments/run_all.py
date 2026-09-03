#!/usr/bin/env python3
"""
Complete experiment reproduction with fixed seed=42.
Saves intermediate results after each section so partial runs are not lost.

Usage:
    cd /data/git/mamintoosi/Deepwalk-SSP
    /data/python-envs/pytorch/bin/python experiments/run_all.py 2>&1 | tee results/reproduced/run.log
"""
import os, sys, time, json, warnings, random
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
OUT = os.path.join(ROOT, "results", "reproduced")
FIG = os.path.join(OUT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from scipy.stats import wilcoxon as sp_wilcoxon

COURSES = [1, 2, 3, 4, 5, 6]
SEED = 42
N_SEEDS = 20

# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════
def ss(d, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(d, f, indent=2, default=str)
    print(f"  [saved {fn}]")

def read_class(fp):
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        for j in range(n):
            p = f.readline().split()
            bv = p[2]
            if len(bv) < m: bv = bv.ljust(m, '0')
            mat[j] = [int(c) for c in bv[:m]]
    return mat

def find_univ(m):
    return np.where(m.sum(axis=0) == m.shape[0])[0].tolist()

def mk_graph(m):
    G = nx.Graph()
    n = m.shape[0]
    for i in range(n): G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            w = int(np.sum(m[i]*m[j]))
            if w > 0: G.add_edge(i, j, weight=w)
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
                if not nbs: break
                if prev is None:
                    nxt = nbs[rng.randint(len(nbs))]
                else:
                    ps = set(G.neighbors(prev))
                    ws = []
                    for z in nbs:
                        a = 1.0/p if z==prev else (1.0 if z in ps else 1.0/q)
                        ws.append(a * G[cur][z].get('weight', 1))
                    t = sum(ws); pr = np.array(ws)/t
                    r = rng.random(); cum = 0.0; nxt = nbs[-1]
                    for idx, prob in enumerate(pr):
                        cum += prob
                        if r <= cum: nxt = nbs[idx]; break
                walk.append(nxt); prev, cur = cur, nxt
            walks.append(walk)
    return walks

def w2v(walks, vs=2, window=5, epochs=30, seed=0):
    m = Word2Vec(walks, vector_size=vs, window=window, hs=1, sg=1,
                 workers=1, seed=seed, min_count=1, sample=0)
    m.train(walks, total_examples=m.corpus_count, epochs=epochs, report_delay=0)
    return m.wv.vectors

def calc_m(data, labels):
    if len(np.unique(labels)) < 2:
        return {"sil": np.nan, "dbi": np.nan, "ch": np.nan}
    return {"sil": float(silhouette_score(data, labels)),
            "dbi": float(davies_bouldin_score(data, labels)),
            "ch": float(calinski_harabasz_score(data, labels))}

def km(data, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(data)

def spec(adj, seed=0):
    return SpectralClustering(n_clusters=2, affinity='precomputed',
                              random_state=seed, assign_labels='kmeans').fit_predict(adj)

def pca_km(data, seed=0):
    r = PCA(n_components=2, random_state=seed).fit_transform(data)
    l = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(r)
    return l, r

def run_n2v(mat, G, p=1.0, q=1.0, seed=0):
    t0 = time.perf_counter()
    w = n2v_walks(G, p=p, q=q, seed=seed)
    tw = time.perf_counter()-t0
    t0 = time.perf_counter()
    e = w2v(w, seed=seed)
    tw2 = time.perf_counter()-t0
    t0 = time.perf_counter()
    l = km(e, seed=seed)
    tk = time.perf_counter()-t0
    me = calc_m(e, l)
    return {"sil": me["sil"], "dbi": me["dbi"], "ch": me["ch"],
            "tw": tw, "tw2": tw2, "tk": tk, "tt": tw+tw2+tk}

# ══════════════════════════════════════════════════════════════════════
# Figure generation
# ══════════════════════════════════════════════════════════════════════
COLORS = {"bow":"#FF9800","pca":"#795548","spec":"#607D8B","n11":"#2196F3","n105":"#4CAF50"}
METHS = ["bow","pca","spec","n11","n105"]
MNAME = {"bow":"BoW +\nKMeans","pca":"PCA +\nKMeans","spec":"Spectral",
          "n11":"Node2Vec\n(p=1, q=1)","n105":"Node2Vec\n(p=1, q=0.5)"}

PS = {"figure.figsize":(7,5),"figure.dpi":300,"font.family":"serif",
      "font.size":11,"axes.titlesize":13,"axes.labelsize":12,
      "legend.fontsize":10,"savefig.dpi":300,"savefig.bbox":"tight"}

def sf(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), format='pdf')
    fig.savefig(os.path.join(FIG, f"{name}.png"), format='png')
    plt.close(fig)
    print(f"  [fig {name}]")

def fig_baseline(r):
    plt.rcParams.update(PS)
    cs = sorted(set(d["course"] for d in r))
    x = np.arange(len(cs)); w=0.15
    off = np.arange(5)-2
    fig,ax = plt.subplots(figsize=(8,4.5))
    for i,mk in enumerate(METHS):
        vals = [[d["sil"] for d in r if d["course"]==c and d["method"]==mk][0] for c in cs]
        bars = ax.bar(x+off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
                      edgecolor='white', linewidth=0.5)
        for b,v in zip(bars,vals):
            if v>0.3: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                              f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (higher is better)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.set_ylim(0,0.85); ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sf(fig, 'baseline_comparison')

def fig_all_metrics(r):
    plt.rcParams.update(PS)
    cs = sorted(set(d["course"] for d in r))
    x = np.arange(len(cs)); w=0.15; off=np.arange(5)-2
    fig,axes = plt.subplots(1,3,figsize=(18,5))
    for ax,(col,ylabel) in zip(axes,[("sil","Silhouette Score (↑)"),
                                      ("dbi","Davies-Bouldin Index (↓)"),
                                      ("ch","Calinski-Harabasz Index (↑)")]):
        for i,mk in enumerate(METHS):
            vals = [[d[col] for d in r if d["course"]==c and d["method"]==mk][0] for c in cs]
            ax.bar(x+off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
                   edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Course'); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels([f'C{i}' for i in cs], fontsize=8)
        ax.grid(axis='y',alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ax==axes[0]: ax.legend(loc='upper left',fontsize=7,framealpha=0.9)
    fig.tight_layout(); sf(fig, 'method_comparison_all_metrics')

    # Individual sil
    fig,ax = plt.subplots(figsize=(8,4.5))
    for i,mk in enumerate(METHS):
        vals = [[d["sil"] for d in r if d["course"]==c and d["method"]==mk][0] for c in cs]
        ax.bar(x+off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Course'); ax.set_ylabel('Silhouette Score (higher is better)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.set_ylim(0,0.85); ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sf(fig, 'silhouette_score_comparison_all_files')

    # DBI
    fig,ax = plt.subplots(figsize=(8,4.5))
    for i,mk in enumerate(METHS):
        vals = [[d["dbi"] for d in r if d["course"]==c and d["method"]==mk][0] for c in cs]
        ax.bar(x+off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Course'); ax.set_ylabel('Davies-Bouldin Index (lower is better)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sf(fig, 'DBI_comparison_all_files')

    # CHI
    fig,ax = plt.subplots(figsize=(8,4.5))
    for i,mk in enumerate(METHS):
        vals = [[d["ch"] for d in r if d["course"]==c and d["method"]==mk][0] for c in cs]
        ax.bar(x+off[i]*w, vals, w, label=MNAME[mk], color=COLORS[mk],
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Course'); ax.set_ylabel('Calinski-Harabasz Index (higher is better)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    sf(fig, 'CHI_comparison_all_files')

def fig_sil_vs_d(sens_d):
    plt.rcParams.update(PS)
    dims = sorted(set(d["dim"] for d in sens_d))
    means_n = [np.mean([d["sil"] for d in sens_d if d["dim"]==dm]) for dm in dims]
    means_b = [np.mean([d["sil_bow"] for d in sens_d if d["dim"]==dm]) for dm in dims]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(dims, means_n, 'o-', color=COLORS["n11"], lw=2, ms=8, label='Node2Vec (p=1, q=1)')
    ax.plot(dims, means_b, 's--', color=COLORS["bow"], lw=2, ms=8, label='BoW + KMeans')
    ax.set_xlabel('Embedding Dimension (d)')
    ax.set_ylabel('Silhouette Score (mean)')
    ax.set_title('Impact of Embedding Dimensionality on Clustering Quality')
    ax.set_xticks(dims); ax.legend(framealpha=0.9)
    ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    sf(fig, 'repro_silhouette_vs_d')

def fig_heatmap(sens):
    plt.rcParams.update(PS)
    df = pd.DataFrame(sens)
    pv = df.groupby(['p','q'])['sil'].mean().reset_index().pivot(index='p',columns='q',values='sil')
    pv = pv.sort_index(ascending=False)
    fig,ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(pv.values, cmap='YlOrRd', aspect='auto',
                   vmin=pv.values.min()-0.002, vmax=pv.values.max()+0.002)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([f'{c:.1f}' for c in pv.columns])
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels([f'{r:.1f}' for r in pv.index])
    ax.set_xlabel('q (in-out)'); ax.set_ylabel('p (return)')
    ax.set_title('Node2Vec Parameter Sensitivity')
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i,j]
            tc = 'white' if v > pv.values.mean() else 'black'
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=10,color=tc,fontweight='bold')
    fig.colorbar(im, label='Silhouette Score', shrink=0.8)
    fig.tight_layout(); sf(fig, 'sensitivity_heatmap')

def fig_stability(stab):
    plt.rcParams.update(PS)
    cs = sorted(stab.keys(), key=int)
    data = [[r["sil"] for r in stab[c]] for c in cs]
    fig,ax = plt.subplots(figsize=(10,6))
    bp = ax.boxplot(data, tick_labels=[f"Course {c}" for c in cs], patch_artist=True,
                    boxprops=dict(facecolor=COLORS["n11"],alpha=0.7),
                    medianprops=dict(color='black',lw=2))
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Node2Vec Stability Across 20 Random Seeds')
    ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    sf(fig, 'stability_boxplot')

def fig_graph_density(gs):
    plt.rcParams.update(PS)
    cs = [str(i) for i in range(1,7)]
    x = np.arange(6); w=0.35
    fig,ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, [gs["old"][c]["d"] for c in cs], w,
           label='Original', color='#E53935', alpha=0.8)
    ax.bar(x+w/2, [gs["new"][c]["d"] for c in cs], w,
           label='Revised (constant removed)', color='#43A047', alpha=0.8)
    for iv,(o,n) in enumerate(zip([gs["old"][c]["d"] for c in cs],
                                   [gs["new"][c]["d"] for c in cs])):
        ax.annotate(f'{o:.2f}',xy=(iv-w/2,o),xytext=(0,3),textcoords="offset points",
                    ha='center',va='bottom',fontsize=9)
        ax.annotate(f'{n:.2f}',xy=(iv+w/2,n),xytext=(0,3),textcoords="offset points",
                    ha='center',va='bottom',fontsize=9)
    ax.set_xlabel('Course'); ax.set_ylabel('Graph Density')
    ax.set_title('Effect of Removing Constant Courses on Graph Density')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in range(1,7)])
    ax.legend(framealpha=0.9); ax.set_ylim(0,1.15)
    ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    sf(fig, 'graph_density_comparison')

def fig_runtime(rt):
    plt.rcParams.update(PS)
    cs = sorted(set(d["course"] for d in rt))
    x = np.arange(len(cs)); ms = sorted(set(d["method"] for d in rt))
    nw = len(ms); w=0.15; clr = plt.cm.Set2(np.linspace(0,1,nw))
    fig,ax = plt.subplots(figsize=(9,5))
    for i,ml in enumerate(ms):
        vals = [[d["tt"] for d in rt if d["course"]==c and d["method"]==ml][0] for c in cs]
        ax.bar(x+(i-nw/2+0.5)*w, vals, w, label=ml, color=clr[i], edgecolor='white')
    ax.set_xlabel('Course'); ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in cs])
    ax.legend(fontsize=7,ncol=2); ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title('Pipeline Runtime Comparison'); fig.tight_layout()
    sf(fig, 'runtime_comparison')

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    random.seed(SEED); np.random.seed(SEED)
    T0 = time.time()
    print("="*70)
    print("  FULL REPRODUCTION (seed=42)")
    print("="*70)

    # ── Load data ──
    print("\n[1/8] Loading data...")
    cd = {}; gs_old, gs_new = {}, {}
    for i in COURSES:
        mat = read_class(os.path.join(DATA, f"{i}.txt"))
        uc = find_univ(mat)
        mf = np.delete(mat, uc, axis=1)
        G = mk_graph(mf)
        n = mf.shape[0]; mx = n*(n-1)//2; e = G.number_of_edges()
        cd[i] = {"m": mf, "G": G, "n": n}
        gs_old[str(i)] = {"n": n, "e": int(n*(n-1)/2), "d": 1.0, "c": 1}
        gs_new[str(i)] = {"n": n, "e": e, "d": e/mx if mx else 0,
                          "c": nx.number_connected_components(G)}
        print(f"  Course {i}: {n} students, {mf.shape[1]} courses, "
              f"{e} edges, density={e/mx:.4f}")
    ss({"old": gs_old, "new": gs_new}, "graph_stats.json")

    # ── Main comparison ──
    print("\n[2/8] Main comparison (5 methods × 6 courses)...")
    all_r = []; rt = []
    pq = {"n11":(1.0,1.0), "n105":(1.0,0.5)}
    labels_map = {"bow":"BoW+KMeans","pca":"PCA+KMeans","spec":"Spectral",
                  "n11":"Node2Vec(1,1)","n105":"Node2Vec(1,0.5)"}

    for i in COURSES:
        m, G = cd[i]["m"], cd[i]["G"]
        print(f"\n  Course {i}:")
        for mk in METHS:
            t0 = time.perf_counter()
            if mk == "bow":
                l = km(m.astype(float), seed=SEED)
                me = calc_m(m.astype(float), l)
            elif mk == "pca":
                l, red = pca_km(m.astype(float), seed=SEED)
                me = calc_m(red, l)
            elif mk == "spec":
                adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
                l = spec(adj, seed=SEED)
                me = calc_m(m.astype(float), l)
            else:
                p, q = pq[mk]
                r = run_n2v(m, G, p=p, q=q, seed=SEED)
                me = {"sil": r["sil"], "dbi": r["dbi"], "ch": r["ch"]}
            tt = time.perf_counter()-t0
            if mk in pq: tt = r["tt"]
            all_r.append({"course":i,"method":mk,"sil":me["sil"],"dbi":me["dbi"],"ch":me["ch"]})
            rt.append({"course":i,"method":labels_map[mk],"tt":tt})
            print(f"    {labels_map[mk]:<20} Sil={me['sil']:.3f} DBI={me['dbi']:.3f} "
                  f"CH={me['ch']:.1f} ({tt:.2f}s)")
    ss(all_r, "all_methods.json")
    ss(rt, "runtime.json")

    # Summary
    print("\n  SUMMARY TABLE:")
    print(f"  {'Method':<20}", end="")
    for i in COURSES: print(f" {'C'+str(i):>6}", end="")
    print(f" {'Avg':>8}")
    print("  "+"-"*60)
    for mk in METHS:
        print(f"  {labels_map[mk]:<20}", end="")
        vs = []
        for i in COURSES:
            v = [d["sil"] for d in all_r if d["course"]==i and d["method"]==mk][0]
            print(f" {v:>6.3f}", end=""); vs.append(v)
        print(f" {np.mean(vs):>8.3f}")

    # ── Sensitivity p,q ──
    print("\n[3/8] Node2Vec (p,q) sensitivity...")
    sens = []
    for p in [0.5,1.0,2.0]:
        for q in [0.5,1.0,2.0]:
            ss_list = []
            for i in COURSES:
                r = run_n2v(cd[i]["m"], cd[i]["G"], p=p, q=q, seed=SEED)
                ss_list.append(r["sil"])
                sens.append({"p":p,"q":q,"course":i,"sil":r["sil"],"dbi":r["dbi"],"ch":r["ch"]})
            print(f"  p={p}, q={q}: Sil={np.mean(ss_list):.3f}")
    ss(sens, "sensitivity_pq.json")

    # ── Hyperparameter sensitivity ──
    print("\n[4/8] Hyperparameter sensitivity...")
    # Dimension
    sens_d = []
    for d in [1,2,3,5,10]:
        ss_list = []
        for i in COURSES:
            w = n2v_walks(cd[i]["G"], seed=SEED)
            e = w2v(w, vs=d, seed=SEED)
            l = km(e, seed=SEED)
            me = calc_m(e, l)
            mb = calc_m(cd[i]["m"].astype(float), l)
            ss_list.append(me["sil"])
            sens_d.append({"dim":d,"course":i,"sil":me["sil"],"dbi":me["dbi"],
                           "ch":me["ch"],"sil_bow":mb["sil"]})
        print(f"  d={d}: Sil={np.mean(ss_list):.3f}")
    ss(sens_d, "sensitivity_dim.json")

    # Walk length
    sens_t = []
    for wl in [5,10,20,40,80]:
        ss_list = []
        for i in COURSES:
            w = n2v_walks(cd[i]["G"], wl=wl, seed=SEED)
            e = w2v(w, seed=SEED)
            me = calc_m(e, km(e, seed=SEED))
            ss_list.append(me["sil"])
            sens_t.append({"wl":wl,"course":i,"sil":me["sil"]})
        print(f"  t={wl}: Sil={np.mean(ss_list):.3f}")
    ss(sens_t, "sensitivity_wl.json")

    # Num walks
    sens_g = []
    for nw in [10,20,40,80,160]:
        ss_list = []
        for i in COURSES:
            w = n2v_walks(cd[i]["G"], nw=nw, seed=SEED)
            e = w2v(w, seed=SEED)
            me = calc_m(e, km(e, seed=SEED))
            ss_list.append(me["sil"])
            sens_g.append({"nw":nw,"course":i,"sil":me["sil"]})
        print(f"  gamma={nw}: Sil={np.mean(ss_list):.3f}")
    ss(sens_g, "sensitivity_nw.json")

    # Window
    sens_w = []
    for ws in [1,2,5,10,20]:
        ss_list = []
        for i in COURSES:
            w = n2v_walks(cd[i]["G"], seed=SEED)
            e = w2v(w, window=ws, seed=SEED)
            me = calc_m(e, km(e, seed=SEED))
            ss_list.append(me["sil"])
            sens_w.append({"ws":ws,"course":i,"sil":me["sil"]})
        print(f"  w={ws}: Sil={np.mean(ss_list):.3f}")
    ss(sens_w, "sensitivity_ws.json")

    # ── Stability (20 seeds) ──
    print("\n[5/8] Stability analysis (20 seeds × 6 courses)...")
    stab = {}
    for i in COURSES:
        sres = []
        for seed in range(N_SEEDS):
            w = n2v_walks(cd[i]["G"], seed=seed)
            e = w2v(w, seed=seed)
            l = km(e, seed=seed)
            me = calc_m(e, l)
            sres.append({"seed":seed,"sil":me["sil"],"dbi":me["dbi"],"ch":me["ch"]})
        sils = [r["sil"] for r in sres]
        stab[str(i)] = sres
        # Save after each course
        ss(stab, "stability.json")
        print(f"  Course {i}: Sil={np.mean(sils):.3f} ± {np.std(sils):.3f} "
              f"(min={np.min(sils):.3f}, max={np.max(sils):.3f})")

    # ARI stability
    print("\n  ARI stability (KMeans on fixed embeddings, 20 seeds):")
    ari = {}
    for i in COURSES:
        w = n2v_walks(cd[i]["G"], seed=SEED)
        e = w2v(w, seed=SEED)
        labs = [km(e, seed=s) for s in range(N_SEEDS)]
        ari_vals = [adjusted_rand_score(labs[a], labs[b])
                    for a in range(len(labs)) for b in range(a+1, len(labs))]
        ari[str(i)] = {"mean": float(np.mean(ari_vals)),
                       "std": float(np.std(ari_vals))}
        print(f"  Course {i}: ARI={np.mean(ari_vals):.3f} ± {np.std(ari_vals):.3f}")
    ss(ari, "ari_stability.json")

    # ── Statistical analysis ──
    print("\n[6/8] Statistical analysis...")
    n11 = np.array([[d["sil"] for d in all_r if d["course"]==i and d["method"]=="n11"][0]
                    for i in COURSES])
    n105 = np.array([[d["sil"] for d in all_r if d["course"]==i and d["method"]=="n105"][0]
                     for i in COURSES])
    bow = np.array([[d["sil"] for d in all_r if d["course"]==i and d["method"]=="bow"][0]
                    for i in COURSES])
    pca = np.array([[d["sil"] for d in all_r if d["course"]==i and d["method"]=="pca"][0]
                    for i in COURSES])
    sp = np.array([[d["sil"] for d in all_r if d["course"]==i and d["method"]=="spec"][0]
                   for i in COURSES])

    print(f"\n  {'Course':<10} {'N2V(1,1)':>10} {'N2V(1,0.5)':>10} {'BoW':>10} "
          f"{'PCA':>10} {'Spectral':>10}")
    print("  "+"-"*60)
    for ci,c in enumerate(COURSES):
        print(f"  Course {c:<4} {n11[ci]:>10.3f} {n105[ci]:>10.3f} "
              f"{bow[ci]:>10.3f} {pca[ci]:>10.3f} {sp[ci]:>10.3f}")
    print(f"  {'Average':<10} {np.mean(n11):>10.3f} {np.mean(n105):>10.3f} "
          f"{np.mean(bow):>10.3f} {np.mean(pca):>10.3f} {np.mean(sp):>10.3f}")

    # Comparisons
    comps = [
        ("Node2Vec(1,1) vs BoW", n11, bow),
        ("Node2Vec(1,1) vs PCA", n11, pca),
        ("Node2Vec(1,1) vs Spectral", n11, sp),
        ("Node2Vec(1,0.5) vs BoW", n105, bow),
        ("Node2Vec(1,0.5) vs PCA", n105, pca),
        ("Node2Vec(1,1) vs Node2Vec(1,0.5)", n11, n105),
    ]
    stat = []
    for nm, x, y in comps:
        try:
            st, pv = sp_wilcoxon(x, y, alternative='two-sided')
            st, pv = float(st), float(pv)
        except: st, pv = float('nan'), float('nan')
        n = len(x)
        mean_w = n*(n+1)/4; std_w = np.sqrt(n*(n+1)*(2*n+1)/24)
        z = (st-mean_w)/std_w if std_w>0 else 0
        r = min(abs(z)/np.sqrt(n), 1.0) if n>=5 else float('nan')
        dom = sum(1 for a in x for b in y if a>b) - sum(1 for a in x for b in y if a<b)
        delta = dom/(n*n)
        stat.append({"name":nm,"mx":float(np.mean(x)),"my":float(np.mean(y)),
                     "diff":float(np.mean(x-y)),"W":st,"p":pv,"r":r,"delta":delta,"n":n})
        print(f"\n  {nm}:")
        print(f"    Mean: {np.mean(x):.3f} vs {np.mean(y):.3f} (diff: {np.mean(x-y):+.3f})")
        print(f"    W={st}, p={pv:.4f}, r={r:.3f}, delta={delta:.3f}")
    ss(stat, "statistical_analysis.json")

    # ── Figures ──
    print("\n[7/8] Generating figures...")
    fig_baseline(all_r)
    fig_all_metrics(all_r)
    fig_sil_vs_d(sens_d)
    fig_heatmap(sens)
    fig_stability(stab)
    fig_graph_density({"old":gs_old,"new":gs_new})
    fig_runtime(rt)

    # ── Done ──
    tt = time.time()-T0
    print("\n[8/8] COMPLETE")
    print(f"  Total: {tt:.0f}s ({tt/60:.1f} min)")
    print(f"  Output: {OUT}")
    print(f"  Figures: {FIG}")
    print(f"\n  Key results:")
    print(f"  N2V(1,1):  {np.mean(n11):.3f}")
    print(f"  N2V(1,0.5): {np.mean(n105):.3f}")
    print(f"  BoW:  {np.mean(bow):.3f}")
    print(f"  PCA:  {np.mean(pca):.3f}")
    print(f"  Spec: {np.mean(sp):.3f}")
    pq_avgs = pd.DataFrame(sens).groupby(['p','q'])['sil'].mean()
    bpq = pq_avgs.idxmax()
    print(f"  Best p,q: p={bpq[0]}, q={bpq[1]} (Sil={pq_avgs.max():.3f})")

if __name__ == "__main__":
    main()
