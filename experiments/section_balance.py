#!/usr/bin/env python3
"""
Section Balance Analysis for all 5 methods.

Re-runs the main comparison (5 methods x 6 courses, seed=42) to capture
cluster labels and compute section sizes (|S1|, |S2|, Balance B) for
every method. This does NOT change any existing tables or reported values.

Outputs:
  results/reproduced/section_balance.json         - all data
  results/reproduced/section_balance_table.tex    - compact table (like tab:baseline_comparison)
  results/reproduced/section_balance_table_detail.tex - detailed section sizes for Node2Vec

Usage:
  cd /data/git/mamintoosi/Deepwalk-SSP
  /data/python-envs/pytorch/bin/python experiments/section_balance.py
"""
import os, sys, json, time, warnings, random
import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "reproduced")
os.makedirs(OUT, exist_ok=True)

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec

COURSES = [1, 2, 3, 4, 5, 6]
SEED = 42

METHS = ["bow", "pca", "spec", "n11", "n105"]
MNAME = {
    "bow": "BoW + KMeans",
    "pca": "PCA + KMeans",
    "spec": "Spectral",
    "n11": "Node2Vec (1,1)",
    "n105": "Node2Vec (1,0.5)",
}
pq = {"n11": (1.0, 1.0), "n105": (1.0, 0.5)}


def read_class(fp):
    with open(fp) as f:
        n, m = map(int, f.readline().split())
        mat = np.zeros((n, m), dtype=int)
        for j in range(n):
            p = f.readline().split()
            bv = p[2]
            if len(bv) < m:
                bv = bv.ljust(m, '0')
            mat[j] = [int(c) for c in bv[:m]]
    return mat


def find_univ(m):
    return np.where(m.sum(axis=0) == m.shape[0])[0].tolist()


def mk_graph(m):
    G = nx.Graph()
    n = m.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
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


def w2v_train(walks, vs=2, window=5, epochs=30, seed=0):
    m = Word2Vec(walks, vector_size=vs, window=window, hs=1, sg=1,
                 workers=1, seed=seed, min_count=1, sample=0)
    m.train(walks, total_examples=m.corpus_count, epochs=epochs, report_delay=0)
    return m.wv.vectors


def km(data, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(data)


def spec_cluster(adj, seed=0):
    return SpectralClustering(n_clusters=2, affinity='precomputed',
                              random_state=seed, assign_labels='kmeans').fit_predict(adj)


def sil_score(data, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(data, labels))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    print("=" * 70)
    print("  SECTION BALANCE ANALYSIS (seed=42)")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    cd = {}
    for i in COURSES:
        mat = read_class(os.path.join(DATA, f"{i}.txt"))
        uc = find_univ(mat)
        mf = np.delete(mat, uc, axis=1)
        G = mk_graph(mf)
        cd[i] = {"m": mf, "G": G, "n": mf.shape[0]}
        print(f"  Course {i}: {mf.shape[0]} students, {mf.shape[1]} courses, "
              f"{G.number_of_edges()} edges")

    # Run all methods, capture labels
    print("\n[2/3] Running 5 methods x 6 courses (seed=42)...")
    results = []
    for i in COURSES:
        m, G = cd[i]["m"], cd[i]["G"]
        print(f"\n  Course {i} ({cd[i]['n']} students):")
        for mk in METHS:
            if mk == "bow":
                labels = km(m.astype(float), seed=SEED)
                emb = m.astype(float)
            elif mk == "pca":
                red = PCA(n_components=2, random_state=SEED).fit_transform(m.astype(float))
                labels = km(red, seed=SEED)
                emb = red
            elif mk == "spec":
                adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
                labels = spec_cluster(adj, seed=SEED)
                emb = m.astype(float)
            elif mk in pq:
                p_val, q_val = pq[mk]
                walks = n2v_walks(G, p=p_val, q=q_val, seed=SEED)
                emb = w2v_train(walks, seed=SEED)
                labels = km(emb, seed=SEED)

            counts = np.bincount(labels)
            s1, s2 = int(max(counts)), int(min(counts))
            balance = s2 / s1 if s1 > 0 else 0.0
            sc = sil_score(emb, labels)

            results.append({
                "course": i, "method": mk, "method_name": MNAME[mk],
                "s1": s1, "s2": s2, "balance": round(balance, 4),
                "silhouette": round(sc, 4), "n_students": cd[i]["n"],
            })
            print(f"    {MNAME[mk]:<22}  |S1|={s1:3d}  |S2|={s2:3d}  "
                  f"B={balance:.3f}  Sil={sc:.4f}")

    # Save JSON
    out_json = os.path.join(OUT, "section_balance.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [saved {out_json}]")

    # ── Generate LaTeX tables ──
    print("\n[3/3] Generating LaTeX tables...")

    # TABLE A: Compact balance table (mirrors tab:baseline_comparison format)
    lines = []
    lines.append("% === AUTO-GENERATED by experiments/section_balance.py ===")
    lines.append("% Paste into sn-article.tex replacing the old tab:section_balance")
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Section balance analysis ($B = \min(|S_1|,|S_2|) / \max(|S_1|,|S_2|)$) "
                 r"across five methods for student sectioning with $k{=}2$ sections at $d{=}2$. "
                 r"$B{=}1$ indicates perfectly balanced sections. "
                 r"Bold values indicate the highest balance per course.}")
    lines.append(r"\label{tab:section_balance}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Course} & \textbf{BoW + KMeans} & \textbf{PCA + KMeans} "
                 r"& \textbf{Spectral} & \multicolumn{2}{c}{\textbf{Node2Vec ($d{=}2$)}} \\")
    lines.append(r"\cmidrule(lr){5-6}")
    lines.append(r" & & & & $p{=}1$, $q{=}1$ & $p{=}0.5$, $q{=}0.5$ \\")
    lines.append(r"\midrule")

    for course in COURSES:
        cr = {r["method"]: r for r in results if r["course"] == course}
        best_bal = max(cr[mk]["balance"] for mk in METHS)
        parts = [f"{course}"]
        for mk in METHS:
            b = cr[mk]["balance"]
            if b == best_bal:
                parts.append(rf"\textbf{{{b:.3f}}}")
            else:
                parts.append(f"{b:.3f}")
        lines.append(" & ".join(parts) + r" \\")

    # Average row
    avgs = {mk: np.mean([r["balance"] for r in results if r["method"] == mk])
            for mk in METHS}
    best_avg_m = max(avgs, key=avgs.get)
    avg_parts = [r"\textbf{Avg.}"]
    for mk in METHS:
        if mk == best_avg_m:
            avg_parts.append(rf"\textbf{{{avgs[mk]:.3f}}}")
        else:
            avg_parts.append(f"{avgs[mk]:.3f}")
    lines.append(" & ".join(avg_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("% === END AUTO-GENERATED TABLE ===")

    out_tex = os.path.join(OUT, "section_balance_table.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved {out_tex}]")

    # TABLE B: Detailed section sizes for Node2Vec only (supplementary)
    lines2 = []
    lines2.append(r"\begin{table}[t]")
    lines2.append(r"\centering")
    lines2.append(r"\caption{Section sizes ($|S_1|$, $|S_2|$) and balance ($B$) "
                  r"for Node2Vec ($p{=}1$, $q{=}1$) with KMeans clustering ($k{=}2$).}")
    lines2.append(r"\label{tab:section_balance_detail}")
    lines2.append(r"\begin{tabular}{lcccr}")
    lines2.append(r"\toprule")
    lines2.append(r"\textbf{Course} & $|S_1|$ & $|S_2|$ & \textbf{Balance} ($B$) "
                  r"& \textbf{Silhouette} \\")
    lines2.append(r"\midrule")

    n11 = [r for r in results if r["method"] == "n11"]
    for r in n11:
        lines2.append(f"{r['course']} & {r['s1']} & {r['s2']} & {r['balance']:.3f} "
                      f"& {r['silhouette']:.3f} " + r"\\")

    avg_s1 = np.mean([r["s1"] for r in n11])
    avg_b = np.mean([r["balance"] for r in n11])
    avg_s = np.mean([r["silhouette"] for r in n11])
    lines2.append(rf"\textbf{{Avg.}} & {avg_s1:.0f} & {cd[1]['n']-avg_s1:.0f} "
                  rf"& {avg_b:.3f} & {avg_s:.3f} " + r"\\")
    lines2.append(r"\bottomrule")
    lines2.append(r"\end{tabular}")
    lines2.append(r"\end{table}")

    out_tex2 = os.path.join(OUT, "section_balance_table_detail.tex")
    with open(out_tex2, "w") as f:
        f.write("\n".join(lines2))
    print(f"  [saved {out_tex2}]")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Section Balance (B) per method")
    print("=" * 70)
    header = f"  {'Method':<22}"
    for c in COURSES:
        header += f"  C{c}"
    header += "  Avg"
    print(header)
    print("  " + "-" * 68)
    for mk in METHS:
        row = f"  {MNAME[mk]:<22}"
        for c in COURSES:
            b = [r["balance"] for r in results if r["course"] == c and r["method"] == mk][0]
            row += f" {b:.3f}"
        row += f"  {avgs[mk]:.3f}"
        print(row)

    tt = time.time() - t0
    print(f"\n  Total: {tt:.0f}s")
    print(f"  JSON:  {out_json}")
    print(f"  LaTeX: {out_tex}")
    print(f"  LaTeX: {out_tex2}")
    print("\n  DONE!")


if __name__ == "__main__":
    main()
