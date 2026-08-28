# -*- coding: utf-8 -*-
"""Stage B: Node2Vec parameter sensitivity (p,q grid)."""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from experiments.shared import (OUT_DIR, FIG_DIR, FILE_INDICES,
    load_course_data, generate_node2vec_walks, train_word2vec,
    cluster_kmeans, compute_metrics)

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  STAGE B: Node2Vec Parameter Sensitivity")
    print("=" * 70)

    course_data = load_course_data()
    p_vals, q_vals = [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]
    sens_results = []

    for p in p_vals:
        for q in q_vals:
            sils, dbis, chs = [], [], []
            for i in FILE_INDICES:
                cd = course_data[i]
                walks = generate_node2vec_walks(cd["graph"], p=p, q=q, seed=0)
                emb = train_word2vec(walks, seed=0)
                labels = cluster_kmeans(emb, seed=0)
                m = compute_metrics(emb, labels)
                sils.append(m["silhouette"])
                dbis.append(m["dbi"])
                chs.append(m["ch"])
                sens_results.append({"p": p, "q": q, "course": i,
                    "silhouette": m["silhouette"], "dbi": m["dbi"], "ch": m["ch"]})
            print(f"  p={p:.1f}, q={q:.1f}: Sil={np.mean(sils):.3f}, DBI={np.mean(dbis):.3f}, CH={np.mean(chs):.1f}")

    df = pd.DataFrame(sens_results)
    df.to_json(os.path.join(OUT_DIR, "sensitivity.json"), orient="records", indent=2)
    df.to_excel(os.path.join(OUT_DIR, "sensitivity.xlsx"), index=False)

    # Heatmaps
    plt.rcParams.update(PUB_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (met, title, cmap) in enumerate([
        ("silhouette", "Silhouette Score (↑)", "YlGn"),
        ("dbi", "Davies-Bouldin Index (↓)", "YlGn_r"),
        ("ch", "Calinski-Harabasz Index (↑)", "YlGn")]):
        pv = df.groupby(["p", "q"])[met].mean().reset_index().pivot(index="p", columns="q", values=met)
        sns.heatmap(pv, annot=True, fmt=".3f" if met != "ch" else ".1f",
                    cmap=cmap, ax=axes[idx], cbar_kws={"shrink": 0.8})
        axes[idx].set_title(title); axes[idx].set_xlabel("q (in-out)"); axes[idx].set_ylabel("p (return)")
    fig.suptitle("Node2Vec Parameter Sensitivity (averaged over 6 courses)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: sensitivity_heatmap.pdf")

    # Summary
    pq = df.groupby(["p", "q"])["silhouette"].mean()
    best = pq.idxmax()
    print(f"\n  Best: p={best[0]}, q={best[1]}, Sil={pq.max():.3f}")
    print(f"  Range: {pq.min():.3f} to {pq.max():.3f} (Δ={pq.max()-pq.min():.3f})")
    print(f"  Stage B complete. Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
