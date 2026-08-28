# -*- coding: utf-8 -*-
"""Stage D2: Stability analysis — seeds 10-14."""
import os, sys, time, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

from experiments.shared import (OUT_DIR, FILE_INDICES, load_course_data,
    generate_unbiased_walks, generate_node2vec_walks, train_word2vec,
    cluster_kmeans, compute_metrics)


def main():
    t0 = time.time()
    print("  STAGE D2: Stability (seeds 10-14)")

    course_data = load_course_data()
    partial_path = os.path.join(OUT_DIR, "stability_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for seed in range(10, 15):
        if f"{seed}_1" in all_results:
            print(f"  Seed {seed}: already done")
            continue
        for i in FILE_INDICES:
            cd = course_data[i]
            m, G = cd["matrix"], cd["graph"]
            dw_walks = generate_unbiased_walks(G, seed=seed)
            dw_emb = train_word2vec(dw_walks, seed=seed)
            dw_labels = cluster_kmeans(dw_emb, seed=seed)
            dw_m = compute_metrics(dw_emb, dw_labels)
            n2v_walks = generate_node2vec_walks(G, p=1.0, q=1.0, seed=seed)
            n2v_emb = train_word2vec(n2v_walks, seed=seed)
            n2v_labels = cluster_kmeans(n2v_emb, seed=seed)
            n2v_m = compute_metrics(n2v_emb, n2v_labels)
            all_results[f"{seed}_{i}"] = {
                "seed": seed,
                "n2v_silhouette": n2v_m["silhouette"], "n2v_dbi": n2v_m["dbi"], "n2v_ch": n2v_m["ch"],
                "dw_silhouette": dw_m["silhouette"], "dw_dbi": dw_m["dbi"], "dw_ch": dw_m["ch"],
            }
        print(f"  Seed {seed} done ({time.time()-t0:.0f}s)")

    with open(partial_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Stage D2 done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
