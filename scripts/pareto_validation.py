#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20-seed validation for the Pareto candidate (p=1.0, q=0.5, d=2).

Reuses the pipeline functions of experiments/final_d1_d2_experiments.py
without modifying that file (its module-level P/Q are NOT used here).

Why this config: the existing 20-seed evidence covers
  - (2.0, 1.0, 1)  primary            -> results/final_d1_primary/
  - (1.0, 1.0, 2)  neutral/DeepWalk   -> results/final_d2/  (n11 artifacts)
  - (1.0, 0.5, 2)  compromise         -> ONLY seed 42 (final_d2 n105 labels)
so this script fills the one genuine gap, using the same seed protocol as the
existing (2,1,1) stability run so all three candidates are directly comparable.

Writes ONLY into results/final_pareto_validation/ (nothing existing is touched):
  - per-seed label JSONs  labels_course{i}_p1.0_q0.5_d2_seed{s}.json
    (course, method, d, p, q, seed, student_index, student_labels,
     cluster_labels, section_sizes, section_balance)
  - stability.json        per course, seeds 0..19: sil/dbi/ch
  - per_seed_metrics.json per course, seeds 0..19: sil/dbi/ch/balance/sizes
  - aggregate.json        per course mean/std/min/max for every metric + seed list
  - ari_stability.json    all-pairs ARI over the 20 seed-clusterings
  - embeddings_course{i}_p1.0_q0.5_d2.npy   seed-42 embeddings (ARI protocol)
  - run_metadata.json     config, seeds, library versions, start/end times
"""

import os
import sys
import json
import time
import random
import platform

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import final_d1_d2_experiments as F   # module constants P/Q are NOT used here

from sklearn.metrics import adjusted_rand_score

# ---------------------------------------------------------------------------
# Configuration (single source of truth for everything saved below)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "final_pareto_validation")
P, Q, D = 1.0, 0.5, 2            # the ONE missing Pareto candidate
METHOD_TAG = "p1.0_q0.5_d2"      # appears in every output filename
N_SEEDS = F.N_SEEDS              # 20  (same protocol as results/final_d1_primary)
SEED_MAIN = F.SEED               # 42  (canonical seed for ARI embeddings)

STAB_JSON = os.path.join(OUT_DIR, "stability.json")
AGG_JSON = os.path.join(OUT_DIR, "aggregate.json")


def balance_of(cluster_labels):
    """B = min(|S1|,|S2|) / max(|S1|,|S2|), computed from the saved labels."""
    c0 = sum(1 for x in cluster_labels if x == 0)
    c1 = len(cluster_labels) - c0
    if c0 == 0 or c1 == 0:
        return None, [c0, c1]
    return min(c0, c1) / max(c0, c1), [c0, c1]


def main():
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    RESUME = os.path.exists(STAB_JSON)

    print("=" * 60)
    print(f"20-seed validation @ p={P}, q={Q}, d={D}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Start time : {t_start}")
    if RESUME:
        print("RESUME     : stability.json exists -> heavy step will be SKIPPED")
    print("=" * 60)

    # ---- data & graphs (identical to run_experiment in the module) ----
    print("[1/4] Loading data & building graphs ...")
    random.seed(SEED_MAIN)
    np.random.seed(SEED_MAIN)
    cd = {}
    for i in F.COURSES:
        mat, student_labels = F.read_class(os.path.join(F.DATA, f"{i}.txt"))
        uc = F.find_univ(mat)
        mf = np.delete(mat, uc, axis=1)
        G = F.mk_graph(mf)
        cd[i] = {"m": mf, "G": G, "n": mf.shape[0], "labels": student_labels}
        print(f"  Course {i}: {mf.shape[0]} students, "
              f"{mf.shape[1]} courses, {G.number_of_edges()} edges")

    # ---- 20-seed stability: walks + Word2Vec + KMeans re-drawn per seed ----
    # (identical protocol to results/final_d1_primary/stability.json)
    if RESUME:
        print(f"[2/4] SKIPPED (resume): reusing existing {STAB_JSON}")
        with open(STAB_JSON) as f:
            stab = json.load(f)
    else:
        print(f"[2/4] 20-seed stability at p={P}, q={Q}, d={D} "
              f"(seeds 0..{N_SEEDS - 1}) ...")
        stab = {}
        for i in F.COURSES:
            sres = []
            for seed in range(N_SEEDS):
                w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH,
                                p=P, q=Q, seed=seed)
                emb, _ = F.w2v(w, vs=D, window=F.WINDOW, epochs=F.EPOCHS, seed=seed)
                l = F.km(emb, seed=seed)
                me = F.calc_m(emb, l)
                B, sizes = balance_of(l.tolist())
                sres.append({"seed": seed, "sil": me["sil"], "dbi": me["dbi"],
                             "ch": me["ch"], "balance": B, "sizes": sizes})

                labels_file = os.path.join(
                    OUT_DIR, f"labels_course{i}_{METHOD_TAG}_seed{seed}.json")
                with open(labels_file, "w") as f:
                    json.dump({
                        "course": i, "method": METHOD_TAG, "d": D,
                        "p": P, "q": Q, "seed": seed,
                        "student_index": list(range(len(cd[i]["labels"]))),
                        "student_labels": cd[i]["labels"],
                        "cluster_labels": l.tolist(),
                        "section_sizes": sizes,
                        "section_balance": B,
                    }, f, indent=2)
            sils = [r["sil"] for r in sres]
            stab[str(i)] = sres
            print(f"  Course {i}: Sil={np.nanmean(sils):.3f} "
                  f"+/- {np.nanstd(sils):.3f}")

        F.ss(stab, "stability.json", OUT_DIR)

    # ---- aggregates across seeds ----
    print("[3/4] Aggregating across seeds ...")
    agg = {}
    for i in F.COURSES:
        rows = stab[str(i)]
        agg[str(i)] = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
        for key in ("sil", "dbi", "ch", "balance"):
            vals = np.array([r[key] for r in rows], dtype=float)
            agg[str(i)][key] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
    F.ss(agg, "aggregate.json", OUT_DIR)
    F.ss({str(i): [{"seed": r["seed"], "sil": r["sil"], "dbi": r["dbi"],
                    "ch": r["ch"], "balance": r["balance"],
                    "sizes": r["sizes"]} for r in stab[str(i)]]
          for i in F.COURSES},
         "per_seed_metrics.json", OUT_DIR)

    # ---- ARI stability: fixed seed-42 embeddings, KMeans over 20 seeds ----
    # (identical protocol to results/final_d1_primary/ari_stability.json)
    print("[4/4] ARI stability (KMeans on seed-42 embeddings, 20 seeds) ...")
    ari = {}
    for i in F.COURSES:
        w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH,
                        p=P, q=Q, seed=SEED_MAIN)
        emb, idx = F.w2v(w, vs=D, window=F.WINDOW, epochs=F.EPOCHS, seed=SEED_MAIN)
        np.save(os.path.join(OUT_DIR, f"embeddings_course{i}_{METHOD_TAG}.npy"), emb)
        labs = [F.km(emb, seed=s) for s in range(N_SEEDS)]
        ari_vals = [adjusted_rand_score(labs[a], labs[b])
                    for a in range(len(labs)) for b in range(a + 1, len(labs))]
        ari[str(i)] = {"mean": float(np.mean(ari_vals)),
                       "std": float(np.std(ari_vals))}
        print(f"  Course {i}: ARI={np.mean(ari_vals):.3f} +/- {np.std(ari_vals):.3f}")
    F.ss(ari, "ari_stability.json", OUT_DIR)

    meta = {
        "purpose": "20-seed validation of Pareto candidate (p=1.0, q=0.5, d=2)",
        "p": P, "q": Q, "d": D,
        "seeds": list(range(N_SEEDS)),
        "seed_main": SEED_MAIN,
        "walk_length": F.WALK_LENGTH, "num_walks": F.NUM_WALKS,
        "window": F.WINDOW, "epochs": F.EPOCHS, "n_clusters": F.N_CLUSTERS,
        "protocol": "identical to results/final_d1_primary (20-seed stability; "
                    "ARI over 20 KMeans initializations on fixed seed-42 embeddings)",
        "balance_definition": "B = min(|S1|,|S2|)/max(|S1|,|S2|) from cluster labels",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "started": t_start,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resumed_from_existing_stability": RESUME,
    }
    F.ss(meta, "run_metadata.json", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
