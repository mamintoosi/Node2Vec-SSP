# -*- coding: utf-8 -*-
"""20-seed stability for the PRIMARY config (p=2.0, q=1.0, d=1).

Reuses the pipeline functions of experiments/final_d1_d2_experiments.py
without modifying that file. Writes only into results/final_d1_primary/.
Protocol matches the existing stability analysis exactly.
"""
import os, sys, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))
import final_d1_d2_experiments as F   # module constants P/Q are NOT used here

from sklearn.metrics import adjusted_rand_score

OUT_DIR = HERE
N_SEEDS = F.N_SEEDS          # 20
SEED_MAIN = F.SEED           # 42
P, Q = 2.0, 1.0              # PRIMARY configuration (only change vs the module)

# ---- resume support: skip the 20-seed step when stability.json already exists ----
STAB_JSON = os.path.join(OUT_DIR, "stability.json")
RESUME = os.path.exists(STAB_JSON)

# ---- data & graphs (identical to run_experiment in the module) ----
print(f"[1/4] Loading data & building graphs ...")
random.seed(SEED_MAIN)
np.random.seed(SEED_MAIN)
cd = {}
for i in F.COURSES:
    mat, student_labels = F.read_class(os.path.join(F.DATA, f"{i}.txt"))
    uc = F.find_univ(mat)
    mf = np.delete(mat, uc, axis=1)
    G = F.mk_graph(mf)
    cd[i] = {"m": mf, "G": G, "n": mf.shape[0], "labels": student_labels}
    print(f"  Course {i}: {mf.shape[0]} students, {mf.shape[1]} courses, {G.number_of_edges()} edges")

# ---- stability: everything re-drawn per seed (matches existing protocol) ----
if RESUME:
    print(f"[2/4] SKIPPED (resume): found existing {STAB_JSON} -> reusing it;")
    print("      only steps 3-4 (ARI stability, cheap) will run.")
    with open(STAB_JSON) as f:
        stab = json.load(f)
else:
    print(f"[2/4] 20-seed stability at p={P}, q={Q}, d=1 (seeds 0..{N_SEEDS-1}) ...")
    stab = {}
    for i in F.COURSES:
        sres = []
        for seed in range(N_SEEDS):
            w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH, p=P, q=Q, seed=seed)
            emb, _ = F.w2v(w, vs=1, window=F.WINDOW, epochs=F.EPOCHS, seed=seed)
            l = F.km(emb, seed=seed)
            me = F.calc_m(emb, l)
            sres.append({"seed": seed, "sil": me["sil"], "dbi": me["dbi"], "ch": me["ch"]})

            labels_file = os.path.join(OUT_DIR, f"labels_course{i}_n11_seed{seed}.json")
            with open(labels_file, "w") as f:
                json.dump({
                    "course": i, "method": "n11", "d": 1, "p": P, "q": Q, "seed": seed,
                    "student_index": list(range(len(cd[i]["labels"]))),
                    "student_labels": cd[i]["labels"],
                    "cluster_labels": l.tolist()
                }, f, indent=2)
        sils = [r["sil"] for r in sres]
        stab[str(i)] = sres
        print(f"  Course {i}: Sil={np.mean(sils):.3f} +/- {np.std(sils):.3f} "
              f"(min={np.min(sils):.3f}, max={np.max(sils):.3f})")

    F.ss(stab, "stability.json", OUT_DIR)

# ---- ARI stability: fixed seed-42 embeddings, KMeans over 20 seeds, all pairs ----
# (No label files are written here; canonical seed-42 labels for the primary
# config already exist as results/grid_search/labels_p2.0_q1.0_d1_course*.json.)
print("[3/4] ARI stability (KMeans on seed-42 embeddings, 20 seeds) ...")
ari = {}
for i in F.COURSES:
    w = F.n2v_walks(cd[i]["G"], nw=F.NUM_WALKS, wl=F.WALK_LENGTH, p=P, q=Q, seed=SEED_MAIN)
    emb, idx = F.w2v(w, vs=1, window=F.WINDOW, epochs=F.EPOCHS, seed=SEED_MAIN)
    np.save(os.path.join(OUT_DIR, f"embeddings_course{i}_n11.npy"), emb)
    labs = [F.km(emb, seed=s) for s in range(N_SEEDS)]
    ari_vals = [adjusted_rand_score(labs[a], labs[b])
                for a in range(len(labs)) for b in range(a + 1, len(labs))]
    ari[str(i)] = {"mean": float(np.mean(ari_vals)), "std": float(np.std(ari_vals))}
    print(f"  Course {i}: ARI={np.mean(ari_vals):.3f} +/- {np.std(ari_vals):.3f}")

print("[4/4] Saving summaries ...")
F.ss(ari, "ari_stability.json", OUT_DIR)
print("Done.")
