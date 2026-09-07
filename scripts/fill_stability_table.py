import json, os, sys
import numpy as np

# Reuse the exact pipeline functions so numbers are protocol-consistent.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import final_d1_d2_experiments as F

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

OUT = os.path.join(ROOT, "results", "final_d1_primary")
OUT_BASE = os.path.join(ROOT, "results")
COURSES = [1, 2, 3, 4, 5, 6]
N_SEEDS = 20

def ari_stats(labs):
    vals = [adjusted_rand_score(labs[a], labs[b])
            for a in range(len(labs)) for b in range(a + 1, len(labs))]
    return float(np.mean(vals)), float(np.std(vals))

def main():
    # Load data (identical to run_experiment in the module)
    print("Loading data ...")
    cd = {}
    for i in COURSES:
        mat, student_labels = F.read_class(os.path.join(F.DATA, f"{i}.txt"))
        uc = F.find_univ(mat)
        mf = np.delete(mat, uc, axis=1)
        cd[i] = mf

    res = {"kmeans": {}, "gmm": {}, "agg": {}, "bow": {}}
    print("\nCourse | KMeans         | GMM            | Agglomerative  | BoW KMeans")
    print("-" * 75)
    for i in COURSES:
        emb = np.load(os.path.join(OUT, f"embeddings_course{i}_n11.npy"))
        m = cd[i]

        # Node2Vec embeddings (seed-42, primary p=2, q=1, d=1): 20 KMeans seeds
        labs_km = [F.km(emb, seed=s) for s in range(N_SEEDS)]
        km_mean, km_std = ari_stats(labs_km)

        # GMM and Agglomerative on the same fixed embeddings.
        # ARI is computed over 20 restarts where the algorithm has a seed
        # (GMM); Agglomerative is deterministic, so its ARI is 1.000 by
        # construction and is reported as such.
        gmm_labs = [GaussianMixture(n_components=2, random_state=s).fit_predict(emb)
                    for s in range(N_SEEDS)]
        gmm_mean, gmm_std = ari_stats(gmm_labs)

        agg_lab = AgglomerativeClustering_wrap(emb)
        res["agg"][i] = 1.0  # deterministic algorithm

        # BoW + KMeans on the raw (university-column-removed) matrix
        bow_labs = [F.km(m.astype(float), seed=s) for s in range(N_SEEDS)]
        bow_mean, bow_std = ari_stats(bow_labs)

        res["kmeans"][i] = {"mean": km_mean, "std": km_std}
        res["gmm"][i] = {"mean": gmm_mean, "std": gmm_std}
        res["bow"][i] = {"mean": bow_mean, "std": bow_std}

        print(f"{i:>6} | {km_mean:.3f} +/- {km_std:.3f} | {gmm_mean:.3f} +/- {gmm_std:.3f} | "
              f"1.000         | {bow_mean:.3f} +/- {bow_std:.3f}")

    def avg(key):
        return float(np.mean([res[key][i]["mean"] for i in COURSES]))

    print("-" * 75)
    print(f"  Avg  | {avg('kmeans'):.3f}          | {avg('gmm'):.3f}          | 1.000          | {avg('bow'):.3f}")

    # LaTeX rows
    print("\nLaTeX rows:")
    for c in COURSES:
        print(f" {c} & {res['kmeans'][c]['mean']:.3f} & {res['gmm'][c]['mean']:.3f} & "
              f"1.000 & {res['bow'][c]['mean']:.3f} \\\\")
    print(f" \\textbf{{Avg.}} & {avg('kmeans'):.3f} & {avg('gmm'):.3f} & 1.000 & {avg('bow'):.3f} \\\\")

    out_json = os.path.join(OUT_BASE, "final_d1_primary", "ari_stability_all_clusterers.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved {out_json}]")

def AgglomerativeClustering_wrap(emb):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=2).fit_predict(emb)

if __name__ == "__main__":
    main()
