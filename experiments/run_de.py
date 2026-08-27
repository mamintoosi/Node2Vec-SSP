# Sections D+E: Sensitivity + Validation (no heavy stability)
import os, sys, json, time, warnings
import numpy as np
import networkx as nx
warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from experiments.config import FILE_INDICES, DATA_DIR, RESULTS_DIR
from experiments.core import read_class, create_graph_from_bow, generate_random_walks, generate_node2vec_walks, run_node2vec_pipeline
from experiments.evaluation import evaluate_clustering
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
LINUX_DIR = os.path.join(RESULTS_DIR, "linux_reexperiment")

# D: Sensitivity (3x3 grid)
print("=" * 70)
print("  SECTION D: Node2Vec Sensitivity (3x3 grid)")
print("=" * 70)
p_values = [0.5, 1.0, 2.0]
q_values = [0.5, 1.0, 2.0]
sensitivity_results = []
for file_idx in FILE_INDICES:
    filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
    scm, _ = read_class(filepath)
    print(f"\n  Course {file_idx} ({scm.shape[0]} students):")
    for p_val in p_values:
        for q_val in q_values:
            t0 = time.perf_counter()
            n2v_result = run_node2vec_pipeline(filepath, p=p_val, q=q_val, seed=0)
            elapsed = time.perf_counter() - t0
            if n2v_result["success"]:
                n2v_eval = evaluate_clustering(n2v_result["embeddings"], n_clusters=2, random_state=0)
                sil = n2v_eval["kmeans"]["metrics"]["silhouette"]
                dbi = n2v_eval["kmeans"]["metrics"]["davies_bouldin"]
                ch = n2v_eval["kmeans"]["metrics"]["calinski_harabasz"]
            else:
                sil = dbi = ch = np.nan
            sensitivity_results.append({"course": file_idx, "p": p_val, "q": q_val,
                "silhouette": float(sil), "davies_bouldin": float(dbi),
                "calinski_harabasz": float(ch), "runtime_s": float(elapsed)})
    print(f"    {'p\\q':<8}", end="")
    for q_val in q_values: print(f"  q={q_val:<6}", end="")
    print()
    for p_val in p_values:
        print(f"    p={p_val:<4}", end="")
        for q_val in q_values:
            m = [r for r in sensitivity_results if r["course"]==file_idx and r["p"]==p_val and r["q"]==q_val]
            if m: print(f"  {m[0]['silhouette']:<8.4f}", end="")
        print()

print("\n  Average Silhouette:")
print(f"    {'p\\q':<8}", end="")
for q_val in q_values: print(f"  q={q_val:<6}", end="")
print()
for p_val in p_values:
    print(f"    p={p_val:<4}", end="")
    for q_val in q_values:
        m = [r for r in sensitivity_results if r["p"]==p_val and r["q"]==q_val]
        if m: print(f"  {np.mean([r['silhouette'] for r in m]):<8.4f}", end="")
    print()
with open(os.path.join(LINUX_DIR, "node2vec_sensitivity.json"), "w") as f:
    json.dump(sensitivity_results, f, indent=2, cls=NumpyEncoder)
print("  Saved: node2vec_sensitivity.json")

# E: Validation
print("\n" + "=" * 70)
print("  SECTION E: Node2Vec Walk Validation")
print("=" * 70)
validation = {"walk_identity_rates": {}, "return_rates": {}}
for file_idx in FILE_INDICES:
    filepath = os.path.join(DATA_DIR, f"{file_idx}.txt")
    scm, _ = read_class(filepath)
    G = create_graph_from_bow(scm)
    dw_w = generate_random_walks(G, num_walks_per_node=50, walk_length=50, seed=42)
    n2v11_w = generate_node2vec_walks(G, num_walks_per_node=50, walk_length=50, p=1.0, q=1.0, seed=42)
    n2v05_w = generate_node2vec_walks(G, num_walks_per_node=50, walk_length=50, p=0.5, q=1.0, seed=42)
    
    def walk_id(a,b):
        n=min(len(a),len(b)); return sum(1 for i in range(n) if a[i]==b[i])/n
    def ret_rate(w):
        r=0; t=0
        for walk in w:
            for i in range(2,len(walk)):
                t+=1
                if walk[i]==walk[i-2]: r+=1
        return r/t if t>0 else 0

    validation["walk_identity_rates"][str(file_idx)] = {
        "dw_vs_n2v11": float(walk_id(dw_w, n2v11_w)),
        "dw_vs_n2v05": float(walk_id(dw_w, n2v05_w)),
    }
    validation["return_rates"][str(file_idx)] = {
        "deepwalk": float(ret_rate(dw_w)),
        "n2v_1_1": float(ret_rate(n2v11_w)),
        "n2v_0_5": float(ret_rate(n2v05_w)),
    }
    print(f"  Course {file_idx}: DW-vs-N2V(1,1)={walk_id(dw_w,n2v11_w):.4f}, "
          f"DW-vs-N2V(0.5,1)={walk_id(dw_w,n2v05_w):.4f}, "
          f"DW return={ret_rate(dw_w):.4f}, N2V(0.5,1) return={ret_rate(n2v05_w):.4f}")

with open(os.path.join(LINUX_DIR, "node2vec_validation.json"), "w") as f:
    json.dump(validation, f, indent=2, cls=NumpyEncoder)
print("  Saved: node2vec_validation.json")
print("\n  DONE: Sections D+E")
