#!/usr/bin/env python3
"""Compute all exact values needed for the manuscript update from JSON data files."""
import json
import os

results_dir = "results/reproduced"

# Load all JSON files
with open(os.path.join(results_dir, "grid_search_pqd.json")) as f:
    grid = json.load(f)

with open(os.path.join(results_dir, "all_methods.json")) as f:
    all_methods = json.load(f)

with open(os.path.join(results_dir, "sensitivity_dim.json")) as f:
    sens_dim = json.load(f)

with open(os.path.join(results_dir, "sensitivity_wl.json")) as f:
    sens_wl = json.load(f)

with open(os.path.join(results_dir, "sensitivity_nw.json")) as f:
    sens_nw = json.load(f)

with open(os.path.join(results_dir, "sensitivity_ws.json")) as f:
    sens_ws = json.load(f)

with open(os.path.join(results_dir, "statistical_analysis.json")) as f:
    stats = json.load(f)

with open(os.path.join(results_dir, "stability.json")) as f:
    stability = json.load(f)

with open(os.path.join(results_dir, "ari_stability.json")) as f:
    ari = json.load(f)

with open(os.path.join(results_dir, "graph_stats.json")) as f:
    gstats = json.load(f)

with open(os.path.join(results_dir, "runtime.json")) as f:
    runtime = json.load(f)

print("=" * 80)
print("STEP 1: MAIN RESULTS TABLE (Table 1) — d=2")
print("=" * 80)

# Baselines from all_methods.json
courses = [1, 2, 3, 4, 5, 6]
baseline_methods = {"bow": "BoW+KMeans", "pca": "PCA+KMeans", "spec": "Spectral"}

for method_key, method_name in baseline_methods.items():
    vals = []
    for c in courses:
        entry = next(m for m in all_methods if m["course"] == c and m["method"] == method_key)
        vals.append(entry["sil"])
    avg = sum(vals) / len(vals)
    print(f"{method_name}: {[round(v, 3) for v in vals]}, avg={avg:.3f}")

# Node2Vec (p=1, q=1, d=2) — neutral
n11_d2 = []
for c in courses:
    entry = next(r for r in grid["results"] if r["p"] == 1.0 and r["q"] == 1.0 and r["d"] == 2 and r["course"] == c)
    n11_d2.append(entry["silhouette"])
avg_n11_d2 = sum(n11_d2) / len(n11_d2)
print(f"Node2Vec (p=1,q=1,d=2): {[round(v, 3) for v in n11_d2]}, avg={avg_n11_d2:.3f}")

# Node2Vec (p=0.5, q=0.5, d=2) — best d=2
n0505_d2 = []
for c in courses:
    entry = next(r for r in grid["results"] if r["p"] == 0.5 and r["q"] == 0.5 and r["d"] == 2 and r["course"] == c)
    n0505_d2.append(entry["silhouette"])
avg_n0505_d2 = sum(n0505_d2) / len(n0505_d2)
print(f"Node2Vec (p=0.5,q=0.5,d=2): {[round(v, 3) for v in n0505_d2]}, avg={avg_n0505_d2:.3f}")

# Winners per row
print("\nWinners per row:")
all_row = {
    "BoW": [0.090, 0.231, 0.154, 0.128, 0.137, 0.202],
    "PCA": [0.376, 0.603, 0.425, 0.428, 0.437, 0.524],
    "Spectral": [0.083, 0.205, 0.135, 0.128, 0.074, 0.143],
    "N2V(1,1)": n11_d2,
    "N2V(0.5,0.5)": n0505_d2,
}
for i, c in enumerate(courses):
    row = {k: v[i] for k, v in all_row.items()}
    winner = max(row, key=row.get)
    print(f"  Course {c}: {winner} ({row[winner]:.3f}) — all: { {k: round(v, 3) for k, v in row.items()} }")

print("\n" + "=" * 80)
print("STEP 2: EMBEDDING DIMENSION SENSITIVITY (Table 4)")
print("=" * 80)
print("Using p=1, q=1 (neutral) from sensitivity_dim.json:")
for dim in [1, 2, 3, 5, 10]:
    dim_entries = [e for e in sens_dim if e["dim"] == dim]
    sil_avg = sum(e["sil"] for e in dim_entries) / len(dim_entries)
    dbi_avg = sum(e["dbi"] for e in dim_entries) / len(dim_entries)
    ch_avg = sum(e["ch"] for e in dim_entries) / len(dim_entries)
    print(f"  d={dim}: Sil={sil_avg:.3f}, DBI={dbi_avg:.3f}, CH={ch_avg:.1f}")

print("\nPer-course for d=1 (p=1,q=1):")
for e in sens_dim:
    if e["dim"] == 1:
        print(f"  Course {e['course']}: sil={e['sil']:.3f}, dbi={e['dbi']:.3f}, ch={e['ch']:.1f}")

print("\n" + "=" * 80)
print("STEP 3: PARAMETER SENSITIVITY (Table 3) — d=2")
print("=" * 80)
for p in [0.5, 1.0, 2.0]:
    row = []
    for q in [0.5, 1.0, 2.0]:
        # Get from grid averages
        avg_entry = next(a for a in grid["averages"] if a["p"] == p and a["q"] == q and a["d"] == 2)
        row.append(avg_entry["avg_silhouette"])
    print(f"  p={p}: q=0.5={row[0]:.3f}, q=1.0={row[1]:.3f}, q=2.0={row[2]:.3f}")

# Also show the old d=2 values (from all_methods for n11 and n105)
print("\nCross-check with all_methods.json:")
n11_avg = sum(m["sil"] for m in all_methods if m["method"] == "n11") / 6
n105_avg = sum(m["sil"] for m in all_methods if m["method"] == "n105") / 6
print(f"  n11 (p=1,q=1) avg: {n11_avg:.3f}")
print(f"  n105 (p=1,q=0.5) avg: {n105_avg:.3f}")

print("\n" + "=" * 80)
print("STEP 4: DEEPWALK REMOVAL")
print("=" * 80)
# The neutral config (p=1,q=1) is the DeepWalk-equivalent
print(f"Node2Vec (p=1,q=1,d=2) avg = {avg_n11_d2:.3f} — this is the DeepWalk-equivalent")
print(f"Node2Vec (p=0.5,q=0.5,d=2) avg = {avg_n0505_d2:.3f} — best d=2")
best_overall = grid["best_overall"]
print(f"Best overall: p={best_overall['p']}, q={best_overall['q']}, d={best_overall['d']}, avg={best_overall['avg_silhouette']:.3f}")

# Relative improvement over PCA
pca_avg = 0.466  # from all_methods
rel_improvement_neutral = (avg_n11_d2 - pca_avg) / pca_avg * 100
rel_improvement_best = (avg_n0505_d2 - pca_avg) / pca_avg * 100
print(f"Relative improvement of N2V(1,1,d=2) over PCA: {rel_improvement_neutral:.1f}%")
print(f"Relative improvement of N2V(0.5,0.5,d=2) over PCA: {rel_improvement_best:.1f}%")

# Win rates
print("\nWin rates:")
for pair_name, method_a, method_b in [
    ("N2V(1,1,d=2) vs BoW", "n11", "bow"),
    ("N2V(0.5,0.5,d=2) vs BoW", None, "bow"),
    ("N2V(1,1,d=2) vs PCA", "n11", "pca"),
    ("N2V(0.5,0.5,d=2) vs PCA", None, "pca"),
]:
    if method_a:
        a_sils = [next(m for m in all_methods if m["course"] == c and m["method"] == method_a)["sil"] for c in courses]
    else:
        a_sils = n0505_d2
    b_sils = [next(m for m in all_methods if m["course"] == c and m["method"] == method_b)["sil"] for c in courses]
    wins = sum(1 for a, b in zip(a_sils, b_sils) if a > b)
    print(f"  {pair_name}: {wins}/6 ({wins/6*100:.0f}%)")

print("\n" + "=" * 80)
print("STEP 5: REMAINING TABLES")
print("=" * 80)

# Walk length sensitivity
print("\nWalk length sensitivity:")
for wl in [5, 10, 20, 40, 80]:
    entries = [e for e in sens_wl if e["wl"] == wl]
    avg = sum(e["sil"] for e in entries) / len(entries)
    per_course = {e["course"]: e["sil"] for e in entries}
    print(f"  WL={wl}: avg={avg:.3f}, per-course={[round(per_course[c], 3) for c in courses]}")

# Number of walks sensitivity
print("\nNumber of walks sensitivity:")
for nw in [10, 20, 40, 80, 160]:
    entries = [e for e in sens_nw if e["nw"] == nw]
    avg = sum(e["sil"] for e in entries) / len(entries)
    print(f"  NW={nw}: avg={avg:.3f}")

# Window size sensitivity
print("\nWindow size sensitivity:")
for ws in [1, 2, 5, 10, 20]:
    entries = [e for e in sens_ws if e["ws"] == ws]
    avg = sum(e["sil"] for e in entries) / len(entries)
    print(f"  WS={ws}: avg={avg:.3f}")

# Stability (silhouette std across seeds)
print("\nStability (silhouette std across 20 seeds):")
for c in courses:
    seeds = stability[str(c)]
    sils = [s["sil"] for s in seeds]
    mean_sil = sum(sils) / len(sils)
    std_sil = (sum((s - mean_sil) ** 2 for s in sils) / len(sils)) ** 0.5
    print(f"  Course {c}: mean={mean_sil:.3f}, std={std_sil:.3f}")
overall_stds = []
for c in courses:
    seeds = stability[str(c)]
    sils = [s["sil"] for s in seeds]
    mean_sil = sum(sils) / len(sils)
    std_sil = (sum((s - mean_sil) ** 2 for s in sils) / len(sils)) ** 0.5
    overall_stds.append(std_sil)
print(f"  Mean std across courses: {sum(overall_stds)/len(overall_stds):.3f}")

# ARI stability
print("\nARI stability:")
ari_vals = [ari[str(c)]["mean"] for c in courses]
print(f"  Per-course ARI: {[round(v, 3) for v in ari_vals]}")
print(f"  Average ARI: {sum(ari_vals)/len(ari_vals):.3f}")

# Runtime — DO NOT CHANGE (measured on another computer)
print("\nRuntime (NOT changing — measured on another computer):")
for c in courses:
    c_entries = [r for r in runtime if r["course"] == c]
    total = next(e["tt"] for e in c_entries if "Node2Vec" in e["method"] and "1,1" in e["method"])
    print(f"  Course {c}: {total:.3f}s")

print("\n" + "=" * 80)
print("INCONSISTENCY CHECK")
print("=" * 80)

# Check: old manuscript numbers vs new JSON numbers
print("Old manuscript numbers that need updating:")
print("  Table 1 old: N2V(1,1) avg=0.613, N2V(1,0.5) avg=0.611, PCA avg=0.460, BoW avg=0.153, Spectral avg=0.213")
print(f"  Table 1 new: N2V(1,1,d=2) avg={avg_n11_d2:.3f}, N2V(0.5,0.5,d=2) avg={avg_n0505_d2:.3f}")
print(f"  Baselines: BoW avg=0.157, PCA avg=0.466, Spectral avg=0.128")

print("\n  Old Table 3: 0.620, 0.616, 0.614, 0.622, 0.611, 0.605, 0.615, 0.615, 0.616")
print("  These were for d=2 but with different config (n105 = p=1, q=0.5)")
print("  New Table 3 should use grid_search_pqd.json d=2 values")

print("\n  Old Table 4: d=1: 0.552, d=2: 0.579, d=3: 0.377, d=5: 0.204, d=10: 0.088")
print("  These were for neutral config (p=1,q=1)")
d1_sil = sum(e["sil"] for e in sens_dim if e["dim"] == 1) / 6
d2_sil = sum(e["sil"] for e in sens_dim if e["dim"] == 2) / 6
d3_sil = sum(e["sil"] for e in sens_dim if e["dim"] == 3) / 6
d5_sil = sum(e["sil"] for e in sens_dim if e["dim"] == 5) / 6
d10_sil = sum(e["sil"] for e in sens_dim if e["dim"] == 10) / 6
print(f"  New Table 4: d=1: {d1_sil:.3f}, d=2: {d2_sil:.3f}, d=3: {d3_sil:.3f}, d=5: {d5_sil:.3f}, d=10: {d10_sil:.3f}")
print(f"  NOTE: d=1 ({d1_sil:.3f}) > d=2 ({d2_sil:.3f}) — the old paper claimed d=2 was best!")

# Check statistical analysis
print("\nStatistical analysis from JSON:")
for s in stats:
    print(f"  {s['name']}: diff={s['diff']:.3f}, delta={s['delta']:.3f}, r={s['r']:.3f}")
