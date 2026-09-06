# -*- coding: utf-8 -*-
"""
Pareto Analysis: Clustering Quality vs Section Balance
======================================================
Uses ONLY saved artifacts:
  - results/grid_search/grid_search_complete.json        (per-course Silhouette, seed 42, all 45 configs)
  - results/grid_search/labels_p*_q*_d*_course{1..6}.json (cluster labels, seed 42)
  - results/reproduced/grid_search_pqd.json              (per-course DBI/CH, seed 42)
  - results/reproduced_seed0/grid_search_pqd.json        (per-course Silhouette, seed 0)
  - results/reproduced/all_methods.json + section_balance.json (baselines, seed 42)

Outputs:
  - results/grid_search/pareto_analysis.json
  - printed tables (avg + per-course)

No Node2Vec/Word2Vec/KMeans re-runs. CPU use is trivial.
"""
import json
import os
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "results", "grid_search")
REP = os.path.join(ROOT, "results", "reproduced")
REP0 = os.path.join(ROOT, "results", "reproduced_seed0")

COURSES = [1, 2, 3, 4, 5, 6]


def load(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------- labels -> balance
def balance_from_labels(path):
    d = load(path)
    cl = d["cluster_labels"]
    c0 = sum(1 for x in cl if x == 0)
    c1 = len(cl) - c0
    if c0 == 0 or c1 == 0:
        return None, (c0, c1)
    return min(c0, c1) / max(c0, c1), (c0, c1)


def collect_all_balances():
    """Compute balance for every saved label file: dict[(p,q,d)][course] = (B, s1, s2)."""
    bal = defaultdict(dict)
    for path in glob.glob(os.path.join(GS, "labels_p*_q*_d*_course*.json")):
        fname = os.path.basename(path)
        # parse p{float}_q{float}_d{int}_course{int}.json
        body = fname[len("labels_"):-len(".json")]
        parts = body.split("_")
        p = float(parts[0][1:])
        q = float(parts[1][1:])
        d = int(parts[2][1:])
        course = int(parts[3][len("course"):])
        B, (c0, c1) = balance_from_labels(path)
        bal[(p, q, d)][course] = {"B": B, "s1": c0, "s2": c1}
    return bal


# ---------------------------------------------------------------- metrics per config
def avg_per_course(grid_json, key_metric, key_p="p", key_q="q", key_d="d"):
    """grid_json['results'] rows -> dict[(p,q,d)][course] = value."""
    per = defaultdict(dict)
    for row in grid_json["results"]:
        k = (float(row[key_p]), float(row[key_q]), int(row[key_d]))
        per[k][int(row["course"])] = row[key_metric]
    return per


def main():
    complete = load(os.path.join(GS, "grid_search_complete.json"))
    rep_grid = load(os.path.join(REP, "grid_search_pqd.json"))
    try:
        seed0_grid = load(os.path.join(REP0, "grid_search_pqd.json"))
    except FileNotFoundError:
        seed0_grid = None

    sil42 = avg_per_course(complete, "silhouette")

    # DBI/CH per configuration: available for d=1 and d=2 (all 9 (p,q) each)
    # via final_d1/final_d2 sensitivity_pq.json, and for the neutral config at
    # all dims via reproduced/sensitivity_dim.json.
    dbi42 = defaultdict(dict)
    ch42 = defaultdict(dict)
    def add_dbich(rows, pk="p", qk="q", dk="d"):
        for row in rows:
            if "dbi" not in row or "ch" not in row:
                continue
            k = (float(row[pk]), float(row[qk]), int(row[dk]))
            dbi42[k][int(row["course"])] = row["dbi"]
            ch42[k][int(row["course"])] = row["ch"]
    for ddir in ("final_d1", "final_d2"):
        pq_path = os.path.join(ROOT, "results", ddir, "sensitivity_pq.json")
        if os.path.exists(pq_path):
            add_dbich(load(pq_path))
    dim_path = os.path.join(REP, "sensitivity_dim.json")
    if os.path.exists(dim_path):
        for row in load(dim_path):
            k = (1.0, 1.0, int(row["dim"]))
            if "dbi" in row:
                dbi42[k][int(row["course"])] = row["dbi"]
                ch42[k][int(row["course"])] = row["ch"]

    sil0 = avg_per_course(seed0_grid, "silhouette") if seed0_grid else {}

    bal = collect_all_balances()

    # baselines from reproduced artifacts
    all_methods = load(os.path.join(REP, "all_methods.json"))
    bal_json = load(os.path.join(REP, "section_balance.json"))

    base_metrics = defaultdict(dict)   # method -> course -> sil/dbi/ch
    for row in all_methods:
        base_metrics[row["method"]][int(row["course"])] = row
    base_bal = defaultdict(dict)       # method -> course -> balance
    for row in bal_json:
        base_bal[row["method"]][int(row["course"])] = row["balance"]

    configs = sorted(sil42.keys())
    print(f"Grid configurations with Silhouette (seed42): {len(configs)}")
    print(f"Configurations with saved cluster labels:     {len(bal)}")
    missing_labels = [c for c in configs if c not in bal or len(bal[c]) < 6]
    if missing_labels:
        print("WARNING - configs missing labels:", missing_labels)

    # ---------------- assemble records
    records = []
    for k in configs:
        p, q, d = k
        sil_per = [sil42[k].get(c) for c in COURSES]
        b_per = [bal[k][c]["B"] for c in COURSES if c in bal[k]]
        B_avg = sum(b_per) / len(b_per) if len(b_per) == 6 else None
        rec = {
            "p": p, "q": q, "d": d,
            "avg_sil": sum(v for v in sil_per if v is not None) / 6.0,
            "per_course_sil": {c: sil42[k].get(c) for c in COURSES},
            "avg_B": B_avg,
            "per_course_B": {c: bal[k][c]["B"] for c in COURSES if c in bal[k]},
            "per_course_sizes": {c: [bal[k][c]["s1"], bal[k][c]["s2"]]
                                 for c in COURSES if c in bal[k]},
            "dbi_avg": sum(dbi42[k].values()) / len(dbi42[k]) if k in dbi42 and dbi42[k] else None,
            "ch_avg": sum(ch42[k].values()) / len(ch42[k]) if k in ch42 and ch42[k] else None,
            "has_labels": k in bal and len(bal[k]) == 6,
            "seed": 42,
        }
        if k in sil0 and len(sil0[k]) == 6:
            rec["avg_sil_seed0"] = sum(sil0[k].values()) / 6.0
        records.append(rec)

    # ---------------- Pareto frontier (maximize avg_sil, maximize avg_B)
    def dominates(a, b):
        """a dominates b: a >= b in both, strictly > in at least one."""
        if a["avg_sil"] is None or a["avg_B"] is None or b["avg_sil"] is None or b["avg_B"] is None:
            return False
        ge = a["avg_sil"] >= b["avg_sil"] and a["avg_B"] >= b["avg_B"]
        gt = a["avg_sil"] > b["avg_sil"] or a["avg_B"] > b["avg_B"]
        return ge and gt

    pareto = []
    for a in records:
        if a["avg_B"] is None:
            continue
        if not any(dominates(b, a) for b in records if b is not a):
            pareto.append(a)

    # "close to frontier": sil >= 0.70 and B >= 0.6 (and not already Pareto)
    close = [r for r in records
             if r["avg_B"] is not None and r not in pareto
             and r["avg_sil"] >= 0.70 and r["avg_B"] >= 0.60]

    out = {
        "description": "Pareto analysis over saved grid-search results (seed=42). "
                       "Objectives: maximize avg Silhouette, maximize avg Section Balance.",
        "pareto_optimal": pareto,
        "close_to_frontier": close,
        "all_configs": records,
    }
    outpath = os.path.join(GS, "pareto_analysis.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- printing
    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if v is not None else "  -  "

    print("\n================ PARETO-OPTIMAL CONFIGURATIONS ================")
    print(f"{'p':>4} {'q':>4} {'d':>3} | {'AvgSil':>7} {'AvgB':>6} {'DBI':>6} {'CH':>7} | {'sil_s0':>7}")
    for r in sorted(pareto, key=lambda r: (-r["avg_sil"], -r["avg_B"])):
        print(f"{r['p']:>4} {r['q']:>4} {r['d']:>3} | {fmt(r['avg_sil'])} {fmt(r['avg_B'],3)} "
              f"{fmt(r['dbi_avg'])} {fmt(r['ch_avg'],1)} | {fmt(r.get('avg_sil_seed0'))}")
    print("(DBI/CH shown where available: all d=1 and d=2 configs; neutral config at other dims)")

    print("\n---- per-course detail (Pareto set) ----")
    for r in sorted(pareto, key=lambda r: (-r["avg_sil"], -r["avg_B"])):
        sils = " ".join(f"{r['per_course_sil'][c]:.3f}" for c in COURSES)
        bs = " ".join(f"{r['per_course_B'][c]:.3f}" for c in COURSES)
        sizes = " ".join(f"{r['per_course_sizes'][c][0]}/{r['per_course_sizes'][c][1]}"
                         for c in COURSES)
        print(f"(p={r['p']},q={r['q']},d={r['d']})")
        print(f"   sil: {sils}")
        print(f"   B  : {bs}")
        print(f"   n1/n2: {sizes}")

    print("\n=========== CLOSE TO FRONTIER (sil>=0.70 and B>=0.6) ===========")
    for r in sorted(close, key=lambda r: (-r["avg_B"], -r["avg_sil"])):
        print(f"(p={r['p']},q={r['q']},d={r['d']})  AvgSil={fmt(r['avg_sil'])}  AvgB={fmt(r['avg_B'],3)}"
              f"  DBI={fmt(r['dbi_avg'])}  CH={fmt(r['ch_avg'],1)}")

    print("\n================ TOP 10 BY AVG SILHOUETTE (all) ================")
    for r in sorted(records, key=lambda r: -r["avg_sil"])[:10]:
        print(f"(p={r['p']},q={r['q']},d={r['d']})  AvgSil={fmt(r['avg_sil'])}  AvgB={fmt(r['avg_B'],3)}")

    print("\n================ TOP 10 BY AVG BALANCE (all) ==================")
    for r in sorted(records, key=lambda r: (-r["avg_B"], -r["avg_sil"]))[:10]:
        print(f"(p={r['p']},q={r['q']},d={r['d']})  AvgB={fmt(r['avg_B'],3)}  AvgSil={fmt(r['avg_sil'])}")

    # ---------------- selected configs detail
    for sel in [(2.0, 1.0, 1), (1.0, 1.0, 1), (1.0, 1.0, 2), (0.5, 0.5, 2), (2.0, 2.0, 1)]:
        match = [r for r in records if (r["p"], r["q"], r["d"]) == sel]
        if match:
            r = match[0]
            print(f"\n--- SELECTED DETAIL (p={sel[0]},q={sel[1]},d={sel[2]}) ---")
            print(f"  AvgSil={fmt(r['avg_sil'])}  AvgB={fmt(r['avg_B'],3)}  DBI={fmt(r['dbi_avg'])}  CH={fmt(r['ch_avg'],1)}")
            print(f"  per-course sil: {[round(r['per_course_sil'][c], 3) for c in COURSES]}")
            print(f"  per-course B  : {[round(r['per_course_B'][c], 3) for c in COURSES]}")
            print(f"  per-course n1/n2: {[tuple(r['per_course_sizes'][c]) for c in COURSES]}")

    # ---------------- baseline comparison
    print("\n================ BASELINES (seed=42, reproduced) ===============")
    names = {"bow": "BoW+KMeans", "pca": "PCA+KMeans", "spec": "Spectral",
             "n11": "Node2Vec(1,1,d2)", "n105": "Node2Vec(1,0.5,d2)"}
    for m in ["bow", "pca", "spec", "n11", "n105"]:
        sils = [base_metrics[m][c]["sil"] for c in COURSES if c in base_metrics[m]]
        dbis = [base_metrics[m][c]["dbi"] for c in COURSES if c in base_metrics[m]]
        chs = [base_metrics[m][c]["ch"] for c in COURSES if c in base_metrics[m]]
        bs = [base_bal[m][c] for c in COURSES if c in base_bal[m]]
        print(f"{names[m]:<22} AvgSil={sum(sils)/6:.4f} AvgB={sum(bs)/6:.4f} "
              f"DBI={sum(dbis)/6:.4f} CH={sum(chs)/6:.2f}")
        print(f"{'':22} per-course B: {[round(b, 3) for b in bs]}")

    # d=1 Node2Vec primary candidates vs baselines
    print("\n--- d=1 candidates vs baselines (avg) ---")
    for sel in [(2.0, 1.0, 1), (1.0, 1.0, 1), (2.0, 2.0, 1)]:
        r = next((x for x in records if (x["p"], x["q"], x["d"]) == sel), None)
        if r:
            print(f"Node2Vec(p={sel[0]},q={sel[1]},d=1): AvgSil={fmt(r['avg_sil'])}  AvgB={fmt(r['avg_B'],3)}")

    print(f"\nSaved: {os.path.relpath(outpath, ROOT)}")


if __name__ == "__main__":
    main()
