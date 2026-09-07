#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitive multi-seed comparison of the three Pareto candidates.

All numbers come from EXISTING artifacts (no experiments re-run):
  (2.0, 1.0, 1)  results/final_d1_primary/         20-seed stability + labels
  (1.0, 1.0, 2)  results/final_d2/                 20-seed stability + labels (n11)
  (1.0, 0.5, 2)  results/final_pareto_validation/  20-seed validation (has balance)

Per-seed balance for the first two is computed here from their saved per-seed
label files using the same definition as everywhere else:
  B = min(|S1|,|S2|) / max(|S1|,|S2|)

Output: results/final_pareto_validation/comparison_3candidates.json
"""

import os
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

CANDIDATES = {
    "(2.0, 1.0, 1)": os.path.join(RES, "final_d1_primary"),
    "(1.0, 1.0, 2)": os.path.join(RES, "final_d2"),
    "(1.0, 0.5, 2)": os.path.join(RES, "final_pareto_validation"),
}
TAG = {"(2.0, 1.0, 1)": "n11",            # final_d1_primary labels: ..._n11_seed*.json
       "(1.0, 1.0, 2)": "n11",            # final_d2 labels: ..._n11_seed*.json
       "(1.0, 0.5, 2)": "p1.0_q0.5_d2"}   # validation labels: ..._p1.0_q0.5_d2_seed*.json


def balance_of(cluster_labels):
    c0 = sum(1 for x in cluster_labels if x == 0)
    c1 = len(cluster_labels) - c0
    if c0 == 0 or c1 == 0:
        return None
    return min(c0, c1) / max(c0, c1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect(cand_dir, tag):
    """Per-course, per-seed sil/balance from stability.json + label files."""
    stab = load_json(os.path.join(cand_dir, "stability.json"))
    out = {}
    for course, rows in stab.items():
        per_seed = []
        for r in rows:
            seed = r["seed"]
            if tag == "p1.0_q0.5_d2":
                lf = os.path.join(cand_dir, f"labels_course{course}_{tag}_seed{seed}.json")
            else:
                lf = os.path.join(cand_dir, f"labels_course{course}_{tag}_seed{seed}.json")
            B = None
            if os.path.exists(lf):
                B = balance_of(load_json(lf)["cluster_labels"])
            per_seed.append({"seed": seed, "sil": r["sil"], "balance": B})
        out[course] = per_seed
    return out


def summarize(per_course):
    sils, bals = [], []
    for course, rows in per_course.items():
        sils += [r["sil"] for r in rows]
        bals += [r["balance"] for r in rows if r["balance"] is not None]
    return {
        "sil_mean": float(np.mean(sils)), "sil_std_within": float(np.mean(
            [np.std([r["sil"] for r in rows]) for rows in per_course.values()])),
        "balance_mean": float(np.mean(bals)),
        "balance_min_course_mean": float(min(
            np.mean([r["balance"] for r in rows if r["balance"] is not None])
            for rows in per_course.values())),
        "n_seed_course_pairs": len(sils),
    }


def main():
    result = {}
    for cfg, cand_dir in CANDIDATES.items():
        per_course = collect(cand_dir, TAG[cfg])
        s = summarize(per_course)
        s["per_course"] = {}
        for course, rows in per_course.items():
            b = [r["balance"] for r in rows if r["balance"] is not None]
            s["per_course"][course] = {
                "sil_mean": float(np.mean([r["sil"] for r in rows])),
                "balance_mean": float(np.mean(b)) if b else None,
            }
        ari = load_json(os.path.join(cand_dir, "ari_stability.json"))
        s["ari_mean"] = float(np.mean([v["mean"] for v in ari.values()]))
        s["ari_per_course"] = {k: v["mean"] for k, v in ari.items()}
        result[cfg] = s

    out_path = os.path.join(RES, "final_pareto_validation",
                            "comparison_3candidates.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out_path}\n")

    hdr = f"{'config':<15} {'Sil(mean±sd)':<16} {'Balance(mean)':<14} {'minB(course)':<13} {'ARI':<6}"
    print(hdr)
    print("-" * len(hdr))
    for cfg, s in result.items():
        print(f"{cfg:<15} "
              f"{s['sil_mean']:.3f} ± {s['sil_std_within']:.3f}      "
              f"{s['balance_mean']:.3f}         "
              f"{s['balance_min_course_mean']:.3f}        "
              f"{s['ari_mean']:.3f}")
    print()
    for cfg, s in result.items():
        print(cfg, "per-course balance:",
              {c: (round(v["balance_mean"], 3) if v["balance_mean"] is not None else None)
               for c, v in s["per_course"].items()})


if __name__ == "__main__":
    main()
