# -*- coding: utf-8 -*-
"""Generate remaining figures: method comparison and runtime."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REEXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "results", "revised_reexperiment")
FIGURES_DIR = os.path.join(REEXPERIMENT_DIR, "figures")

PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {"bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B", "node2vec": "#4CAF50"}

def main():
    plt.rcParams.update(PUB_STYLE)
    
    # Load all methods data
    with open(os.path.join(REEXPERIMENT_DIR, "all_methods.json")) as f:
        methods_data = json.load(f)
    
    courses = sorted(set(d["course"] for d in methods_data))
    
    # Method comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    methods_info = [
        ("bow", "BoW + KMeans", COLORS["bow"]),
        ("pca", "PCA + KMeans", COLORS["pca"]),
        ("spectral", "Spectral Clustering", COLORS["spectral"]),
        ("node2vec", "Node2Vec (p=1, q=1)", COLORS["node2vec"]),
    ]
    x = np.arange(len(courses))
    width = 0.18
    for i_m, (mk, ml, mc) in enumerate(methods_info):
        vals = []
        for c in courses:
            match = [d for d in methods_data if d["course"] == c and d["method"] == mk]
            vals.append(match[0]["sil"] if match else 0)
        offset = (i_m - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=6)
    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score (↑ higher is better)')
    ax.set_title('Method Comparison: Clustering Quality (Revised Graph Construction)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "method_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: method_comparison.pdf")
    
    # Runtime comparison
    with open(os.path.join(REEXPERIMENT_DIR, "runtime.json")) as f:
        rt_data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_rt = ["BoW+KMeans", "PCA+KMeans", "Spectral", "Node2Vec"]
    colors_rt = [COLORS["bow"], COLORS["pca"], COLORS["spectral"], COLORS["node2vec"]]
    for i_m, (ml, mc) in enumerate(zip(methods_rt, colors_rt)):
        vals = []
        for c in courses:
            match = [d for d in rt_data if d["course"] == c and d["method"] == ml]
            vals.append(match[0]["t_total"] if match else 0)
        offset = (i_m - 1.5) * 0.18
        ax.bar(x + offset, vals, 0.18, label=ml, color=mc, alpha=0.85)
    ax.set_xlabel('Course')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in courses])
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGURES_DIR, "runtime_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: runtime_comparison.pdf")
    
    # Summary table
    print("\n  METHOD COMPARISON SUMMARY")
    print(f"  {'Method':<20}", end="")
    for c in courses:
        print(f" {'C'+str(c):>6}", end="")
    print(f" {'Avg':>8}")
    print("  " + "-" * 60)
    for mk, ml, _ in methods_info:
        print(f"  {ml:<20}", end="")
        vals = []
        for c in courses:
            match = [d for d in methods_data if d["course"] == c and d["method"] == mk]
            v = match[0]["sil"] if match else 0
            vals.append(v)
            print(f" {v:>6.3f}", end="")
        print(f" {np.mean(vals):>8.3f}")


if __name__ == "__main__":
    main()
