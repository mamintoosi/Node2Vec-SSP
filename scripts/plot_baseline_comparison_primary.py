#!/usr/bin/env python3
"""Regenerate the baseline-comparison figure for the PRIMARY (Pareto-selected)
Node2Vec configuration (p=1.0, q=0.5, d=2).

Data sources (existing artifacts only, no experiments re-run):
  - results/final_d2/all_methods.json    (bow, pca, spec, n105 = (1.0, 0.5, d=2), seed 42)
  - results/reproduced/all_methods.json  (bow, pca, spec, seed 42)

Output (new files; nothing existing is overwritten):
  - paper/baseline_comparison_primary.pdf / .png
  - results/final_pareto_validation/figures/baseline_comparison_primary.pdf / .png

Style follows experiments/generate_paper_figures.py.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / 'paper'
FIG_DIR = ROOT / 'results' / 'final_pareto_validation' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / 'results' / 'final_d2' / 'all_methods.json') as f:
    methods_data = json.load(f)

# n105 is the (P, Q-0.5) variant in the module => (p=1.0, q=0.5, d=2): the PRIMARY config.
METHOD_ORDER = ['bow', 'pca', 'spec', 'n105']
METHOD_NAMES = {
    'bow': 'BoW +\nKMeans',
    'pca': 'PCA +\nKMeans',
    'spec': 'Spectral',
    'n105': 'Node2Vec\n(p=1.0, q=0.5)',
}
METHOD_COLORS = {
    'bow': '#d62728', 'pca': '#ff7f0e', 'spec': '#1f77b4', 'n105': '#9467bd',
}
COURSES = [1, 2, 3, 4, 5, 6]

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(COURSES))
width = 0.19
offsets = np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2
for i, method in enumerate(METHOD_ORDER):
    vals = [r['sil'] for r in methods_data if r['method'] == method]
    bars = ax.bar(x + offsets[i] * width, vals, width,
                  label=METHOD_NAMES[method], color=METHOD_COLORS[method],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6.5)

ax.set_xlabel('Course')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for out_dir in (PAPER_DIR, FIG_DIR):
    fig.savefig(out_dir / 'baseline_comparison_primary.pdf', format='pdf')
    fig.savefig(out_dir / 'baseline_comparison_primary.png', format='png')
plt.close(fig)

print('Saved: paper/baseline_comparison_primary.{pdf,png}')
print(f'Saved: {FIG_DIR}/baseline_comparison_primary.{{pdf,png}}')

# Quick sanity print of averages
for m in METHOD_ORDER:
    vals = [r['sil'] for r in methods_data if r['method'] == m]
    print(f'  {m:>5}: avg sil = {np.mean(vals):.4f}')
