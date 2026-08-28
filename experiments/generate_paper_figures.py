#!/usr/bin/env python3
"""Generate publication figures for paper/ directory from final experimental results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PAPER_DIR = Path('paper')
FIG_DIR = Path('results/final_reexperiment/figures')

with open('results/final_reexperiment/all_methods.json') as f:
    methods_data = json.load(f)
with open('results/final_reexperiment/sensitivity.json') as f:
    sens_data = json.load(f)
with open('results/final_reexperiment/graph_analysis.json') as f:
    graph_data = json.load(f)
with open('results/final_reexperiment/stability.json') as f:
    stab_data = json.load(f)
with open('results/final_reexperiment/runtime.json') as f:
    runtime_data = json.load(f)

df = pd.DataFrame(methods_data)
METHOD_ORDER = ['bow', 'pca', 'spectral', 'deepwalk', 'node2vec']
METHOD_NAMES = {
    'bow': 'BoW +\nKMeans', 'pca': 'PCA +\nKMeans', 'spectral': 'Spectral',
    'deepwalk': 'DeepWalk', 'node2vec': 'Node2Vec\n(p=1, q=1)'
}
METHOD_COLORS = {
    'bow': '#d62728', 'pca': '#ff7f0e', 'spectral': '#1f77b4',
    'deepwalk': '#2ca02c', 'node2vec': '#9467bd'
}
COURSES = [1, 2, 3, 4, 5, 6]

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.pdf', format='pdf')
    fig.savefig(FIG_DIR / f'{name}.png', format='png')
    fig.savefig(PAPER_DIR / f'{name}.png', format='png')
    print(f'  Saved: {name}')
    plt.close(fig)


# 1. Baseline Comparison
print('1. Baseline comparison...')
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(COURSES))
width = 0.15
offsets = np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2
for i, method in enumerate(METHOD_ORDER):
    mdf = df[df['method'] == method].sort_values('course')
    vals = mdf['silhouette_emb'].values
    bars = ax.bar(x + offsets[i] * width, vals, width,
                  label=METHOD_NAMES[method], color=METHOD_COLORS[method],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
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
save_fig(fig, 'baseline_comparison')

# 2. All Metrics
print('2. All metrics comparison...')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metrics = [('silhouette_emb', 'Silhouette Score (higher is better)'),
           ('dbi_emb', 'Davies-Bouldin Index (lower is better)'),
           ('ch_emb', 'Calinski-Harabasz Index (higher is better)')]
for ax, (col, ylabel) in zip(axes, metrics):
    x = np.arange(len(COURSES))
    for i, method in enumerate(METHOD_ORDER):
        mdf = df[df['method'] == method].sort_values('course')
        vals = mdf[col].values
        ax.bar(x + offsets[i] * width, vals, width,
               label=METHOD_NAMES[method], color=METHOD_COLORS[method],
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Course')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in COURSES], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax == axes[0]:
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
fig.tight_layout()
save_fig(fig, 'method_comparison_all_metrics')

# 3. Silhouette only
print('3. Silhouette comparison...')
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, method in enumerate(METHOD_ORDER):
    mdf = df[df['method'] == method].sort_values('course')
    vals = mdf['silhouette_emb'].values
    ax.bar(x + offsets[i] * width, vals, width,
           label=METHOD_NAMES[method], color=METHOD_COLORS[method],
           edgecolor='white', linewidth=0.5)
ax.set_xlabel('Course')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_fig(fig, 'method_comparison_silhouette')

# 4. Node2Vec vs DeepWalk
print('4. Node2Vec vs DeepWalk...')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (col, ylabel) in zip(axes, [
    ('silhouette_emb', 'Silhouette Score (higher is better)'),
    ('dbi_emb', 'Davies-Bouldin Index (lower is better)'),
    ('ch_emb', 'Calinski-Harabasz Index (higher is better)')
]):
    for method, color, label in [
        ('deepwalk', '#2ca02c', 'DeepWalk'),
        ('node2vec', '#9467bd', 'Node2Vec (p=1, q=1)')
    ]:
        mdf = df[df['method'] == method].sort_values('course')
        vals = mdf[col].values
        ax.plot(COURSES, vals, 'o-', color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('Course')
    ax.set_ylabel(ylabel)
    ax.set_xticks(COURSES)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'node2vec_vs_deepwalk')

# 5. Sensitivity Heatmap
print('5. Sensitivity heatmap...')
sens_df = pd.DataFrame(sens_data)
pq_pivot = sens_df.groupby(['p', 'q'])['silhouette'].mean().reset_index()
pq_matrix = pq_pivot.pivot(index='p', columns='q', values='silhouette')
pq_matrix = pq_matrix.sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(pq_matrix.values, cmap='YlOrRd', aspect='auto', vmin=0.60, vmax=0.625)
ax.set_xticks(range(len(pq_matrix.columns)))
ax.set_xticklabels([f'{c:.1f}' for c in pq_matrix.columns])
ax.set_yticks(range(len(pq_matrix.index)))
ax.set_yticklabels([f'{r:.1f}' for r in pq_matrix.index])
ax.set_xlabel('q (in-out parameter)')
ax.set_ylabel('p (return parameter)')
ax.set_title('Node2Vec Parameter Sensitivity\n(avg Silhouette across 6 courses)')
for i in range(len(pq_matrix.index)):
    for j in range(len(pq_matrix.columns)):
        val = pq_matrix.values[i, j]
        text_color = 'white' if val > 0.618 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='bold')
fig.colorbar(im, label='Silhouette Score', shrink=0.8)
fig.tight_layout()
save_fig(fig, 'sensitivity_heatmap')

# 6. Stability
print('6. Stability...')
fig, ax = plt.subplots(figsize=(8, 5))
n2v_means, n2v_stds = [], []
dw_means, dw_stds = [], []
labels = []
for cid in ['1', '2', '3', '4', '5', '6']:
    n2v_sil = stab_data[cid]['node2vec']['silhouette']
    dw_sil = stab_data[cid]['deepwalk']['silhouette']
    n2v_means.append(n2v_sil['mean'])
    n2v_stds.append(n2v_sil['std'])
    dw_means.append(dw_sil['mean'])
    dw_stds.append(dw_sil['std'])
    labels.append(f'C{cid}')
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, n2v_means, width, yerr=n2v_stds,
       label='Node2Vec (p=1, q=1)', color='#9467bd', alpha=0.7,
       capsize=3, edgecolor='white')
ax.bar(x + width/2, dw_means, width, yerr=dw_stds,
       label='DeepWalk', color='#2ca02c', alpha=0.7,
       capsize=3, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Course')
ax.set_ylabel('Silhouette Score')
ax.set_title('Clustering Stability across 20 Random Seeds (mean +/- std)')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'stability_boxplot')

# 7. Graph Density
print('7. Graph density comparison...')
fig, ax = plt.subplots(figsize=(7, 4))
old_density = [1.0] * 6
new_density = [graph_data['new'][str(c)]['density'] for c in COURSES]
edges_new = [graph_data['new'][str(c)]['n_edges'] for c in COURSES]
x = np.arange(len(COURSES))
width = 0.35
ax.bar(x - width/2, old_density, width, label='Before preprocessing',
       color='#d62728', alpha=0.7, edgecolor='white')
bars2 = ax.bar(x + width/2, new_density, width, label='After preprocessing',
               color='#2ca02c', alpha=0.7, edgecolor='white')
for bar, e in zip(bars2, edges_new):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{e}', ha='center', va='bottom', fontsize=8, color='#2ca02c')
ax.set_xlabel('Course')
ax.set_ylabel('Graph Density')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend()
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Effect of Constant-Course Removal on Graph Density')
fig.tight_layout()
save_fig(fig, 'graph_density_comparison')

# 8. Runtime Comparison
print('8. Runtime comparison...')
rt_df = pd.DataFrame(runtime_data)
rt_pivot = rt_df.pivot(index='course', columns='method', values='t_total')
fig, ax = plt.subplots(figsize=(8, 5))
rt_methods = list(rt_pivot.columns)
x = np.arange(len(COURSES))
n_methods = len(rt_methods)
width = 0.8 / n_methods
colors_rt = plt.cm.Set2(np.linspace(0, 1, n_methods))
for i, mname in enumerate(rt_methods):
    vals = [rt_pivot.loc[c, mname] for c in COURSES]
    ax.bar(x + (i - n_methods/2 + 0.5) * width, vals, width,
           label=mname, color=colors_rt[i], edgecolor='white')
ax.set_xlabel('Course')
ax.set_ylabel('Time (seconds)')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(fontsize=7, ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Pipeline Runtime Comparison')
fig.tight_layout()
save_fig(fig, 'runtime_comparison')

print('\n=== All figures generated ===')
