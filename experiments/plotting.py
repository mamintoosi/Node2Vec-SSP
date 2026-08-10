# -*- coding: utf-8 -*-
"""
Plotting Module for DeepWalk-SSP
==================================
Publication-quality figures for the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from experiments.config import (
    FIGURES_DIR, COLORS, PUBLICATION_STYLE, PLOT_STYLE
)


def setup_style(style="publication"):
    """Apply plotting style."""
    if style == "publication":
        plt.rcParams.update(PUBLICATION_STYLE)
    else:
        plt.rcParams.update(PLOT_STYLE)
    sns.set_palette("husl")


def save_fig(fig, filename, show=False):
    """Save figure to figures directory."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if not show:
        plt.close(fig)
    return path


# ── Figure 1: Silhouette Score Comparison Bar Chart ───────────────────────────

def plot_silhouette_comparison_bar(df, vector_size=2, filename="fig_silhouette_bar.png"):
    """
    Bar chart comparing Silhouette scores for BoW vs DeepWalk at given vector_size.
    """
    setup_style()
    
    filtered = df[df['vector_size'] == vector_size].copy()
    filtered = filtered.drop_duplicates(subset='file_index')
    
    # Prepare data for grouped bar chart
    file_indices = sorted(filtered['file_index'].unique())
    bow_scores = []
    dw_scores = []
    
    for fi in file_indices:
        row = filtered[filtered['file_index'] == fi]
        bow_scores.append(row['BoW.silhouette_score'].values[0])
        dw_scores.append(row['embeddings.silhouette_score'].values[0])
    
    x = np.arange(len(file_indices))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bow_scores, width, label='Student-Course Representation',
                   color=COLORS['bow'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, dw_scores, width, label='DeepWalk-SSP',
                   color=COLORS['deepwalk'], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Course')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'Silhouette Score Comparison (vector_size={vector_size})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {i}' for i in file_indices])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    return save_fig(fig, filename)


# ── Figure 2: Silhouette Score vs Vector Size ────────────────────────────────

def plot_silhouette_vs_vectorsize(df, filename="fig_silhouette_vs_vectorsize.png"):
    """Line plot of silhouette score vs embedding dimension."""
    setup_style()
    
    avg_scores = df.groupby('vector_size').agg({
        'BoW.silhouette_score': 'mean',
        'embeddings.silhouette_score': 'mean'
    }).reset_index()
    
    # Also compute std for error bars
    std_scores = df.groupby('vector_size').agg({
        'BoW.silhouette_score': 'std',
        'embeddings.silhouette_score': 'std'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(avg_scores['vector_size'], avg_scores['embeddings.silhouette_score'],
                yerr=std_scores['embeddings.silhouette_score'],
                marker='o', capsize=4, label='DeepWalk-SSP',
                color=COLORS['deepwalk'], linewidth=2, markersize=8)
    
    ax.errorbar(avg_scores['vector_size'], avg_scores['BoW.silhouette_score'],
                yerr=std_scores['BoW.silhouette_score'],
                marker='s', capsize=4, label='Student-Course Representation',
                color=COLORS['bow'], linewidth=2, markersize=8)
    
    ax.set_xlabel('Embedding Dimension (d)')
    ax.set_ylabel('Silhouette Score (mean ± std)')
    ax.set_title('Impact of Embedding Dimensionality on Clustering Quality')
    ax.set_xticks(avg_scores['vector_size'])
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.4, label='Poor threshold')
    ax.axhline(y=0.50, color='green', linestyle='--', alpha=0.4, label='Good threshold')
    
    return save_fig(fig, filename)


# ── Figure 3: Clustering Methods Comparison ───────────────────────────────────

def plot_clustering_methods_comparison(df_all_results, 
                                        methods,
                                        score_name="Silhouette Score",
                                        metric_key="silhouette",
                                        filename="fig_clustering_methods.png"):
    """Multi-panel line plot comparing clustering methods across courses."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    method_colors = {
        'DeepWalk-SSP kmeans': COLORS['kmeans'],
        'DeepWalk-SSP affinity': COLORS['affinity'],
        'DeepWalk-SSP gmm': COLORS['gmm'],
        'DeepWalk-SSP agglomerative': COLORS['agglomerative'],
        'Student-Course Representation kmeans': COLORS['kmeans'],
        'Student-Course Representation affinity': COLORS['affinity'],
        'Student-Course Representation gmm': COLORS['gmm'],
        'Student-Course Representation agglomerative': COLORS['agglomerative'],
    }
    
    method_linestyles = {
        'kmeans': '-',
        'affinity': '--',
        'gmm': '-.',
        'agglomerative': ':',
    }
    method_markers = {
        'kmeans': 'o',
        'affinity': 's',
        'gmm': '^',
        'agglomerative': 'D',
    }
    
    for method_name in df_all_results['Method'].unique():
        subset = df_all_results[df_all_results['Method'] == method_name]
        
        # Parse method type
        if 'kmeans' in method_name:
            mt = 'kmeans'
        elif 'affinity' in method_name:
            mt = 'affinity'
        elif 'gmm' in method_name:
            mt = 'gmm'
        else:
            mt = 'agglomerative'
        
        is_dw = 'DeepWalk' in method_name
        prefix = 'DeepWalk' if is_dw else 'BoW'
        
        label = f"{prefix} - {mt.capitalize()}"
        color = COLORS[mt]
        ls = '-' if is_dw else '--'
        marker = method_markers[mt]
        
        ax.plot(subset['File Index'], subset[score_name], 
                marker=marker, label=label, 
                color=color, linestyle=ls, linewidth=1.5, markersize=7)
    
    ax.set_xlabel('Course')
    ax.set_ylabel(score_name)
    ax.set_title(f'{score_name} Across Courses by Clustering Method')
    ax.set_xticks(range(1, 7))
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    return save_fig(fig, filename)


# ── Figure 4: Embedding Visualization ────────────────────────────────────────

def plot_embedding_visualization(data, labels, title, filename,
                                  reduced_2d=None, method_name="",
                                  axis_labels=None):
    """Plot 2D embedding visualization with cluster coloring.
    
    Args:
        axis_labels: Tuple of (xlabel, ylabel). If None, defaults to
                     ('Component 1', 'Component 2'). Use
                     ('Embedding Dimension 1', 'Embedding Dimension 2')
                     when plotting raw d=2 embeddings directly.
    """
    setup_style()
    
    if reduced_2d is None:
        return None
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    unique_labels = np.unique(labels)
    cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                   c=[cluster_colors[idx]], label=f'Cluster {label}',
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_title(title, fontsize=13)
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=11)
        ax.set_ylabel(axis_labels[1], fontsize=11)
    else:
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    return save_fig(fig, filename)


# ── Figure 5: Hyperparameter Sensitivity ─────────────────────────────────────

def plot_hyperparameter_sensitivity(results_dict, param_name, 
                                     xlabel, title,
                                     filename="fig_hyperparam_sensitivity.png"):
    """
    Plot hyperparameter sensitivity curves.
    
    Args:
        results_dict: {param_value: {metric_name: value, ...}}
        param_name: Parameter name to plot on x-axis.
        xlabel: X-axis label.
        title: Plot title.
    """
    setup_style()
    
    params = sorted(results_dict.keys())
    
    metrics = set()
    for v in results_dict.values():
        metrics.update(v.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, sorted(metrics)):
        values = [results_dict[p].get(metric, np.nan) for p in params]
        ax.plot(params, values, marker='o', linewidth=2, markersize=8, color=COLORS['deepwalk'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs {xlabel}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    
    return save_fig(fig, filename)


# ── Figure 6: Runtime Analysis ────────────────────────────────────────────────

def plot_runtime_analysis(runtime_data, filename="fig_runtime.png"):
    """Bar chart of runtime breakdown."""
    setup_style()
    
    categories = list(runtime_data.keys())
    graph_times = [v.get('graph_construction', 0) for v in runtime_data.values()]
    walk_times = [v.get('random_walks', 0) for v in runtime_data.values()]
    w2v_times = [v.get('word2vec_training', 0) for v in runtime_data.values()]
    total_times = [v.get('total', 0) for v in runtime_data.values()]
    
    x = np.arange(len(categories))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, graph_times, width, label='Graph Construction', color='#E91E63')
    ax.bar(x - 0.5*width, walk_times, width, label='Random Walks', color='#2196F3')
    ax.bar(x + 0.5*width, w2v_times, width, label='Word2Vec Training', color='#4CAF50')
    ax.bar(x + 1.5*width, total_times, width, label='Total', color='#FF9800', alpha=0.7)
    
    ax.set_xlabel('Course')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Breakdown by Course')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Course {c}' for c in categories])
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    return save_fig(fig, filename)


# ── Figure 7: Seed Stability ─────────────────────────────────────────────────

def plot_seed_stability(stability_results, filename="fig_seed_stability.png"):
    """Box plot of metric distributions across seeds."""
    setup_style()
    
    courses = sorted(stability_results.keys())
    
    # Collect all metrics
    all_metrics = set()
    for course_data in stability_results.values():
        for method_data in course_data.values():
            if 'metrics' in method_data:
                all_metrics.update(method_data['metrics'].keys())
    
    for metric in sorted(all_metrics):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        box_data = []
        labels = []
        
        for course in courses:
            for method in ['kmeans']:
                if method in stability_results[course]:
                    metrics_list = stability_results[course][method].get(f'{metric}_values', [])
                    if metrics_list:
                        box_data.append(metrics_list)
                        labels.append(f"Course {course}\n({method})")
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['deepwalk_alpha'])
                patch.set_edgecolor(COLORS['deepwalk'])
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution Across Seeds')
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        save_fig(fig, f"fig_seed_stability_{metric}.png")


# ── Figure 8: Statistical Comparison ─────────────────────────────────────────

def plot_statistical_comparison(stat_results, filename="fig_stat_comparison.png"):
    """Forest plot of Wilcoxon test results."""
    setup_style()
    
    metrics = list(stat_results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), len(metrics) * 1.5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        data = stat_results[metric]
        p_vals = data.get('p_values', [])
        effect_sizes = data.get('effect_sizes', [])
        methods = data.get('methods', [])
        
        y_pos = np.arange(len(methods))
        
        ax.barh(y_pos, effect_sizes, color=COLORS['deepwalk'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel('Effect Size (r)')
        ax.set_title(f'{metric}\np-values shown')
        
        for i, (p, e) in enumerate(zip(p_vals, effect_sizes)):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.annotate(f'p={p:.4f} {sig}', xy=(e, i), xytext=(5, 0),
                       textcoords='offset points', va='center', fontsize=8)
    
    fig.suptitle('Statistical Significance of DeepWalk Improvement', fontsize=14, y=1.02)
    fig.tight_layout()
    
    return save_fig(fig, filename)
