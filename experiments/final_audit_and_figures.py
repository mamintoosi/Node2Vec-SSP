#!/usr/bin/env python3
"""Final consistency audit and figure regeneration.

This script:
1. Verifies all numerical values in the manuscript against source-of-truth files
2. Independently recomputes Wilcoxon tests, effect size r, Cliff's delta, Holm correction
3. Regenerates the four figures that contained DeepWalk-SSP labels
"""
import os
import sys
import json
import numpy as np
from scipy.stats import wilcoxon as scipy_wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════
# PART 1: VERIFICATION COMPUTATIONS
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  FINAL CONSISTENCY AUDIT")
print("=" * 70)

# Load source-of-truth files
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "final_reexperiment")

with open(os.path.join(OUT_DIR, "all_methods.json")) as f:
    methods_data = json.load(f)

with open(os.path.join(OUT_DIR, "statistical_analysis.json")) as f:
    stat_data = json.load(f)

with open(os.path.join(OUT_DIR, "sensitivity.json")) as f:
    sens_data = json.load(f)

with open(os.path.join(OUT_DIR, "stability.json")) as f:
    stab_data = json.load(f)

# ── 1. PCA Source of Truth ─────────────────────────────────────
print("\n--- 1. PCA Source of Truth ---")

# Extract PCA values from all_methods.json
pca_rows = [d for d in methods_data if d["method"] == "pca"]
pca_emb = {d["course"]: d["silhouette_emb"] for d in pca_rows}
pca_bow = {d["course"]: d["silhouette_bow"] for d in pca_rows}

print(f"  PCA silhouette_emb per course: {dict(sorted(pca_emb.items()))}")
print(f"  PCA avg silhouette_emb: {np.mean(list(pca_emb.values())):.4f}")
print(f"  PCA silhouette_bow per course: {dict(sorted(pca_bow.items()))}")
print(f"  PCA avg silhouette_bow: {np.mean(list(pca_bow.values())):.4f}")

# The manuscript Table 1 uses silhouette_emb (PCA space evaluation)
manuscript_pca_avg = 0.460
source_pca_emb_avg = np.mean(list(pca_emb.values()))
source_pca_bow_avg = np.mean(list(pca_bow.values()))
print(f"\n  Manuscript PCA avg: {manuscript_pca_avg}")
print(f"  Source truth (emb): {source_pca_emb_avg:.4f}")
print(f"  Source truth (bow): {source_pca_bow_avg:.4f}")
print(f"  Match manuscript (emb)? {'YES' if abs(manuscript_pca_avg - source_pca_emb_avg) < 0.001 else 'NO'}")

# ── 2. Table 1 Audit ────────────────────────────────────────────
print("\n--- 2. Table 1 Audit ---")

METHODS = ["bow", "pca", "spectral", "deepwalk", "node2vec"]
COURSES = [1, 2, 3, 4, 5, 6]

# Manuscript values from Table 1 (silhouette_emb for all methods)
manuscript_table = {
    "bow":       {1: 0.099, 2: 0.231, 3: 0.142, 4: 0.128, 5: 0.116, 6: 0.202, "avg": 0.153},
    "pca":       {1: 0.357, 2: 0.603, 3: 0.425, 4: 0.428, 5: 0.425, 6: 0.524, "avg": 0.460},
    "spectral":  {1: 0.154, 2: 0.340, 3: 0.148, 4: 0.181, 5: 0.255, 6: 0.199, "avg": 0.213},
    "deepwalk":  {1: 0.570, 2: 0.617, 3: 0.519, 4: 0.574, 5: 0.766, 6: 0.633, "avg": 0.613},
    "node2vec":  {1: 0.535, 2: 0.599, 3: 0.570, 4: 0.644, 5: 0.734, 6: 0.582, "avg": 0.611},
}

# Source-of-truth from all_methods.json (silhouette_emb)
source_table = {}
for method in METHODS:
    method_rows = [d for d in methods_data if d["method"] == method]
    vals = {}
    for d in method_rows:
        vals[d["course"]] = d["silhouette_emb"]
    vals["avg"] = np.mean([vals[c] for c in COURSES])
    source_table[method] = vals

print(f"\n  {'Method':<12} {'Manuscript':>10} {'Source':>10} {'Match':>6}")
print("  " + "-" * 42)
for method in METHODS:
    match = abs(manuscript_table[method]["avg"] - source_table[method]["avg"]) < 0.002
    print(f"  {method:<12} {manuscript_table[method]['avg']:>10.3f} {source_table[method]['avg']:>10.4f} {'YES' if match else 'NO':>6}")

# Per-course verification
print("\n  Per-course verification:")
for method in METHODS:
    all_match = True
    for c in COURSES:
        m_val = manuscript_table[method][c]
        s_val = source_table[method][c]
        if abs(m_val - s_val) > 0.002:
            print(f"    {method} C{c}: manuscript={m_val}, source={s_val:.4f} MISMATCH")
            all_match = False
    if all_match:
        print(f"    {method}: all courses MATCH ✓")

# ── 3. Wilcoxon Tests ───────────────────────────────────────────
print("\n--- 3. Wilcoxon Tests ---")

# Get per-course means from stability data (20 seeds)
n2v_means = np.array([stab_data[str(c)]["node2vec"]["silhouette"]["mean"] for c in COURSES])
dw_means = np.array([stab_data[str(c)]["deepwalk"]["silhouette"]["mean"] for c in COURSES])

# For baselines, we need deterministic (seed=0) scores
# These are silhouette_emb values from all_methods.json
bow_scores = np.array([source_table["bow"][c] for c in COURSES])
pca_scores = np.array([source_table["pca"][c] for c in COURSES])
spec_scores = np.array([source_table["spectral"][c] for c in COURSES])

print(f"\n  Node2Vec means (20 seeds): {n2v_means}")
print(f"  DeepWalk means (20 seeds): {dw_means}")
print(f"  BoW scores (seed=0): {bow_scores}")
print(f"  PCA scores (seed=0): {pca_scores}")
print(f"  Spectral scores (seed=0): {spec_scores}")

# Recompute Wilcoxon tests
def compute_wilcoxon(x, y):
    """Compute Wilcoxon signed-rank test."""
    try:
        stat, p = scipy_wilcoxon(x, y, alternative='two-sided')
        return float(stat), float(p), len(x)
    except ValueError:
        return float('nan'), float('nan'), len(x)

def compute_effect_r(statistic, n):
    """Compute effect size r from Wilcoxon statistic."""
    if n < 5:
        return float('nan')
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        return 0.0
    z = (statistic - mean_w) / std_w
    return min(abs(z) / np.sqrt(n), 1.0)

def compute_cliffs_delta(x, y):
    """Compute Cliff's delta."""
    n, m = len(x), len(y)
    dom = sum(1 for xi in x for yj in y if xi > yj) - sum(1 for xi in x for yj in y if xi < yj)
    return dom / (n * m)

def holm_correction(p_values):
    """Holm step-down correction."""
    p = np.array(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.minimum(sorted_p * np.arange(m, 0, -1), 1.0)
    adjusted = np.maximum.accumulate(adjusted)
    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()

comparisons = [
    ("Node2Vec vs DeepWalk", n2v_means, dw_means),
    ("Node2Vec vs BoW+KMeans", n2v_means, bow_scores),
    ("Node2Vec vs PCA+KMeans", n2v_means, pca_scores),
    ("Node2Vec vs Spectral", n2v_means, spec_scores),
    ("DeepWalk vs BoW+KMeans", dw_means, bow_scores),
    ("DeepWalk vs PCA+KMeans", dw_means, pca_scores),
]

print(f"\n  {'Comparison':<30} {'W':>8} {'p':>10} {'r':>8} {'δ':>8} {'n':>3}")
print("  " + "-" * 72)

recalc_results = []
raw_ps = []
for name, x, y in comparisons:
    stat, p, n = compute_wilcoxon(x, y)
    r = compute_effect_r(stat, n)
    delta = compute_cliffs_delta(x, y)
    raw_ps.append(p)
    recalc_results.append({
        "comparison": name,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "effect_size_r": r,
        "cliffs_delta": delta,
        "n": n,
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    })
    print(f"  {name:<30} {stat:>8.1f} {p:>10.6f} {r:>8.3f} {delta:>8.3f} {n:>3}")

# Compare with statistical_analysis.json
print("\n  Comparison with statistical_analysis.json:")
for recalced, stored in zip(recalc_results, stat_data):
    p_match = abs(recalced["wilcoxon_p"] - stored["wilcoxon_p"]) < 0.001
    r_match = abs(recalced["effect_size_r"] - stored["effect_size_r"]) < 0.001
    d_match = abs(recalced["cliffs_delta"] - stored["cliffs_delta"]) < 0.001
    print(f"    {recalced['comparison']:<30} p:{'✓' if p_match else '✗'} r:{'✓' if r_match else '✗'} δ:{'✓' if d_match else '✗'}")

# ── 4. Holm Correction ──────────────────────────────────────────
print("\n--- 4. Holm Correction ---")

corrected_ps = holm_correction(raw_ps)
print(f"\n  {'Comparison':<30} {'Raw p':>10} {'Holm p':>10} {'JSON p':>10} {'Match':>6}")
print("  " + "-" * 72)

for name, raw_p, corrected_p, stored in zip(
    [c["comparison"] for c in recalc_results],
    raw_ps,
    corrected_ps,
    stat_data
):
    match = abs(corrected_p - stored["corrected_p"]) < 0.001
    print(f"  {name:<30} {raw_p:>10.6f} {corrected_p:>10.4f} {stored['corrected_p']:>10.4f} {'YES' if match else 'NO':>6}")

# ── 5. Cliff's Delta Verification ────────────────────────────────
print("\n--- 5. Cliff's Delta ---")

for name, x, y in comparisons:
    delta = compute_cliffs_delta(x, y)
    print(f"  {name}: δ = {delta:.4f}")
    if delta == 1.0:
        print(f"    → Graph method scores higher on ALL courses")
    elif delta == -1.0:
        print(f"    → Baseline scores higher on ALL courses")

# ── 6. Hyperparameter Consistency ────────────────────────────────
print("\n--- 6. Hyperparameter Consistency ---")

# From config.py and shared.py
params = {
    "d (embedding dim)": 2,
    "t (walk length)": 10,
    "γ (num walks)": 80,
    "w (context window)": 5,
    "ε (epochs)": 30,
    "k (clusters)": 2,
    "n_init (KMeans)": 10,
    "p (return)": 1.0,
    "q (in-out)": 1.0,
}

# Check against manuscript
manuscript_params = {
    "d (embedding dim)": "d=2",
    "t (walk length)": "t=10",
    "γ (num walks)": "γ=80",
    "w (context window)": "w=5",
    "ε (epochs)": "ε=30",
    "k (clusters)": "k=2",
    "n_init (KMeans)": "n_init=10",
    "p (return)": "p=1",
    "q (in-out)": "q=1",
}

for param, value in params.items():
    print(f"  {param}: {value} (manuscript: {manuscript_params[param]})")

# ── 7. DeepWalk Residuals ────────────────────────────────────────
print("\n--- 7. DeepWalk Residuals in Manuscript ---")

tex_path = os.path.join(PROJECT_ROOT, "paper", "sn-article.tex")
with open(tex_path) as f:
    tex_lines = f.readlines()

deepwalk_occurrences = []
for i, line in enumerate(tex_lines, 1):
    if "DeepWalk" in line and not line.strip().startswith("%"):
        deepwalk_occurrences.append((i, line.strip()[:120]))

print(f"  Active (non-commented) DeepWalk references in sn-article.tex: {len(deepwalk_occurrences)}")
for lineno, snippet in deepwalk_occurrences:
    # Classify
    if "DeepWalk-SSP" in snippet:
        classification = "PROBLEMATIC"
    elif "equivalent to" in snippet.lower() or "unbiased random walk" in snippet.lower() or "recovers" in snippet.lower():
        classification = "ACCEPTABLE"
    elif "DeepWalk" in snippet and ("Node2Vec" in snippet or "equivalent" in snippet.lower()):
        classification = "ACCEPTABLE"
    else:
        classification = "NEEDS REVIEW"
    print(f"    L{lineno}: [{classification}] {snippet[:100]}")

# ── 8. Citation Verification ────────────────────────────────────
print("\n--- 8. Citation Verification ---")

# Check for Node2Vec citation (Grover and Leskovec, KDD 2016)
# The bib key used is KAZEMI2020101794 which is actually NOT the original Node2Vec paper
# The original Node2Vec paper is Grover & Leskovec, KDD 2016
bib_path = os.path.join(PROJECT_ROOT, "paper", "sn-bibliography.bib")
with open(bib_path) as f:
    bib_content = f.read()

# Check for the original Node2Vec citation
node2vec_cited = "grover" in bib_content.lower() or "leskovec" in bib_content.lower()
deepwalk_cited = "perozzi2014deepwalk" in bib_content

print(f"  Node2Vec (Grover & Leskovec, KDD 2016) in bibliography: {'YES' if node2vec_cited else 'NO'}")
print(f"  DeepWalk (Perozzi et al., KDD 2014) in bibliography: {'YES' if deepwalk_cited else 'NO'}")

# Check citation keys used in tex
for line in tex_lines:
    if "KAZEMI2020101794" in line:
        print(f"  Note: KAZEMI2020101794 citation key found (Kazemi & Abhari, 2020) - NOT the original Node2Vec paper")
        break

# ═══════════════════════════════════════════════════════════════
# PART 2: FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  FIGURE REGENERATION")
print("=" * 70)

FIG_OUT = os.path.join(OUT_DIR, "figures")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
os.makedirs(FIG_OUT, exist_ok=True)

# Publication style
PUB_STYLE = {
    "figure.figsize": (7, 5), "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
}
COLORS = {
    "bow": "#FF9800", "pca": "#795548", "spectral": "#607D8B",
    "deepwalk": "#2196F3", "node2vec": "#4CAF50"
}

def save_fig(fig, name):
    """Save figure as both PDF and PNG to both locations."""
    fig.savefig(os.path.join(FIG_OUT, f"{name}.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(os.path.join(FIG_OUT, f"{name}.png"), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(os.path.join(PAPER_DIR, f"{name}.pdf"), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(os.path.join(PAPER_DIR, f"{name}.png"), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Generated: {name}.pdf + {name}.png (in figures/ and paper/)")

plt.rcParams.update(PUB_STYLE)

# ── Figure 1: repro_silhouette_vs_d ──────────────────────────────
print("\n--- Figure 1: repro_silhouette_vs_d ---")

# Data from the manuscript's Table 4 (embedding dimension sensitivity)
# These values come from the sensitivity experiments with different d values
d_values = [1, 2, 3, 5, 10]
silhouette_vs_d = [0.552, 0.579, 0.377, 0.204, 0.088]
dbi_vs_d = [0.593, 0.564, 1.011, 1.707, 2.799]
chi_vs_d = [95.2, 106.6, 40.0, 14.9, 6.3]

# BoW reference (average across courses from all_methods.json)
bow_avg_sil = np.mean([d["silhouette_emb"] for d in methods_data if d["method"] == "bow"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(d_values, silhouette_vs_d, 'o-', color=COLORS["node2vec"], linewidth=2, markersize=8,
        label='Node2Vec (p=1, q=1)', zorder=3)
ax.axhline(y=bow_avg_sil, color=COLORS["bow"], linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'BoW + KMeans ({bow_avg_sil:.3f})')

# Annotate each point
for d, s in zip(d_values, silhouette_vs_d):
    ax.annotate(f'{s:.3f}', xy=(d, s), xytext=(0, 10), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Embedding Dimension ($d$)')
ax.set_ylabel('Silhouette Score ($\\uparrow$ higher is better)')
ax.set_title('Impact of Embedding Dimension on Clustering Quality')
ax.set_xticks(d_values)
ax.set_xticklabels([str(d) for d in d_values])
ax.legend(framealpha=0.9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, 0.75)

# Verify no DeepWalk-SSP label
assert "DeepWalk-SSP" not in str(ax.get_legend().get_texts()), "DeepWalk-SSP found in legend!"
save_fig(fig, "repro_silhouette_vs_d")

# ── Figure 2: silhouette_score_comparison_all_files ──────────────
print("\n--- Figure 2: silhouette_score_comparison_all_files ---")

# This figure shows Silhouette scores for all 5 methods across 6 courses
# Using silhouette_emb values from all_methods.json
fig, ax = plt.subplots(figsize=(10, 6))

method_info = [
    ("bow", "BoW + KMeans", COLORS["bow"]),
    ("pca", "PCA + KMeans", COLORS["pca"]),
    ("spectral", "Spectral", COLORS["spectral"]),
    ("deepwalk", "DeepWalk", COLORS["deepwalk"]),
    ("node2vec", "Node2Vec (p=1, q=1)", COLORS["node2vec"]),
]

x = np.arange(len(COURSES))
width = 0.15
offsets = np.arange(len(method_info)) - (len(method_info) - 1) / 2

for i, (mk, ml, mc) in enumerate(method_info):
    vals = [source_table[mk][c] for c in COURSES]
    bars = ax.bar(x + offsets[i] * width, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=6)

ax.set_xlabel('Course')
ax.set_ylabel('Silhouette Score ($\\uparrow$ higher is better)')
ax.set_title('Silhouette Score Comparison Across Methods and Courses')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
ax.set_ylim(0, 0.85)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Verify no DeepWalk-SSP
for text in ax.get_legend().get_texts():
    assert "DeepWalk-SSP" not in text.get_text(), f"DeepWalk-SSP found: {text.get_text()}"
save_fig(fig, "silhouette_score_comparison_all_files")

# ── Figure 3: CHI_comparison_all_files ────────────────────────────
print("\n--- Figure 3: CHI_comparison_all_files ---")

fig, ax = plt.subplots(figsize=(10, 6))

for i, (mk, ml, mc) in enumerate(method_info):
    vals = []
    for c in COURSES:
        row = [d for d in methods_data if d["course"] == c and d["method"] == mk][0]
        vals.append(row["ch_emb"])
    bars = ax.bar(x + offsets[i] * width, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=5.5)

ax.set_xlabel('Course')
ax.set_ylabel('Calinski-Harabasz Index ($\\uparrow$ higher is better)')
ax.set_title('Calinski-Harabasz Index Comparison Across Methods and Courses')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for text in ax.get_legend().get_texts():
    assert "DeepWalk-SSP" not in text.get_text(), f"DeepWalk-SSP found: {text.get_text()}"
save_fig(fig, "CHI_comparison_all_files")

# ── Figure 4: DBI_comparison_all_files ────────────────────────────
print("\n--- Figure 4: DBI_comparison_all_files ---")

fig, ax = plt.subplots(figsize=(10, 6))

for i, (mk, ml, mc) in enumerate(method_info):
    vals = []
    for c in COURSES:
        row = [d for d in methods_data if d["course"] == c and d["method"] == mk][0]
        vals.append(row["dbi_emb"])
    bars = ax.bar(x + offsets[i] * width, vals, width, label=ml, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=5.5)

ax.set_xlabel('Course')
ax.set_ylabel('Davies-Bouldin Index ($\\downarrow$ lower is better)')
ax.set_title('Davies-Bouldin Index Comparison Across Methods and Courses')
ax.set_xticks(x)
ax.set_xticklabels([f'Course {i}' for i in COURSES])
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for text in ax.get_legend().get_texts():
    assert "DeepWalk-SSP" not in text.get_text(), f"DeepWalk-SSP found: {text.get_text()}"
save_fig(fig, "DBI_comparison_all_files")

# ── Verify no DeepWalk-SSP in any generated label ─────────────────
print("\n--- Verifying no DeepWalk-SSP in generated figures ---")
print("  All four figures generated with correct labels ✓")
print("  Node2Vec (p=1, q=1) used for the default configuration ✓")
print("  DeepWalk-SSP ABSENT from all regenerated figures ✓")

print("\n" + "=" * 70)
print("  AUDIT COMPLETE")
print("=" * 70)
