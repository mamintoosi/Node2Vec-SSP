import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(r"C:\git\mamintoosi\Node2Vec-SSP")
gs = json.loads((root / "results/grid_search/grid_search_complete.json").read_text())
df = pd.DataFrame(gs["results"])
P_VALUES = [0.5, 1.0, 2.0]
Q_VALUES = [0.5, 1.0, 2.0]
D_VALUES = [1, 2, 3, 5, 10]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 22, "axes.labelsize": 14})

fig, axes = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 5.5), sharey=True)
for idx, d in enumerate(D_VALUES):
    sub = df[df["d"] == d]
    pivot = sub.groupby(["p", "q"])["silhouette"].mean().unstack()
    vmin_d = pivot.values.min()
    vmax_d = pivot.values.max()
    axes[idx].imshow(
        pivot.values, cmap="YlOrRd", aspect="auto", vmin=vmin_d, vmax=vmax_d
    )
    axes[idx].set_xticks(range(len(Q_VALUES)))
    axes[idx].set_xticklabels([f"{q:.1f}" for q in Q_VALUES], fontsize=12)
    if idx == 0:
        axes[idx].set_yticks(range(len(P_VALUES)))
        axes[idx].set_yticklabels([f"{p:.1f}" for p in P_VALUES], fontsize=12)
        axes[idx].set_ylabel("p", fontsize=14)
    axes[idx].set_xlabel("q", fontsize=14)
    axes[idx].set_title(f"d={d}\n[{vmin_d:.3f}-{vmax_d:.3f}]", fontsize=22)
    for i, p in enumerate(P_VALUES):
        for j, q in enumerate(Q_VALUES):
            val = pivot.values[i, j]
            axes[idx].text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=16,
                color="white" if val > pivot.values.mean() else "black",
            )

fig.tight_layout()
out_dir = root / "results/grid_search/figures"
paper_dir = root / "paper"
for ext in ("pdf", "png"):
    path = out_dir / f"grid_heatmap_all_d_independent.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("wrote", path)
    if ext == "pdf":
        shutil.copy2(path, paper_dir / "grid_heatmap_all_d_independent.pdf")
        print("copied to paper/")
plt.close(fig)
