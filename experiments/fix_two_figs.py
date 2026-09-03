"""Generate the two missing figures that failed due to a key mismatch."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results", "reproduced")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Publicatio-style params
PS = {'font.size':11, 'axes.titlesize':13, 'axes.labelsize':12,
      'figure.dpi':150, 'savefig.dpi':300, 'savefig.bbox':'tight'}

def sf(fig, name):
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    print(f"  [fig {name}]")
    plt.close(fig)

# ── Load data ──
with open(os.path.join(RES, "graph_stats.json")) as f:
    gs = json.load(f)
with open(os.path.join(RES, "runtime.json")) as f:
    rt = json.load(f)

# ── Figure 1: Graph density ──
plt.rcParams.update(PS)
cs = [str(i) for i in range(1,7)]
x = np.arange(6); w=0.35
fig, ax = plt.subplots(figsize=(8,5))
# Keys in the JSON are "d" for density
ax.bar(x-w/2, [gs["old"][c]["d"] for c in cs], w,
       label='Original', color='#E53935', alpha=0.8)
ax.bar(x+w/2, [gs["new"][c]["d"] for c in cs], w,
       label='Revised (constant removed)', color='#43A047', alpha=0.8)
for iv, (o, n) in enumerate(zip([gs["old"][c]["d"] for c in cs],
                                 [gs["new"][c]["d"] for c in cs])):
    ax.annotate(f'{o:.2f}', xy=(iv-w/2, o), xytext=(0,3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.annotate(f'{n:.2f}', xy=(iv+w/2, n), xytext=(0,3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Course'); ax.set_ylabel('Graph Density')
ax.set_title('Effect of Removing Constant Courses on Graph Density')
ax.set_xticks(x); ax.set_xticklabels([f'Course {i}' for i in range(1,7)])
ax.legend(framealpha=0.9); ax.set_ylim(0, 1.15)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
sf(fig, 'graph_density_comparison')

# ── Figure 2: Runtime comparison ──
plt.rcParams.update(PS)
cs_rt = sorted(set(d["course"] for d in rt))
x = np.arange(len(cs_rt))
methods = sorted(set(d["method"] for d in rt))
nw = len(methods); ww = 0.15
clr = plt.cm.Set2(np.linspace(0, 1, nw))
fig, ax = plt.subplots(figsize=(9,5))
for i, ml in enumerate(methods):
    vals = [[d["tt"] for d in rt if d["course"]==c and d["method"]==ml][0] for c in cs_rt]
    ax.bar(x + (i - nw/2 + 0.5)*ww, vals, ww, label=ml, color=clr[i], edgecolor='white')
ax.set_xlabel('Course'); ax.set_ylabel('Time (seconds)')
ax.set_xticks(x); ax.set_xticklabels([f'Course {c}' for c in cs_rt])
ax.legend(framealpha=0.9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.set_title('Runtime Comparison')
sf(fig, 'runtime_comparison')

print("\nDone — both missing figures generated.")
