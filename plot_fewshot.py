"""Plot few-shot curves (condition D), with in-language accuracy (B) as dashed reference lines."""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

KS = [0, 8, 32, 128]
TARGETS = {"es-ES": "Spanish", "ja-JP": "Japanese"}
MODELS = {"minilm": ("MiniLM", "#2a78d6"),
          "mpnet": ("mpnet", "#eb6834"),
          "gemma": ("EmbeddingGemma", "#1baf7a")}
INK, SECONDARY, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DASH = (0, (4, 3))

# Few-shot test accuracy: {(model, target): {k: accuracy}}
fewshot, max_std = {}, 0.0
with open("results_fewshot.csv") as f:
    for row in csv.DictReader(f):
        fewshot.setdefault((row["model"], row["target"]), {})[int(row["k"])] = float(row["acc_mean"])
        max_std = max(max_std, float(row["acc_std"]))

# In-language test accuracy (B): rows of results_lr.txt where train and test language match
in_language = {}
pattern = re.compile(r"^(\w+)\s+(\S+) -> (\S+) test: accuracy ([\d.]+)")
with open("results_lr.txt") as f:
    for line in f:
        m = pattern.match(line)
        if m and m.group(2) == m.group(3):
            in_language[(m.group(1), m.group(2))] = float(m.group(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.2, wspace=0.08)
x = list(range(len(KS)))

for ax, (target, name) in zip(axes, TARGETS.items()):
    ax.set_facecolor(SURFACE)
    for model, (label, color) in MODELS.items():
        ys = [fewshot[(model, target)][k] for k in KS]
        ax.axhline(in_language[(model, target)], color=color, lw=1.5, ls=DASH, zorder=2)
        ax.plot(x, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    ax.set_title(name, color=INK, loc="left", fontsize=12, pad=8)
    ax.set_xticks(x, ["0\n(zero-shot)", "8", "32", "128"])
    ax.set_xlim(-0.25, len(KS) - 0.75)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)

axes[0].set_ylim(0.70, 0.90)
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("Test accuracy", color=SECONDARY, fontsize=10)
fig.supxlabel("Target-language examples per intent added to English training data (k)", color=SECONDARY, fontsize=10, y=0.04)

fig.text(0.08, 0.95, "Few-shot transfer: English + k target-language examples per intent",
         color=INK, fontsize=13, fontweight="bold")
fig.text(0.08, 0.895, f"Test accuracy, mean of 3 random draws per k (standard deviation ≤ {100 * max_std:.1f} points)",
         color=SECONDARY, fontsize=10)

handles = [Line2D([], [], color=color, lw=2, marker="o", ms=7, mew=0) for _, color in MODELS.values()]
labels = [label for label, _ in MODELS.values()]
handles.append(Line2D([], [], color=MUTED, lw=1.5, ls=DASH))
labels.append("Dashed: trained only on that language (B)")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 0.875), ncol=4, frameon=False,
           labelcolor=INK, fontsize=10, handlelength=2.2, columnspacing=1.6)

fig.savefig("fewshot_curves.png", dpi=200, facecolor=SURFACE)
print("Saved fewshot_curves.png")