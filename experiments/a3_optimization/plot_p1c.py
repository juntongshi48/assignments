"""Problem 1(c): full interpolation curves L((1 - alpha) theta_1 + alpha theta_2) for representative cells.

Reads results/a3-mode-connectivity.json (alpha in {0, 0.25, 0.5, 0.75, 1}).
One panel per cell in CELLS: interpolation curve, the straight line between the endpoint losses,
and the 0.5/0.5 prediction-ensemble loss. Last panel: every curve minus its endpoint line, so
the alpha = 0.5 value is the midpoint barrier B.
Writes plots/p1c_interpolation.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.plot_p1a import load_grid, savefig
from experiments.a3_optimization.plot_p1b import THRESHOLD

# (x, y): long prefix + short branch, short prefix + short branch, at the threshold, far past it.
CELLS = ((1000, 25), (100, 50), (50, 250), (0, 500))


def chord(cell):
    alphas = np.array(cell["alphas"])
    return (1 - alphas) * cell["a"] + alphas * cell["b"]


def main():
    grid = load_grid()
    colors = dict(zip(CELLS, plt.cm.plasma(np.linspace(0, 0.85, len(CELLS)))))
    fig, axes = plt.subplots(1, len(CELLS) + 1, figsize=(4.2 * (len(CELLS) + 1), 4.2))

    for ax, (x, y) in zip(axes, CELLS):
        cell, color = grid[(x, y)], colors[(x, y)]
        alphas = cell["alphas"]
        ax.plot(alphas, cell["curve"], "o-", color=color, label="weight interpolation")
        ax.plot(alphas, chord(cell), ":", color="0.4", label="endpoint line")
        ax.axhline(cell["ensemble"], color=color, ls="--", lw=1, label="0.5/0.5 ensemble")
        ax.annotate("", xy=(0.5, cell["avg"]), xytext=(0.5, cell["endpoint"]),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1))
        ax.annotate(f"B = {cell['barrier']:+.3f}", ((0.5, (cell["avg"] + cell["endpoint"]) / 2)),
                    xytext=(6, 12 if abs(cell["barrier"]) < 0.01 else 0), textcoords="offset points",
                    fontsize=8, va="center")
        ax.set(xlabel="α", ylabel="validation loss", title=f"x = {x}, y = {y}  (T = {x + y})")
        ax.set_xticks(alphas)
        ax.margins(y=0.15)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper center")

    ox = axes[-1]
    for (x, y), color in colors.items():
        cell = grid[(x, y)]
        ox.plot(cell["alphas"], np.array(cell["curve"]) - chord(cell), "o-", color=color, label=f"({x}, {y})")
        ox.plot(0.5, cell["ensemble"] - cell["endpoint"], "s", mfc="none", color=color, ms=8)
    ox.axhline(0, color="k", lw=0.8)
    ox.axhline(THRESHOLD, color="r", ls="--", lw=1, label=f"B = {THRESHOLD}")
    ox.plot([], [], "ks", mfc="none", label="ensemble (at α = 0.5)")
    ox.set(xlabel="α", ylabel="L(α) − endpoint line", title="All cells, relative to endpoint line")
    ox.set_xticks(grid[CELLS[0]]["alphas"])
    ox.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ox.grid(alpha=0.3)

    fig.suptitle("P1(c): loss along the line between two branches; d8, constant LR after warmup")
    fig.tight_layout()
    savefig(fig, "p1c_interpolation.png")


if __name__ == "__main__":
    main()
