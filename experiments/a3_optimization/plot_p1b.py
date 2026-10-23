"""Problem 1(b): weight averaging and the midpoint barrier B(x, y) across the (x, y) grid.

B = L((theta_1 + theta_2) / 2) - (L(theta_1) + L(theta_2)) / 2, read from the alpha = 0.5 point of
results/a3-mode-connectivity.json. Weight averaging "works" for a cell when B <= THRESHOLD.
Left: B heatmap; cells with B > THRESHOLD are hatched and outlined.
Right: B vs. branch length y for each prefix length x.
Writes plots/p1b_barrier.png and plots/p1b_table.md, and prints the table.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from experiments.a3_optimization.p1_mode_connectivity import BRANCH_STEPS, PREFIX_STEPS
from experiments.a3_optimization.plot_p1a import PLOT_DIR, load_grid, matrix, savefig

THRESHOLD = 0.01


def outline_region(ax, mask, **kwargs):
    """Draw the boundary between True and False cells of a (rows, cols) mask in imshow coordinates."""
    rows, cols = mask.shape
    for i, j in np.ndindex(mask.shape):
        if not mask[i, j]:
            continue
        if i == 0 or not mask[i - 1, j]:
            ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], **kwargs)
        if i == rows - 1 or not mask[i + 1, j]:
            ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], **kwargs)
        if j == 0 or not mask[i, j - 1]:
            ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], **kwargs)
        if j == cols - 1 or not mask[i, j + 1]:
            ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], **kwargs)


def table(grid):
    lines = ["| x | y | L(θ1) | L(θ2) | L(θavg) | B(x, y) | B ≤ 0.01 |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for (x, y), c in sorted(grid.items()):
        lines.append(f"| {x} | {y} | {c['a']:.4f} | {c['b']:.4f} | {c['avg']:.4f} | {c['barrier']:+.4f} | "
                     f"{'yes' if c['barrier'] <= THRESHOLD else '**no**'} |")
    return "\n".join(lines) + "\n"


def main():
    grid = load_grid()
    barrier = matrix(grid, "barrier")
    fails = barrier > THRESHOLD
    colors = dict(zip(PREFIX_STEPS, plt.cm.viridis(np.linspace(0, 0.9, len(PREFIX_STEPS)))))
    fig, (hx, ax) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.1, 1]))

    norm = TwoSlopeNorm(vmin=min(np.nanmin(barrier), -THRESHOLD), vcenter=0.0,
                        vmax=max(np.nanmax(barrier), THRESHOLD))
    im = hx.imshow(barrier, cmap="RdBu_r", norm=norm, aspect="auto")
    for i, j in np.ndindex(barrier.shape):
        if fails[i, j]:
            hx.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, color="0.25"))
        hx.text(j, i, f"{barrier[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                fontweight="bold" if fails[i, j] else "normal",
                color="white" if abs(norm(barrier[i, j]) - 0.5) > 0.35 else "black")
    outline_region(hx, fails, color="k", lw=2)
    hx.set_xticks(range(len(BRANCH_STEPS)), [str(y) for y in BRANCH_STEPS])
    hx.set_yticks(range(len(PREFIX_STEPS)), [str(x) for x in PREFIX_STEPS])
    hx.set(xlabel="branch updates y", ylabel="shared prefix updates x",
           title=f"Midpoint barrier B(x, y); unhatched cells: B ≤ {THRESHOLD}")
    hx.legend(handles=[Patch(facecolor="none", edgecolor="0.25", hatch="///", label=f"B > {THRESHOLD}: averaging fails"),
                       Line2D([], [], color="k", lw=2, label=f"boundary B = {THRESHOLD}")],
              fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.colorbar(im, ax=hx, label="B = L(θavg) − mean endpoint loss")

    for x in PREFIX_STEPS:
        ys = [y for y in BRANCH_STEPS if (x, y) in grid]
        ax.plot(ys, [grid[(x, y)]["barrier"] for y in ys], "o-", color=colors[x], label=f"x={x}")
    ax.axhline(THRESHOLD, color="r", ls="--", lw=1, label=f"B = {THRESHOLD}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xscale="log", xlabel="branch updates y", ylabel="midpoint barrier B",
           title="Barrier vs. branch length")
    ax.set_xticks(BRANCH_STEPS, [str(y) for y in BRANCH_STEPS])
    ax.minorticks_off()
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("P1(b): 0.5/0.5 weight averaging of two branches; d8, constant LR after warmup")
    fig.tight_layout()
    savefig(fig, "p1b_barrier.png")

    text = table(grid)
    (PLOT_DIR / "p1b_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
