"""Problem 1(a), matched training budget: does branching early or late give the better ensemble?

Cells with the same total step T = x + y have seen the same number of updates at the same constant
LR, so their endpoint losses are comparable; only the branch point x differs.
Left: endpoint and ensemble loss vs. total steps T for every prefix x.
Right: for each T shared by two cells, the ensemble gain (mean endpoint - ensemble loss) of the
earlier and the later branch point.
Writes plots/p1a_matched_budget.png and prints the matched pairs.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p1_mode_connectivity import PREFIX_STEPS
from experiments.a3_optimization.plot_p1a import load_grid, savefig


def matched_budgets(grid):
    """{T: [(x, y, cell), ...]} for every total step T reached by at least two cells, sorted by x."""
    by_total = defaultdict(list)
    for (x, y), cell in grid.items():
        by_total[x + y].append((x, y, cell))
    return {T: sorted(cells, key=lambda c: c[0]) for T, cells in sorted(by_total.items()) if len(cells) > 1}


def main():
    grid = load_grid()
    colors = dict(zip(PREFIX_STEPS, plt.cm.viridis(np.linspace(0, 0.9, len(PREFIX_STEPS)))))
    fig, (ax, mx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))

    for x in PREFIX_STEPS:
        cells = sorted((x + y, c) for (gx, y), c in grid.items() if gx == x)
        totals = [T for T, _ in cells]
        ax.plot(totals, [c["endpoint"] for _, c in cells], "o-", color=colors[x], label=f"x={x}")
        ax.plot(totals, [c["ensemble"] for _, c in cells], "s--", color=colors[x], mfc="none")
    ax.plot([], [], "k-", label="endpoint (mean of 2)")
    ax.plot([], [], "ks--", mfc="none", label="0.5/0.5 ensemble")
    ax.set(xscale="log", xlabel="total updates T = x + y", ylabel="validation loss",
           title="Loss vs. total training steps")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    groups = matched_budgets(grid)
    print(f"{'T':>5} {'early (x,y)':>12} {'late (x,y)':>12} {'d_endpt':>8} {'d_ens':>8} {'gain_e':>7} {'gain_l':>7}")
    for i, (T, cells) in enumerate(groups.items()):
        (xe, ye, early), (xl, yl, late) = cells[0], cells[-1]
        gains = [early["endpoint"] - early["ensemble"], late["endpoint"] - late["ensemble"]]
        mx.plot([i, i], gains, color="0.6", lw=1.5, zorder=1)
        for (x, y, _), gain, offset in zip((cells[0], cells[-1]), gains, (-0.12, 0.12)):
            mx.scatter(i, gain, s=60, color=colors[x], edgecolor="k", lw=0.5, zorder=2)
            mx.annotate(f"({x},{y})", (i, gain), xytext=(0, 9 if gain == max(gains) else -13),
                        textcoords="offset points", ha="center", fontsize=7)
        print(f"{T:>5} {f'({xe},{ye})':>12} {f'({xl},{yl})':>12} {early['endpoint'] - late['endpoint']:8.4f} "
              f"{early['ensemble'] - late['ensemble']:8.4f} {gains[0]:7.3f} {gains[1]:7.3f}")
    mx.set_xticks(range(len(groups)), [str(T) for T in groups])
    mx.set(xlabel="total updates T (cells labeled (x, y))", ylabel="ensemble gain: endpoint − ensemble",
           title="Same budget, different branch point")
    for x in PREFIX_STEPS:
        mx.scatter([], [], s=40, color=colors[x], edgecolor="k", lw=0.5, label=f"x={x}")
    mx.legend(fontsize=8, ncol=3, loc="lower left")
    mx.grid(alpha=0.3)
    mx.margins(y=0.15)

    fig.suptitle("P1(a) at matched budgets: d8, constant LR after warmup; branches differ only in data order")
    fig.tight_layout()
    savefig(fig, "p1a_matched_budget.png")


if __name__ == "__main__":
    main()
