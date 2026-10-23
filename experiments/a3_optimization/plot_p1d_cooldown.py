"""Problem 1(d), fixed budget T = 1000: does a final LR cooldown change averaging and ensembling?

Reads results/a3-mode-connectivity-cooldown.json from p1d_cooldown's measure stage.
(a) midpoint barrier B at T vs. branch point x, constant LR vs. cooldown (750 -> 1000).
(b) ensemble gain (mean endpoint - ensemble loss) at T vs. x.
(c) losses at T vs. x: endpoint mean, 0.5/0.5 weight average, ensemble.
(d) barrier during the cooldown (steps 800, 900, 1000) for each x.
Writes plots/p1d_cooldown.png and prints the table.
"""

from __future__ import annotations

import json

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p1d_cooldown import (
    BRANCH_POINTS,
    COOLDOWN_CHECKPOINTS,
    COOLDOWN_START,
    RESULTS_PATH,
    T,
)
from experiments.a3_optimization.plot_p1a import cell, savefig
from experiments.a3_optimization.plot_p1b import THRESHOLD

STYLE = dict(constant=dict(ls="-", marker="o", label="constant LR"),
             cooldown=dict(ls="--", marker="s", mfc="none", label=f"cooldown {COOLDOWN_START}→{T}"))


def load(path=RESULTS_PATH):
    """{(variant, x, step): cell}"""
    return {(row["variant"], row["x"], row["step"]): cell(row) for row in json.loads(path.read_text())}


def barrier_axis(ax):
    ax.set_yscale("symlog", linthresh=0.1, linscale=2)
    ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 0.3, 1.0])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


def series(cells, variant, key, step=T):
    xs = [x for x in BRANCH_POINTS if (variant, x, step) in cells]
    return xs, [key(cells[(variant, x, step)]) for x in xs]


def main():
    cells = load()
    fig, ((bx, gx), (lx, dx)) = plt.subplots(2, 2, figsize=(13, 9))

    for variant, style in STYLE.items():
        bx.plot(*series(cells, variant, lambda c: c["barrier"]), color="k", **style)
        gx.plot(*series(cells, variant, lambda c: c["endpoint"] - c["ensemble"]), color="k", **style)
    bx.axhline(THRESHOLD, color="r", ls=":", lw=1, label=f"B = {THRESHOLD}")
    bx.axhline(0, color="0.5", lw=0.8)
    bx.set(xlabel="branch point x (y = T − x)", ylabel="midpoint barrier B (symlog, linear within ±0.1)",
           title=f"Barrier at T = {T}")
    barrier_axis(bx)
    gx.set(xlabel="branch point x (y = T − x)", ylabel="mean endpoint − ensemble loss",
           title=f"Ensemble gain at T = {T}")

    loss_colors = dict(endpoint="C0", avg="C3", ensemble="C2")
    loss_labels = dict(endpoint="endpoint (mean of 2)", avg="weight average", ensemble="ensemble")
    for variant, style in STYLE.items():
        for key, color in loss_colors.items():
            xs, ys = series(cells, variant, lambda c: c[key])
            lx.plot(xs, ys, color=color, ls=style["ls"], marker=style["marker"],
                    mfc=style.get("mfc", color))
    for key, color in loss_colors.items():
        lx.plot([], [], color=color, label=loss_labels[key])
    for style in STYLE.values():
        lx.plot([], [], color="k", ls=style["ls"], marker=style["marker"], mfc=style.get("mfc", "k"),
                label=style["label"])
    lx.set(xlabel="branch point x (y = T − x)", ylabel=f"validation loss at T = {T}",
           title="Losses at the end of the budget")
    lo = min(min(series(cells, v, lambda c: c[k])[1]) for v in STYLE for k in loss_colors)
    lx.set_ylim(lo - 0.03, lo + 0.45)  # x = 0 constant-LR weight average is off the top
    lx.legend(fontsize=8, ncol=2)

    colors = dict(zip(BRANCH_POINTS, plt.cm.viridis(np.linspace(0, 0.9, len(BRANCH_POINTS)))))
    for x in BRANCH_POINTS:
        steps = [s for s in COOLDOWN_CHECKPOINTS if ("cooldown", x, s) in cells]
        dx.plot(steps, [cells[("cooldown", x, s)]["barrier"] for s in steps], "s--", mfc="none",
                color=colors[x], label=f"x={x}")
        if ("constant", x, T) in cells:
            dx.plot([T], [cells[("constant", x, T)]["barrier"]], "o", color=colors[x])
    dx.plot([], [], "ko", label=f"constant LR at {T}")
    dx.axhline(THRESHOLD, color="r", ls=":", lw=1)
    dx.axhline(0, color="0.5", lw=0.8)
    dx.set(xlabel="update", ylabel="midpoint barrier B (symlog, linear within ±0.1)",
           title="Barrier during the cooldown", xticks=COOLDOWN_CHECKPOINTS)
    barrier_axis(dx)
    dx.legend(fontsize=8, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.6))

    for ax in (bx, gx, dx):
        if ax is not dx:
            ax.legend(fontsize=8)
    for ax in (bx, gx, lx, dx):
        ax.grid(alpha=0.3)
    fig.suptitle(f"P1(d): fixed budget T = {T}; constant LR vs. linear cooldown over the last "
                 f"{T - COOLDOWN_START} updates (same data order)")
    fig.tight_layout()
    savefig(fig, "p1d_cooldown.png")

    print(f"{'variant':>9} {'x':>4} {'step':>5} {'L(a)':>7} {'L(b)':>7} {'avg':>7} {'ens':>7} {'B':>8} {'gain':>7}")
    for (variant, x, step), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        print(f"{variant:>9} {x:>4} {step:>5} {c['a']:7.4f} {c['b']:7.4f} {c['avg']:7.4f} {c['ensemble']:7.4f} "
              f"{c['barrier']:+8.4f} {c['endpoint'] - c['ensemble']:7.4f}")


if __name__ == "__main__":
    main()
