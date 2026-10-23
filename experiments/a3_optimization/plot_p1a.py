"""Problem 1(a): endpoint and 0.5/0.5 prediction-ensemble validation loss across the (x, y) grid.

Reads results/a3-mode-connectivity.json from the `measure` stage of p1_mode_connectivity.
Left: the same, but against the better of the two endpoints, to check that the ensemble beats both.
Middle: loss vs. branch length y for each prefix length x (endpoint mean with the two endpoints
shaded, ensemble dashed). Right: ensemble gain, mean endpoint loss minus ensemble loss.
Writes plots/p1a_endpoint_ensemble.png and prints the grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p1_mode_connectivity import (
    BRANCH_STEPS,
    BRANCHES,
    PREFIX_STEPS,
    RESULTS_PATH,
)
from train import training_run_name

PLOT_DIR = Path(__file__).resolve().parent / "plots"


def savefig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path.resolve())
    return path


def load_grid(path=RESULTS_PATH):
    """{(x, y): dict(a, b, endpoint, ensemble, avg, barrier, alphas, curve)} from the measure-stage rows.

    avg is the loss of the 0.5/0.5 weight average; barrier = avg - endpoint; curve is the
    interpolation loss at each alpha.
    """
    prefix_of = {training_run_name(run): x for x, runs in BRANCHES.items() for run in runs}
    grid = {}
    for row in json.loads(Path(path).read_text()):
        x = prefix_of[row["run_a"]]
        grid[(x, row["step_a"] - x)] = cell(row)
    return grid


def cell(row):
    """Endpoint, weight-average, barrier, and ensemble losses from one measure-stage row."""
    alphas = row["alphas"]
    a = row["interpolation"][alphas.index(0.0)]
    b = row["interpolation"][alphas.index(1.0)]
    avg = row["interpolation"][alphas.index(0.5)]
    return dict(a=a, b=b, endpoint=(a + b) / 2, ensemble=row["ensemble"], avg=avg,
                barrier=avg - (a + b) / 2, alphas=alphas, curve=row["interpolation"])


def matrix(grid, key):
    return np.array([[grid.get((x, y), {}).get(key, np.nan) for y in BRANCH_STEPS] for x in PREFIX_STEPS])


def draw_curves(ax, grid, xs, colors, legend=None, endpoint="mean"):
    """Loss vs. y per prefix x: endpoint (solid) and ensemble (dashed).

    endpoint="mean" plots the mean of the two branches with both endpoints shaded;
    endpoint="best" plots the better (lower-loss) of the two branches.
    """
    for x in xs:
        ys = [y for y in BRANCH_STEPS if (x, y) in grid]
        cells = [grid[(x, y)] for y in ys]
        if endpoint == "mean":
            ax.fill_between(ys, [min(c["a"], c["b"]) for c in cells], [max(c["a"], c["b"]) for c in cells],
                            color=colors[x], alpha=0.2, lw=0)
            values = [c["endpoint"] for c in cells]
        else:
            values = [min(c["a"], c["b"]) for c in cells]
        ax.plot(ys, values, "o-", color=colors[x], label=f"x={x}")
        ax.plot(ys, [c["ensemble"] for c in cells], "s--", color=colors[x], mfc="none")
    ax.set(xscale="log", xlabel="branch updates y", ylabel="validation loss")
    ax.set_xticks(BRANCH_STEPS, [str(y) for y in BRANCH_STEPS])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    if legend is not None:
        label = "endpoint (mean; band = both)" if endpoint == "mean" else "best endpoint (min of 2)"
        ax.plot([], [], "k-", label=label)
        ax.plot([], [], "ks--", mfc="none", label="0.5/0.5 ensemble")
        ax.legend(fontsize=8, **legend)


def main():
    grid = load_grid()
    colors = dict(zip(PREFIX_STEPS, plt.cm.viridis(np.linspace(0, 0.9, len(PREFIX_STEPS)))))
    fig, (bx, ax, hx) = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))

    draw_curves(bx, grid, PREFIX_STEPS, colors, legend=dict(ncol=2, loc="upper right"), endpoint="best")
    bx.set_title("Best endpoint vs. ensemble loss")

    draw_curves(ax, grid, PREFIX_STEPS, colors, legend=dict(ncol=2, loc="upper right"))
    ax.set_title("Endpoint vs. ensemble loss")

    gain = matrix(grid, "endpoint") - matrix(grid, "ensemble")
    im = hx.imshow(gain, cmap="magma_r", aspect="auto")
    for i, j in np.ndindex(gain.shape):
        if np.isfinite(gain[i, j]):
            hx.text(j, i, f"{gain[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if gain[i, j] > np.nanmean(gain) else "black")
    hx.set_xticks(range(len(BRANCH_STEPS)), [str(y) for y in BRANCH_STEPS])
    hx.set_yticks(range(len(PREFIX_STEPS)), [str(x) for x in PREFIX_STEPS])
    hx.set(xlabel="branch updates y", ylabel="shared prefix updates x",
           title="Ensemble gain: mean endpoint − ensemble loss")
    fig.colorbar(im, ax=hx, label="loss reduction")
    fig.suptitle("P1(a): d8, 614M-token run, constant LR after warmup; branches differ only in data order")
    fig.tight_layout()
    savefig(fig, "p1a_endpoint_ensemble.png")

    print(f"{'x':>5} {'y':>4} {'L(a)':>7} {'L(b)':>7} {'ens':>7} {'gain':>7} {'vs_best':>7}")
    for (x, y), c in sorted(grid.items()):
        print(f"{x:>5} {y:>4} {c['a']:7.4f} {c['b']:7.4f} {c['ensemble']:7.4f} {c['endpoint'] - c['ensemble']:7.4f} {min(c['a'], c['b']) - c['ensemble']:7.4f}")


if __name__ == "__main__":
    main()
