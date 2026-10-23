"""Problem 1(d): barrier and ensemble-gain heatmaps for the LR / batch-size ablation vs. the P1 base.

Reads results/a3-mode-connectivity-ablation-<variant>.json (p1d_ablation measure) and the base grid
from results/a3-mode-connectivity.json, restricted to the ablation's prefixes (x <= 500).
Top row: midpoint barrier B(x, y), one shared diverging color scale; cells with B > THRESHOLD hatched.
Bottom row: ensemble gain (mean endpoint - ensemble loss), one shared scale.
Writes plots/p1d_ablation.png and prints per-variant summaries.
"""

from __future__ import annotations

import json

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from experiments.a3_optimization.p1_mode_connectivity import BRANCH_STEPS
from experiments.a3_optimization.p1d_ablation import PREFIX_STEPS, VARIANTS, results_path
from experiments.a3_optimization.plot_p1a import cell, load_grid, savefig
from experiments.a3_optimization.plot_p1b import THRESHOLD, outline_region

TITLES = {"base": "base: LR 3e-3, batch 64", "lr1.5e-3": "LR 1.5e-3 (×0.5)", "lr6e-3": "LR 6e-3 (×2)",
          "bs128": "batch 128 (×2), LR 3e-3"}


def load_all():
    base = load_grid()
    grids = {"base": {k: v for k, v in base.items() if k[0] in PREFIX_STEPS}}
    for variant in VARIANTS:
        path = results_path(variant)
        if path.exists():
            grids[variant] = {(row["x"], row["y"]): cell(row) for row in json.loads(path.read_text())}
    return grids


def matrix(grid, key):
    return np.array([[key(grid[(x, y)]) if (x, y) in grid else np.nan for y in BRANCH_STEPS]
                     for x in PREFIX_STEPS])


def heatmap(ax, values, norm, cmap, fmt, fails=None):
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for i, j in np.ndindex(values.shape):
        if not np.isfinite(values[i, j]):
            continue
        failed = fails is not None and fails[i, j]
        if failed:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, color="0.25"))
        ax.text(j, i, fmt(values[i, j]), ha="center", va="center", fontsize=7,
                fontweight="bold" if failed else "normal",
                color="white" if abs(norm(values[i, j]) - 0.5) > 0.35 else "black")
    if fails is not None:
        outline_region(ax, fails, color="k", lw=1.5)
    ax.set_xticks(range(len(BRANCH_STEPS)), [str(y) for y in BRANCH_STEPS])
    ax.set_yticks(range(len(PREFIX_STEPS)), [str(x) for x in PREFIX_STEPS])
    return im


def main():
    grids = load_all()
    names = list(grids)
    barrier = {n: matrix(grids[n], lambda c: c["barrier"]) for n in names}
    gain = {n: matrix(grids[n], lambda c: c["endpoint"] - c["ensemble"]) for n in names}

    lo = min(np.nanmin(v) for v in barrier.values())
    hi = max(np.nanmax(v) for v in barrier.values())
    bnorm = TwoSlopeNorm(vmin=min(lo, -THRESHOLD), vcenter=0.0, vmax=max(hi, THRESHOLD))
    gnorm = mpl.colors.Normalize(vmin=min(np.nanmin(v) for v in gain.values()),
                                 vmax=max(np.nanmax(v) for v in gain.values()))

    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names) + 1.5, 9), squeeze=False,
                             layout="constrained")
    for j, n in enumerate(names):
        im_b = heatmap(axes[0, j], barrier[n], bnorm, "RdBu_r", lambda v: f"{v:+.3f}",
                       fails=barrier[n] > THRESHOLD)
        im_g = heatmap(axes[1, j], gain[n], gnorm, "magma_r", lambda v: f"{v:.3f}")
        passed = int(np.nansum(barrier[n] <= THRESHOLD))
        axes[0, j].set_title(f"{TITLES[n]}\nB ≤ {THRESHOLD}: {passed}/{np.isfinite(barrier[n]).sum()} cells",
                             fontsize=10)
        axes[1, j].set_title(TITLES[n], fontsize=10)
        for i in range(2):
            axes[i, j].set_xlabel("branch updates y")
        axes[0, j].set_ylabel("shared prefix updates x" if j == 0 else "")
        axes[1, j].set_ylabel("shared prefix updates x" if j == 0 else "")
    cb = fig.colorbar(im_b, ax=axes[0, :].tolist(), label="midpoint barrier B (hatched: B > 0.01)", shrink=0.9)
    cb.set_ticks([t for t in (-0.1, -0.05, 0, 0.25, 0.5, 0.75, 1.0) if bnorm.vmin <= t <= bnorm.vmax])
    fig.colorbar(im_g, ax=axes[1, :].tolist(), label="ensemble gain: endpoint − ensemble", shrink=0.9)
    fig.suptitle("P1(d): which training choices keep branches in one basin? (constant LR after warmup; "
                 "one knob changed per column)")
    savefig(fig, "p1d_ablation.png")

    print(f"{'variant':>9} {'pass':>6} {'B min':>7} {'B max':>7} {'gain mean':>9} {'L(500,500) endpoint':>20}")
    for n in names:
        b, g = barrier[n], gain[n]
        end = grids[n].get((500, 500), {}).get("endpoint", float("nan"))
        print(f"{n:>9} {int(np.nansum(b <= THRESHOLD)):>3}/{int(np.isfinite(b).sum()):<2} {np.nanmin(b):+7.3f} "
              f"{np.nanmax(b):+7.3f} {np.nanmean(g):9.4f} {end:20.4f}")


if __name__ == "__main__":
    main()
