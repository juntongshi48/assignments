"""Problem 3.1(c) follow-up: loss-vs-LR curves for RMSProp and Adam on the 2D noisy quadratic at a given
noise scale (default sigma = 300), on one dense LR grid shared by every batch, plus the near-optimal LR band
(loss within 5% of the best) vs. batch.
Writes plots/p31c_curves_sigma{sigma}.png.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.nqm import simulate
from experiments.a2.plot_p1a import savefig

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
OPTIMIZERS = {"rmsprop": "RMSProp", "adam": "Adam"}
LRS = np.geomspace(1e-5, 3.0, 121)
N_SAMPLES = 2048
TOLERANCE = 0.05  # near-optimal = loss within 5% of the best on the grid


def _sweep(job):
    optimizer, batch, sigma = job
    loss, _ = simulate(optimizer, LRS, batch, noise_var=float(sigma) ** 2, n_samples=N_SAMPLES, seed=batch)
    return job, loss


def near_optimal_band(loss):
    """(lo, hi) LR range around the argmin where loss stays within TOLERANCE of the best."""
    i = int(np.argmin(loss))
    ok = loss <= (1 + TOLERANCE) * loss[i]
    lo, hi = i, i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(loss) - 1 and ok[hi + 1]:
        hi += 1
    return LRS[lo], LRS[hi]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=300)
    sigma = parser.parse_args(argv).sigma
    jobs = [(o, b, sigma) for o in OPTIMIZERS for b in BATCHES]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        curves = dict(pool.map(_sweep, jobs))

    print(f"sigma = {sigma:g}: best LR and near-optimal band (loss within {TOLERANCE:.0%} of best)")
    bands = {}
    for o, name in OPTIMIZERS.items():
        for b in BATCHES:
            loss = curves[(o, b, sigma)]
            i = int(np.argmin(loss))
            lo, hi = bands[(o, b)] = near_optimal_band(loss)
            print(f"  {name:>7} B={b:>3}: best lr {LRS[i]:.2e} (loss {loss[i]:.3f})  band {lo:.2e}-{hi:.2e} "
                  f"(x{hi / lo:.1f} wide)  best/sqrt(B) {LRS[i] / np.sqrt(b):.2e}")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(BATCHES)))
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=(1, 1, 0.9)))
    for ax, (o, name) in zip(axes[:2], OPTIMIZERS.items()):
        for color, b in zip(colors, BATCHES):
            loss = curves[(o, b, sigma)]
            i = int(np.argmin(loss))
            lo, hi = bands[(o, b)]
            ax.plot(LRS, loss, color=color, lw=1.4, label=f"B = {b}")
            ax.plot([LRS[i]], [loss[i]], "o", color=color, mec="k", ms=5, zorder=3)
            band = (LRS >= lo) & (LRS <= hi)
            ax.plot(LRS[band], loss[band], color=color, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.set(xscale="log", yscale="log", ylim=(0.68, 1.3), xlim=(3e-4, 0.3), xlabel="learning rate",
               title=rf"{name}, $\sigma$ = {sigma:g}: loss vs. LR (thick = within {TOLERANCE:.0%} of best)")
        ticks = [0.7, 0.8, 0.9, 1, 1.2]
        ax.set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[0].set_ylabel(r"$L_N(B, \eta)$")

    ax = axes[2]
    b_arr = np.array(BATCHES)
    for (o, name), color in zip(OPTIMIZERS.items(), ("C0", "C3")):
        lo = np.array([bands[(o, b)][0] for b in BATCHES])
        hi = np.array([bands[(o, b)][1] for b in BATCHES])
        best = np.array([LRS[int(np.argmin(curves[(o, b, sigma)]))] for b in BATCHES])
        ax.fill_between(b_arr, lo, hi, color=color, alpha=0.2, label=f"{name}: within {TOLERANCE:.0%} of best")
        ax.plot(b_arr, best, "o-", color=color, mec="k", ms=5, label=f"{name}: best LR")
    ref = np.array([LRS[int(np.argmin(curves[("adam", 1, sigma)]))]]) * np.sqrt(b_arr)
    ax.plot(b_arr, ref, ":", color="0.4", label=r"$\propto\sqrt{B}$ (through Adam, B = 1)")
    ax.set(xscale="log", yscale="log", xlabel="batch size B", ylabel="learning rate",
           title="Near-optimal LR band vs. batch")
    ax.set_xticks(BATCHES, labels=[str(b) for b in BATCHES], fontsize=7)
    ax.minorticks_off()
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, f"p31c_curves_sigma{sigma:g}.png")


if __name__ == "__main__":
    main()
