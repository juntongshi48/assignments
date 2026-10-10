"""Problem 3.1(a): SGD on the 2D noisy quadratic. Fit eta*(B) = c B^p over B = 1..64, predict and test
B = 256, and extend through B = 512.

Losses are Monte Carlo estimates (8192 samples); the closed-form SGD expectation is overlaid as a check.
Writes plots/p31a_sgd_curves.png and plots/p31a_sgd_scaling.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.nqm import H_2D, N_EXAMPLES, power_law, sgd_exact, tune
from experiments.a2.plot_p1a import savefig

SOURCE = (1, 2, 4, 8, 16, 32, 64)
EXTENDED = (128, 256, 512)
TARGET = 256
N_SAMPLES = 8192
STABILITY_LIMIT = 2 / H_2D.max()  # plain SGD diverges for lr >= 2 / h_max


def sweep(batch, extra_lrs=()):
    """Two-stage LR sweep (simulated) plus the exact loss on the same grid; returns (results, optimum)."""
    lrs, loss, se, lr_opt, best, edge = tune(
        "sgd", batch, np.geomspace(1e-4, 0.999 * STABILITY_LIMIT, 40), max_lr=0.999 * STABILITY_LIMIT,
        extra_lrs=extra_lrs, n_samples=N_SAMPLES, seed=batch)
    exact = np.array([sgd_exact(lr, batch) for lr in lrs])
    return (lrs, loss, se, exact), (lr_opt, best, edge)


def main():
    results, opt = {}, {}
    for b in SOURCE:
        results[b], opt[b] = sweep(b)
    c, p, predict = power_law(SOURCE, [opt[b][0] for b in SOURCE])
    lr_pred = float(predict(TARGET))
    print(f"Source fit (B = 1..64): eta*(B) = {c:.3e} * B^{p:.3f}; prediction at B = {TARGET}: {lr_pred:.4f}")

    for b in EXTENDED:
        results[b], opt[b] = sweep(b, extra_lrs=[lr_pred] if b == TARGET else [])

    exact_opt = {}
    fine = np.geomspace(1e-5, 0.9999 * STABILITY_LIMIT, 4000)
    for b in results:
        losses = np.array([sgd_exact(lr, b) for lr in fine])
        exact_opt[b] = (float(fine[np.argmin(losses)]), float(losses.min()))

    print(f"\n{'B':>4} {'steps':>5}  {'eta* (sim)':>10}  {'eta* (exact)':>12}  {'eta*/B':>8}  "
          f"{'best loss (sim)':>15}  {'exact min':>9}  {'fit pred':>8}")
    for b in results:
        lr_opt, loss_best, edge = opt[b]
        print(f"{b:>4} {N_EXAMPLES // b:>5}  {lr_opt:>10.5f}  {exact_opt[b][0]:>12.5f}  {lr_opt / b:>8.2e}  "
              f"{loss_best:>15.3e}  {exact_opt[b][1]:>9.3e}  {float(predict(b)):>8.4f}"
              + ("  (argmin at grid edge)" if edge else ""))

    lrs, loss, se, exact = results[TARGET]
    i_pred = int(np.argmin(np.abs(lrs - lr_pred)))
    i_best = int(np.argmin(loss))
    print(f"\nB = {TARGET}: predicted LR {lr_pred:.4f} vs tuned {opt[TARGET][0]:.4f} "
          f"(ratio {lr_pred / opt[TARGET][0]:.2f})")
    print(f"  loss at predicted LR {loss[i_pred]:.3e} ± {se[i_pred]:.1e} (exact {sgd_exact(lr_pred, TARGET):.3e}); "
          f"best sampled {loss[i_best]:.3e} ± {se[i_best]:.1e} at LR {lrs[i_best]:.4f}; "
          f"ratio {loss[i_pred] / loss[i_best]:.2f}")

    # Loss-vs-LR curves.
    batches = list(results)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batches)))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for color, b in zip(colors, batches):
        lrs, loss, _, exact = results[b]
        ax.plot(lrs, loss, color=color, lw=1.6, label=f"B = {b}" + (" (target)" if b == TARGET else ""))
        ax.plot(lrs, exact, ":", color=color, lw=1)
        ax.plot([opt[b][0]], [opt[b][1]], "o", color=color, mec="k", ms=6, zorder=3)
    ax.axvline(STABILITY_LIMIT, color="C3", lw=1, ls="--", label=r"stability limit $2/h_{max}$")
    ax.set(xscale="log", yscale="log", ylim=(2e-4, 2), xlabel="learning rate", ylabel=r"$L_N(B, \eta)$",
           title="SGD on the 2D NQM, N = 8192 (solid: simulation, dotted: exact; dots: optima)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig(fig, "p31a_sgd_curves.png")

    # Optimal LR and best loss vs. batch.
    all_b = np.array(batches)
    b_grid = np.geomspace(1, 512, 100)
    fig, (ax_lr, ax_norm, ax_loss) = plt.subplots(1, 3, figsize=(16, 4.4))
    sim_opt = np.array([opt[b][0] for b in batches])
    src = np.isin(all_b, SOURCE)
    for ax, values, ylabel in ((ax_lr, sim_opt, "optimal LR"), (ax_norm, sim_opt / all_b, "optimal LR / B")):
        scale = 1 if ax is ax_lr else b_grid
        ax.plot(all_b[src], values[src], "o", color="C0", mec="k", ms=7, label="simulated optimum (fit)")
        ax.plot(all_b[~src], values[~src], "o", mfc="white", color="C0", mec="C0", ms=7, mew=1.5,
                label="simulated optimum (held out)")
        ax.plot(b_grid, predict(b_grid) / scale, "--", color="C1", label=rf"fit: ${c:.2e}\,B^{{{p:.3f}}}$")
        ax.plot(all_b, [exact_opt[b][0] / (1 if ax is ax_lr else b) for b in batches], ":", color="k", lw=1,
                label="exact optimum")
        ax.plot(b_grid, np.full_like(b_grid, STABILITY_LIMIT) / scale, color="C3", lw=1,
                label=r"stability limit $2/h_{max}$")
        ax.set(xscale="log", yscale="log", xlabel="batch size B", ylabel=ylabel)
        ax.set_xticks(batches, labels=[str(b) for b in batches])
        ax.minorticks_off()
        ax.legend(fontsize=7)
    ax_lr.plot([TARGET], [lr_pred], "X", color="C3", ms=10, mec="k", label="prediction at B = 256")
    ax_lr.set_title("Optimal LR vs. batch")
    ax_norm.set_title("Optimal LR / B (flat = linear scaling)")
    norm_ticks = [3e-4, 4e-4, 5e-4, 6e-4, 7e-4]
    ax_norm.set_ylim(2.8e-4, 7.5e-4)
    ax_norm.set_yticks(norm_ticks, labels=[f"{t:.0e}" for t in norm_ticks])
    ax_loss.plot(all_b, [opt[b][1] for b in batches], "o-", color="C2", mec="k", label="best loss after tuning LR")
    ax_loss.plot(all_b, [exact_opt[b][1] for b in batches], ":", color="k", lw=1, label="exact minimum")
    lrs, loss, _, _ = results[TARGET]
    ax_loss.plot([TARGET], [loss[int(np.argmin(np.abs(lrs - lr_pred)))]], "X", color="C3", ms=10, mec="k",
                 label="loss at predicted LR (B = 256)")
    ax_loss.set(xscale="log", yscale="log", xlabel="batch size B", ylabel="best final loss",
                title="Best loss after tuning LR (fixed N = 8192 examples)")
    ax_loss.set_xticks(batches, labels=[str(b) for b in batches])
    loss_ticks = [3e-4, 4e-4, 5e-4, 7e-4, 1e-3, 2e-3]
    ax_loss.set_yticks(loss_ticks, labels=[f"{t:.0e}" for t in loss_ticks])
    ax_loss.minorticks_off()
    ax_loss.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p31a_sgd_scaling.png")


if __name__ == "__main__":
    main()
