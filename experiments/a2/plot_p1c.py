"""Problem 1(c) analysis: two LR power laws (all six source budgets vs. the larger three),
their 4.9152B predictions, and a comparison against the measured target sweep.

Target runs are read from W&B by tag once finished; unfinished runs are skipped because their
summary val_loss is a mid-training value. Writes plots/p1c_lr_scaling.png when results exist.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.p1_learning_rate import EXPERIMENT_KEY, LEARNING_RATES, TARGET_TOKENS
from experiments.a2.plot_p1a import by_budget, fit_lr_law, fit_optimum, savefig
from experiments.a2.plot_p1b import curve_loss
from experiments.a2.provided_sweeps import load

# Rounded predictions passed to the launcher (see main() for the unrounded fits).
PREDICTED_LRS = {"all six": 0.00373, "larger three": 0.00223}


def source_laws():
    curves = {**by_budget(load("P1a")), **by_budget(load("P1b"))}
    optima = {t: fit_optimum(*curves[t])[1] for t in sorted(curves)}
    budgets = list(optima)
    laws = {name: fit_lr_law(ts, [optima[t] for t in ts])
            for name, ts in (("all six", budgets), ("larger three", budgets[3:]))}
    return optima, laws


def target_measurements():
    """lr -> final val loss for finished target runs."""
    runs = [r for r in load_runs([EXPERIMENT_KEY]) if r.state == "finished" and r.val_loss is not None]
    return dict(sorted((float(r.config["learning_rate"]), r.val_loss) for r in runs))


def main():
    optima, laws = source_laws()
    print(f"Source optima: " + ", ".join(f"{t / 1e9:.4g}B: {lr:.5f}" for t, lr in optima.items()))
    for name, (a, alpha, predict) in laws.items():
        ts = list(optima) if name == "all six" else list(optima)[3:]
        resid = np.log([optima[t] for t in ts]) - np.log(predict(ts))
        print(f"{name:>13}: lr*(D) = {a:.4g} * D^{alpha:.4f}   rms log-resid {np.sqrt(np.mean(resid**2)):.3f}"
              f"   -> lr*(4.9152B) = {float(predict(TARGET_TOKENS)):.5f}"
              f" (launched {PREDICTED_LRS[name]:g})")

    measured = target_measurements()
    expected = set(LEARNING_RATES) | set(PREDICTED_LRS.values())
    missing = sorted(expected - set(measured))
    print(f"\nFinished target runs: {len(measured)}/{len(expected)}"
          + (f"  (waiting on lr = {', '.join(f'{lr:g}' for lr in missing)})" if missing else ""))
    for lr, loss in measured.items():
        print(f"  lr = {lr:<8g} loss = {loss:.4f}")
    if len(measured) < 3:
        return

    lrs, losses = map(np.array, zip(*measured.items()))
    c2, c1, c0 = np.polyfit(np.log(lrs), losses, 2)
    coefs = (c0, c1, c2)
    fit_resid = losses - curve_loss(coefs, lrs)
    lr_opt = float(np.exp(-c1 / (2 * c2))) if c2 > 0 else float("nan")
    grid_best_lr = min((lr for lr in LEARNING_RATES if lr in measured), key=measured.get)
    grid_best = measured[grid_best_lr]
    print(f"\nTarget fit over {len(lrs)} LRs: lr* = {lr_opt:.5f}, loss* = "
          f"{float(curve_loss(coefs, lr_opt)):.4f}, rms resid = {np.sqrt(np.mean(fit_resid**2)):.4f}")
    print(f"Best sampled grid LR: {grid_best_lr:g} (loss {grid_best:.4f})")
    for name, lr in PREDICTED_LRS.items():
        if lr in measured:
            print(f"  {name:>13}: lr = {lr:g}  pred/fit = {lr / lr_opt:.3f}  "
                  f"measured loss {measured[lr]:.4f}  gap vs best grid = {measured[lr] - grid_best:+.4f}")

    fig, (ax_curve, ax_law) = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = np.geomspace(1e-3, 9e-3, 200)
    ax_curve.plot(grid, curve_loss(coefs, grid), color="0.3", lw=1.5, label="quadratic fit in log(LR)")
    grid_lrs = [lr for lr in LEARNING_RATES if lr in measured]
    ax_curve.scatter(grid_lrs, [measured[lr] for lr in grid_lrs], color="0.3", zorder=3,
                     label="target grid")
    for color, (name, lr) in zip(("C0", "C1"), PREDICTED_LRS.items()):
        if lr in measured:
            ax_curve.scatter([lr], [measured[lr]], color=color, marker="X", s=110, edgecolor="k",
                             zorder=4, label=f"predicted ({name}): {lr:g}")
    if np.isfinite(lr_opt):
        ax_curve.scatter([lr_opt], [curve_loss(coefs, lr_opt)], color="gold", marker="*", s=220,
                         edgecolor="k", zorder=5, label=f"fitted optimum: {lr_opt:.5f}")
    ax_curve.set(xscale="log", xlabel="peak LR", ylabel="final validation loss",
                 title="Target budget D = 4.9152B")
    ax_curve.set_xticks([.0015, .003, .006], labels=[".0015", ".003", ".006"])
    ax_curve.legend(fontsize=9)

    budgets = np.array(list(optima))
    for color, (name, (a, alpha, predict)) in zip(("C0", "C1"), laws.items()):
        # Draw each law from the first budget it was fit on, so it is not read as covering earlier ones.
        d_grid = np.geomspace(budgets[0 if name == "all six" else 3] / 1.15, TARGET_TOKENS * 1.3, 100)
        ax_law.plot(d_grid, predict(d_grid), "--", color=color, lw=1.2,
                    label=rf"{name}: $\alpha = {alpha:.3f}$")
        ax_law.scatter([TARGET_TOKENS], [PREDICTED_LRS[name]], color=color, marker="X", s=90,
                       edgecolor="k", zorder=4)
    ax_law.scatter(budgets, list(optima.values()), color="0.3", s=60, edgecolor="k", zorder=3,
                   label="fitted source optima")
    if np.isfinite(lr_opt):
        ax_law.scatter([TARGET_TOKENS], [lr_opt], color="gold", marker="*", s=220, edgecolor="k",
                       zorder=5, label="fitted target optimum")
    ax_law.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel="optimal peak LR",
               title="Two LR scaling laws extrapolated to 4.9152B")
    ax_law.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "p1c_lr_scaling.png")


if __name__ == "__main__":
    main()
