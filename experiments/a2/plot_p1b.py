"""Problem 1(b) analysis: fit the larger-budget loss-vs-LR curves and compare with the P1(a) law.

Writes plots/p1b_lr_scaling.png and prints predicted vs. fitted optima and the loss cost of
using the predicted LR (read off each budget's fitted curve).
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import by_budget, fit_lr_law, fit_optimum, savefig
from experiments.a2.provided_sweeps import load


def curve_loss(coefs, lr):
    c0, c1, c2 = coefs
    x = np.log(lr)
    return c0 + c1 * x + c2 * x**2


def main():
    src_curves = by_budget(load("P1a"))
    src_fits = {t: fit_optimum(*src_curves[t]) for t in src_curves}
    src_budgets = np.array(list(src_fits))
    a, alpha, predict = fit_lr_law(src_budgets, [src_fits[t][1] for t in src_budgets])

    curves = by_budget(load("P1b"))
    fits = {t: fit_optimum(*curves[t]) for t in curves}
    budgets = np.array(list(fits))

    print(f"P1(a) law: lr*(D) = {a:.4g} * D^{alpha:.3f}\n")
    print(f"{'D (tokens)':>14}  {'predicted':>9}  {'fitted lr*':>10}  {'pred/fit':>8}  "
          f"{'loss*':>7}  {'loss@pred':>9}  {'gap':>7}  {'best sampled':>14}")
    for tokens in budgets:
        lrs, losses = curves[tokens]
        coefs, lr_opt, loss_opt = fits[tokens]
        lr_pred = float(predict(tokens))
        loss_pred = float(curve_loss(coefs, lr_pred))
        best = int(np.argmin(losses))
        print(f"{tokens:>14,}  {lr_pred:>9.5f}  {lr_opt:>10.5f}  {lr_pred / lr_opt:>8.3f}  "
              f"{loss_opt:>7.4f}  {loss_pred:>9.4f}  {loss_pred - loss_opt:>7.4f}  "
              f"{lrs[best]:>6g} ({losses[best]:.4f})")

    colors = plt.cm.plasma(np.linspace(0.1, 0.75, len(budgets)))
    fig, (ax_curve, ax_law) = plt.subplots(1, 2, figsize=(12, 4.5))

    grid = np.geomspace(1e-3, 9e-3, 200)
    for color, tokens in zip(colors, budgets):
        lrs, losses = curves[tokens]
        coefs, lr_opt, loss_opt = fits[tokens]
        lr_pred = float(predict(tokens))
        ax_curve.plot(grid, curve_loss(coefs, grid), color=color, lw=1.5)
        ax_curve.scatter(lrs, losses, color=color, zorder=3, label=f"D = {tokens / 1e9:.4f}B")
        ax_curve.scatter([lr_opt], [loss_opt], color=color, marker="*", s=180,
                         edgecolor="k", zorder=4)
        ax_curve.scatter([lr_pred], [curve_loss(coefs, lr_pred)], color=color, marker="X", s=90,
                         edgecolor="k", zorder=4)
    ax_curve.set(xscale="log", xlabel="peak LR", ylabel="final validation loss",
                 title="P1(b) loss vs. LR (stars = fitted optima, X = P1(a) predictions)")
    ax_curve.set_xticks([.0015, .003, .006], labels=[".0015", ".003", ".006"])
    ax_curve.legend()

    all_budgets = np.concatenate([src_budgets, budgets])
    d_grid = np.geomspace(all_budgets.min() / 1.3, all_budgets.max() * 1.3, 100)
    mant, exp = f"{a:.2e}".split("e")
    ax_law.plot(d_grid, predict(d_grid), "k--", lw=1.2,
                label=rf"P1(a) law: $\eta^* = {mant}\times10^{{{int(exp)}}}\,D^{{{alpha:.3f}}}$")
    ax_law.scatter(src_budgets, [src_fits[t][1] for t in src_budgets], color="C0", s=70,
                   edgecolor="k", zorder=3, label="fitted optima, P1(a) budgets")
    ax_law.scatter(budgets, predict(budgets), marker="X", color="C3", s=80, zorder=3,
                   label="P1(a) predictions")
    ax_law.scatter(budgets, [fits[t][1] for t in budgets], color=colors, marker="*", s=180,
                   edgecolor="k", zorder=4, label="fitted optima, P1(b) budgets")
    ax_law.axhline(.003, color="0.6", lw=0.8, ls=":")
    ax_law.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel="optimal peak LR",
               title="Predicted vs. fitted optimal LR")
    ax_law.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "p1b_lr_scaling.png")


if __name__ == "__main__":
    main()
