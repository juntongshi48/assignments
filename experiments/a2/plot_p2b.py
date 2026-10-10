"""Problem 2(b) analysis: LR-WD coupling and a joint scaling law (no new runs).

For each budget's P2a paraboloid fit:
  * draw the iso-product line eta * lambda = eta* lambda* and the valley axis (the direction of least
    curvature of the fit) through the optimum, and measure how much loss a 2x move costs along the
    product line vs. along LR or WD alone;
  * fit power laws in D to eta*, lambda*, and eta* lambda*, reporting R^2 and the RMS log residual
    (R^2 alone favours quantities that vary over a wide range).
Writes plots/p2b_contours.png and plots/p2b_scaling.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import fit_lr_law, savefig
from experiments.a2.plot_p2a import fit_paraboloid, grids, r_squared_loglog, surface


def valley(fit):
    """Unit direction (dlog lr, dlog wd) of least curvature, its angle from the LR axis measured toward
    decreasing WD (the iso-product direction is 45 deg), and the curvature ratio across/along it."""
    _, _, _, d, e, f = fit["coefs"]
    eigs, vecs = np.linalg.eigh(np.array([[2 * d, e], [e, 2 * f]]))
    v = vecs[:, 0] * np.sign(vecs[0, 0])
    return v, float(np.degrees(np.arctan2(-v[1], v[0]))), float(eigs[1] / eigs[0])


def move_costs(fit, factor=2.0):
    """Loss increase from the optimum for a `factor` move along the product line, LR only, WD only."""
    lr, wd, coefs = fit["lr"], fit["wd"], fit["coefs"]
    return {name: float(surface(coefs, lr * a, wd * b) - fit["loss"])
            for name, (a, b) in (("product line", (factor, 1 / factor)), ("LR only", (factor, 1)),
                                 ("WD only", (1, factor)))}


def main():
    data = grids()
    budgets = np.array(list(data))
    fits = {t: fit_paraboloid(*data[t]) for t in budgets}

    print(f"{'D':>8}  {'lr*':>8}  {'wd*':>6}  {'lr*wd*':>9}  {'valley angle':>12}  {'curv ratio':>10}  "
          f"{'2x cost: product':>16}  {'LR only':>7}  {'WD only':>7}")
    for t in budgets:
        fit = fits[t]
        _, angle, ratio = valley(fit)
        costs = move_costs(fit)
        print(f"{t / 1e6:>7.1f}M  {fit['lr']:>8.5f}  {fit['wd']:>6.3f}  {fit['lr'] * fit['wd']:>9.2e}  "
              f"{angle:>10.1f}°  {ratio:>10.1f}  {costs['product line']:>16.4f}  {costs['LR only']:>7.4f}  "
              f"{costs['WD only']:>7.4f}")

    quantities = {
        r"$\eta^*$ (optimal LR)": np.array([fits[t]["lr"] for t in budgets]),
        r"$\lambda^*$ (optimal WD)": np.array([fits[t]["wd"] for t in budgets]),
        r"$\eta^*\lambda^*$ (product)": np.array([fits[t]["lr"] * fits[t]["wd"] for t in budgets]),
    }
    print(f"\n{'quantity':>14}  {'range':>6}  {'exponent':>8}  {'R^2':>6}  {'rms log resid':>13}")
    stats = {}
    for name, values in quantities.items():
        _, alpha, predict = fit_lr_law(budgets, values)
        resid = np.log(values) - np.log(predict(budgets))
        stats[name] = (alpha, r_squared_loglog(budgets, values), float(np.sqrt(np.mean(resid**2))), predict)
        label = name.split("(")[1].rstrip(")")
        print(f"{label:>14}  x{values.max() / values.min():>5.2f}  {alpha:>+8.3f}  {stats[name][1]:>6.3f}  "
              f"{stats[name][2]:>13.3f}")

    # Contours with the iso-product line and the valley axis.
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 4.6))
    for ax, t in zip(axes, budgets):
        lrs, wds, losses = data[t]
        fit = fits[t]
        lr_g = np.geomspace(lrs.min() / 1.6, lrs.max() * 1.6, 160)
        wd_g = np.geomspace(wds.min() / 1.6, wds.max() * 1.6, 160)
        LR, WD = np.meshgrid(lr_g, wd_g)
        Z = surface(fit["coefs"], LR, WD)
        levels = fit["loss"] + np.geomspace(0.002, 0.25, 14)
        cs = ax.contour(LR, WD, Z, levels=levels, cmap="viridis_r", linewidths=1)
        ax.clabel(cs, fmt="%.3f", fontsize=6)
        ax.scatter(lrs, wds, c=losses, cmap="viridis_r", edgecolor="k", s=55, zorder=3)
        for lr, wd, loss in zip(lrs, wds, losses):
            ax.annotate(f"{loss:.4f}", (lr, wd), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=7)
        ax.plot(lr_g, fit["lr"] * fit["wd"] / lr_g, "k--", lw=1.4,
                label=rf"$\eta\lambda = \eta^*\lambda^*$ = {fit['lr'] * fit['wd']:.1e}")
        v, angle, ratio = valley(fit)
        s = np.linspace(-3, 3, 50)
        ax.plot(fit["lr"] * np.exp(s * v[0]), fit["wd"] * np.exp(s * v[1]), ":", color="C3", lw=1.8,
                label=f"valley axis ({angle:.0f}°; product line = 45°)")
        ax.scatter([fit["lr"]], [fit["wd"]], marker="*", s=260, color="gold", edgecolor="k", zorder=5)
        costs = move_costs(fit)
        ax.set_title(f"D = {t / 1e9:.4g}B\n2x move costs: product line {costs['product line']:.4f}, "
                     f"LR only {costs['LR only']:.4f}", fontsize=9)
        ax.set(xscale="log", yscale="log", xlabel="peak LR", xlim=(lr_g[0], lr_g[-1]), ylim=(wd_g[0], wd_g[-1]))
        ax.set_xticks(sorted(set(lrs)), labels=[f"{x:g}" for x in sorted(set(lrs))])
        ax.set_yticks(sorted(set(wds)), labels=[f"{x:g}" for x in sorted(set(wds))])
        ax.minorticks_off()
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("weight decay")
    fig.suptitle("Paraboloid fits with the iso-product line through each optimum", y=1.03)
    fig.tight_layout()
    savefig(fig, "p2b_contours.png")

    # Power laws in D for each quantity.
    fig, axes = plt.subplots(1, len(quantities), figsize=(5 * len(quantities), 4.2))
    d_grid = np.geomspace(budgets.min() / 1.3, budgets.max() * 1.3, 100)
    y_ticks = ([.0016, .0018, .002, .0022, .0025, .0028], [.15, .2, .3, .5, .7, 1],
               [4e-4, 5e-4, 7e-4, 1e-3, 1.5e-3])
    for ax, (name, values), color, ticks in zip(axes, quantities.items(), ("C3", "C2", "C4"), y_ticks):
        alpha, r2, rms, predict = stats[name]
        ax.plot(budgets, values, "o", color=color, ms=8, mec="k", zorder=3, label="fitted optima")
        ax.plot(d_grid, predict(d_grid), "--", color=color, lw=1.2,
                label=rf"$\propto D^{{{alpha:+.3f}}}$, $R^2$ = {r2:.3f}, rms log resid = {rms:.3f}")
        ax.set(xscale="log", yscale="log", xlabel="training tokens D", title=name)
        ax.set_yticks(ticks, labels=[f"{y:g}" for y in ticks])
        ax.minorticks_off()
        ax.legend(fontsize=8)
    fig.suptitle("Which quantity follows the tightest power law in D?", y=1.02)
    fig.tight_layout()
    savefig(fig, "p2b_scaling.png")


if __name__ == "__main__":
    main()
