"""Problem 2(a) analysis: joint LR-WD paraboloid fits per budget, fitted optima vs. tokens, and the
value of tuning WD relative to Problem 1's WD = .1 sweeps.

With x = log(lr), y = log(wd), each budget's nine runs are fit by least squares to
    L(x, y) = a + b x + c y + d x^2 + e x y + f y^2.
Writes plots/p2a_contours.png, plots/p2a_optima.png, plots/p2a_wd_benefit.png.
"""

from __future__ import annotations

import collections

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import by_budget, fit_lr_law, fit_optimum, savefig
from experiments.a2.provided_sweeps import load


def grids(part="P2a"):
    """tokens -> (lrs, wds, losses) arrays, one entry per run."""
    out = collections.defaultdict(list)
    for r in load(part):
        out[r["tokens"]].append((r["learning_rate"], r["weight_decay"], r["final_val_loss"]))
    return {t: tuple(map(np.array, zip(*out[t]))) for t in sorted(out)}


def paraboloid_terms(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def fit_paraboloid(lrs, wds, losses):
    """Least-squares paraboloid in (log lr, log wd); returns dict with coefs, optimum, and diagnostics.

    Raises if the Hessian is not positive definite (no interior minimum).
    """
    x, y = np.log(lrs), np.log(wds)
    coefs, *_ = np.linalg.lstsq(paraboloid_terms(x, y), losses, rcond=None)
    _, b, c, d, e, f = coefs
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    eigs = np.linalg.eigvalsh(hessian)
    if eigs.min() <= 0:
        raise ValueError(f"paraboloid is not convex (Hessian eigenvalues {eigs})")
    x_opt, y_opt = np.linalg.solve(hessian, [-b, -c])
    resid = losses - paraboloid_terms(x, y) @ coefs
    return dict(
        coefs=coefs, lr=float(np.exp(x_opt)), wd=float(np.exp(y_opt)),
        loss=float(paraboloid_terms(x_opt, y_opt) @ coefs), eigs=eigs,
        rms=float(np.sqrt(np.mean(resid**2))), max_resid=float(np.abs(resid).max()),
        inside=bool(lrs.min() <= np.exp(x_opt) <= lrs.max() and wds.min() <= np.exp(y_opt) <= wds.max()),
    )


def surface(coefs, lr, wd):
    return paraboloid_terms(np.log(lr), np.log(wd)) @ coefs


def r_squared_loglog(tokens, values):
    """R^2 of a straight-line fit of log(values) on log(tokens)."""
    lx, ly = np.log(tokens), np.log(values)
    pred = np.polyval(np.polyfit(lx, ly, 1), lx)
    return 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)


def p1_reference(budgets):
    """Problem 1 (WD = .1) fitted optimal LR and best measured loss at the given budgets."""
    curves = {**by_budget(load("P1a")), **by_budget(load("P1b"))}
    return ({t: fit_optimum(*curves[t])[1] for t in budgets},
            {t: float(np.min(curves[t][1])) for t in budgets})


def main():
    data = grids()
    budgets = np.array(list(data))
    fits = {t: fit_paraboloid(*data[t]) for t in budgets}
    p1_lr, p1_best = p1_reference(budgets)

    print(f"{'D':>8}  {'lr*':>8}  {'wd*':>6}  {'loss*':>7}  {'in grid':>7}  {'rms':>6}  {'max':>6}  "
          f"{'Hessian eigs':>16}  {'best measured (lr, wd)':>26}  {'P1 lr* (wd .1)':>14}")
    for t in budgets:
        lrs, wds, losses = data[t]
        fit, i = fits[t], int(np.argmin(losses))
        print(f"{t / 1e6:>7.1f}M  {fit['lr']:>8.5f}  {fit['wd']:>6.3f}  {fit['loss']:>7.4f}  {str(fit['inside']):>7}  "
              f"{fit['rms']:>6.4f}  {fit['max_resid']:>6.4f}  {np.array2string(fit['eigs'], precision=4):>16}  "
              f"{losses[i]:.4f} ({lrs[i]:g}, {wds[i]:g}){'':>4}  {p1_lr[t]:>14.5f}")

    opt_lr = np.array([fits[t]["lr"] for t in budgets])
    opt_wd = np.array([fits[t]["wd"] for t in budgets])
    print()
    for name, values in (("joint lr*", opt_lr), ("joint wd*", opt_wd),
                         ("P1 lr* (wd .1)", np.array([p1_lr[t] for t in budgets]))):
        a, alpha, _ = fit_lr_law(budgets, values)
        print(f"{name:>15}: power law exponent {alpha:+.3f} (x{2**alpha:.3f} per doubling), "
              f"log-log R^2 = {r_squared_loglog(budgets, values):.3f}")

    best_joint = np.array([data[t][2].min() for t in budgets])
    best_p1 = np.array([p1_best[t] for t in budgets])
    print("\nBest measured loss: joint LR-WD grid vs. Problem 1 (WD = .1)")
    for t, j, p in zip(budgets, best_joint, best_p1):
        print(f"  {t / 1e6:>7.1f}M  joint {j:.4f}  P1 {p:.4f}  difference {j - p:+.4f}")

    # Contours, one panel per budget.
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 4.4))
    for ax, t in zip(axes, budgets):
        lrs, wds, losses = data[t]
        fit = fits[t]
        lr_g = np.geomspace(lrs.min() / 1.6, lrs.max() * 1.6, 120)
        wd_g = np.geomspace(wds.min() / 1.6, wds.max() * 1.6, 120)
        LR, WD = np.meshgrid(lr_g, wd_g)
        Z = surface(fit["coefs"], LR, WD)
        levels = fit["loss"] + np.geomspace(0.002, 0.25, 14)
        cs = ax.contour(LR, WD, Z, levels=levels, cmap="viridis_r", linewidths=1)
        ax.clabel(cs, fmt="%.3f", fontsize=6)
        sc = ax.scatter(lrs, wds, c=losses, cmap="viridis_r", edgecolor="k", s=55, zorder=3)
        for lr, wd, loss in zip(lrs, wds, losses):
            ax.annotate(f"{loss:.4f}", (lr, wd), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)
        i = int(np.argmin(losses))
        ax.scatter([lrs[i]], [wds[i]], marker="s", s=140, facecolor="none", edgecolor="C3", lw=1.8,
                   zorder=4, label="best measured")
        ax.scatter([fit["lr"]], [fit["wd"]], marker="*", s=260, color="gold", edgecolor="k", zorder=5,
                   label=f"fit: lr {fit['lr']:.4f}, wd {fit['wd']:.3f}")
        ax.set(xscale="log", yscale="log", xlabel="peak LR", title=f"D = {t / 1e9:.4g}B")
        ax.set_xticks(sorted(set(lrs)), labels=[f"{v:g}" for v in sorted(set(lrs))])
        ax.set_yticks(sorted(set(wds)), labels=[f"{v:g}" for v in sorted(set(wds))])
        ax.minorticks_off()
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("weight decay")
    fig.suptitle("Paraboloid fits of final val loss in (log LR, log WD)", y=1.02)
    fig.tight_layout()
    savefig(fig, "p2a_contours.png")

    # Fitted optima vs. tokens.
    fig, (ax_lr, ax_wd) = plt.subplots(1, 2, figsize=(11, 4.2))
    d_grid = np.geomspace(budgets.min() / 1.3, budgets.max() * 1.3, 100)
    for ax, values, label, color in ((ax_lr, opt_lr, "joint fit", "C3"), (ax_wd, opt_wd, "joint fit", "C2")):
        _, alpha, predict = fit_lr_law(budgets, values)
        ax.plot(budgets, values, "o", color=color, ms=8, mec="k", zorder=3, label=label)
        ax.plot(d_grid, predict(d_grid), "--", color=color, lw=1.2,
                label=rf"power law $\alpha = {alpha:+.3f}$, $R^2 = {r_squared_loglog(budgets, values):.2f}$")
    p1 = np.array([p1_lr[t] for t in budgets])
    _, alpha, predict = fit_lr_law(budgets, p1)
    ax_lr.plot(budgets, p1, "s", color="C0", ms=7, mec="k", zorder=3, label="Problem 1 (WD = .1)")
    ax_lr.plot(d_grid, predict(d_grid), ":", color="C0", lw=1.2,
               label=rf"P1 power law $\alpha = {alpha:+.3f}$, $R^2 = {r_squared_loglog(budgets, p1):.2f}$")
    ax_lr.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel="optimal peak LR",
              title="Optimal LR vs. budget")
    ax_wd.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel="optimal WD",
              title="Optimal WD vs. budget (joint fit)")
    wd_ticks = [.1, .15, .2, .3, .5, .7, 1]
    ax_wd.set_yticks(wd_ticks, labels=[f"{t:g}" for t in wd_ticks])
    ax_wd.minorticks_off()
    for ax in (ax_lr, ax_wd):
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p2a_optima.png")

    # Value of tuning WD.
    fig, (ax_loss, ax_diff) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_loss.plot(budgets, best_p1, "s-", color="C0", label="Problem 1: tune LR, WD = .1")
    ax_loss.plot(budgets, best_joint, "o-", color="C3", label="Problem 2: tune LR and WD")
    ax_loss.set(xscale="log", xlabel="training tokens D", ylabel="best measured val loss",
                title="Best measured loss vs. budget")
    ax_loss.legend()
    ax_diff.plot(budgets, best_joint - best_p1, "o-", color="k")
    ax_diff.axhline(0, color="0.6", lw=0.8)
    ax_diff.axhspan(-0.0017, 0.0017, color="0.85", label="±1 seed std (A1, 614.4M)")
    for t, v in zip(budgets, best_joint - best_p1):
        ax_diff.annotate(f"{v:+.4f}", (t, v), textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8)
    ax_diff.set(xscale="log", xlabel="training tokens D", ylabel="joint best − P1 best",
                title="Gain from also tuning WD")
    ax_diff.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p2a_wd_benefit.png")


if __name__ == "__main__":
    main()
