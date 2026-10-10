"""Problem 1(a) analysis: per-budget loss-vs-LR fits, fitted optima, and an LR scaling rule in D.

Only the P1a rows are loaded, so the P1(b) budgets stay unseen until predictions are recorded.
Writes plots/p1a_lr_scaling.png and prints the fitted optima, the power law, and P1(b) predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.provided_sweeps import load

PLOT_DIR = Path(__file__).resolve().parent / "plots"
P1B_BUDGETS = (1_228_800_000, 1_843_200_000, 2_457_600_000)


def by_budget(rows):
    """tokens -> (lrs, losses), sorted by LR."""
    out = {}
    for tokens in sorted({r["tokens"] for r in rows}):
        pts = sorted((r["learning_rate"], r["final_val_loss"]) for r in rows if r["tokens"] == tokens)
        out[tokens] = tuple(np.array(v) for v in zip(*pts))
    return out


def fit_optimum(lrs, losses):
    """Fit loss = c0 + c1*x + c2*x^2 with x = log(lr); return (coefs, lr*, loss*).

    With three LRs per budget the quadratic interpolates exactly, so the vertex is the optimum
    of the parabola through the samples. A non-convex fit (c2 <= 0) has no interior optimum.
    """
    c2, c1, c0 = np.polyfit(np.log(lrs), losses, 2)
    if c2 <= 0:
        raise ValueError("loss-vs-log(LR) fit is not convex; no interior optimum")
    x_star = -c1 / (2 * c2)
    return (c0, c1, c2), float(np.exp(x_star)), float(c0 + c1 * x_star + c2 * x_star**2)


def fit_lr_law(tokens, lr_opts):
    """Fit lr*(D) = A * D^alpha by least squares in log-log space; return (A, alpha, predict)."""
    alpha, log_a = np.polyfit(np.log(tokens), np.log(lr_opts), 1)
    a = float(np.exp(log_a))
    return a, float(alpha), lambda d: a * np.asarray(d, dtype=float) ** alpha


def savefig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path.resolve())
    return path


def main():
    curves = by_budget(load("P1a"))
    fits = {tokens: fit_optimum(*curves[tokens]) for tokens in curves}
    budgets = np.array(list(fits))
    lr_opts = np.array([fits[t][1] for t in budgets])
    a, alpha, predict = fit_lr_law(budgets, lr_opts)

    print(f"{'D (tokens)':>14}  {'lr*':>9}  {'loss*':>7}  {'best sampled':>14}  {'law lr*':>9}")
    for tokens in budgets:
        lrs, losses = curves[tokens]
        _, lr_opt, loss_opt = fits[tokens]
        best = int(np.argmin(losses))
        print(f"{tokens:>14,}  {lr_opt:>9.5f}  {loss_opt:>7.4f}  "
              f"{lrs[best]:>6g} ({losses[best]:.4f})  {float(predict(tokens)):>9.5f}")
    print(f"\nScaling rule: lr*(D) = {a:.4g} * D^{alpha:.3f}"
          f"  (x2 data -> x{2**alpha:.3f} LR; lr* = {float(predict(614.4e6)):.5f} at 614.4M)")
    print("\nP1(b) predictions (record before loading P1b):")
    for tokens in P1B_BUDGETS:
        print(f"  D = {tokens / 1e9:.4f}B  ->  lr* = {float(predict(tokens)):.5f}")

    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(budgets)))
    fig, (ax_curve, ax_law) = plt.subplots(1, 2, figsize=(12, 4.5))

    grid = np.geomspace(1e-3, 9e-3, 200)
    for color, tokens in zip(colors, budgets):
        lrs, losses = curves[tokens]
        (c0, c1, c2), lr_opt, loss_opt = fits[tokens]
        x = np.log(grid)
        label = f"D = {tokens / 1e6:.1f}M"
        ax_curve.plot(grid, c0 + c1 * x + c2 * x**2, color=color, lw=1.5)
        ax_curve.scatter(lrs, losses, color=color, zorder=3, label=label)
        ax_curve.scatter([lr_opt], [loss_opt], color=color, marker="*", s=180,
                         edgecolor="k", zorder=4)
    ax_curve.set(xscale="log", xlabel="peak LR", ylabel="final validation loss",
                 title="Loss vs. LR with quadratic-in-log(LR) fits (stars = fitted optima)")
    ax_curve.set_xticks([.0015, .003, .006], labels=[".0015", ".003", ".006"])
    ax_curve.legend()

    d_grid = np.geomspace(budgets.min() / 1.3, max(P1B_BUDGETS) * 1.3, 100)
    mant, exp = f"{a:.2e}".split("e")
    ax_law.plot(d_grid, predict(d_grid), "k--", lw=1.2,
                label=rf"fit: $\eta^*(D) = {mant}\times10^{{{int(exp)}}}\,D^{{{alpha:.3f}}}$")
    ax_law.scatter(budgets, lr_opts, c=colors, s=70, zorder=3, edgecolor="k",
                   label="fitted optima (P1a)")
    ax_law.scatter(P1B_BUDGETS, predict(P1B_BUDGETS), marker="x", color="C3", s=70, zorder=3,
                   label="P1(b) predictions")
    ax_law.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel="optimal peak LR",
               title="Fitted optimal LR vs. training budget")
    ax_law.legend()
    fig.tight_layout()
    savefig(fig, "p1a_lr_scaling.png")


if __name__ == "__main__":
    main()
