"""Problem 3.1(d): momentum at small (B = 16) and large (B = 256) batch on the 2D noisy quadratic, sigma = 1.

SGD with heavy-ball momentum (s <- mu s + g, w <- w - lr s), mu in {0, .9} plus extra values; Adam with
beta2 = .95 and a joint sweep of beta1 and LR (beta1 = 0 is RMSProp). LR is tuned separately for every
(batch, momentum) pair. Learning curves compare the best momentum setting with the best no-momentum one.
Writes plots/p31d_momentum.png and plots/p31d_curves.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.nqm import H_2D, N_EXAMPLES, simulate, tune
from experiments.a2.plot_p1a import savefig

BATCHES = (16, 256)
SGD_MUS = (0.0, 0.5, 0.9, 0.95, 0.99)
ADAM_BETA1S = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
N_SAMPLES = 4096
CURVE_POINTS = 64


def sgd_max_lr(mu):
    """Heavy-ball SGD on a quadratic is stable for lr * h < 2 (1 + mu)."""
    return 0.999 * 2 * (1 + mu) / H_2D.max()


def tune_sgd(batch, mu):
    _, _, _, lr, best, edge = tune("sgd", batch, np.geomspace(1e-5, sgd_max_lr(mu), 45), max_lr=sgd_max_lr(mu),
                                   momentum=mu, n_samples=N_SAMPLES, seed=batch)
    return lr, best, edge


def tune_adam(batch, beta1):
    optimizer = "rmsprop" if beta1 == 0 else "adam"
    _, _, _, lr, best, edge = tune(optimizer, batch, np.geomspace(1e-5, 3.0, 45), beta1=beta1,
                                   n_samples=N_SAMPLES, seed=batch)
    return lr, best, edge


def curve(kind, batch, momentum, lr):
    """Mean loss vs. examples processed for one tuned configuration (same seed as the sweep)."""
    every = max(1, (N_EXAMPLES // batch) // CURVE_POINTS)
    if kind == "sgd":
        _, _, c = simulate("sgd", [lr], batch, momentum=momentum, n_samples=N_SAMPLES, seed=batch, record_every=every)
    else:
        _, _, c = simulate("rmsprop" if momentum == 0 else "adam", [lr], batch, beta1=momentum,
                           n_samples=N_SAMPLES, seed=batch, record_every=every)
    return np.arange(1, len(c) + 1) * every * batch, c[:, 0]


def main():
    sgd = {(b, mu): tune_sgd(b, mu) for b in BATCHES for mu in SGD_MUS}
    adam = {(b, beta1): tune_adam(b, beta1) for b in BATCHES for beta1 in ADAM_BETA1S}

    for name, table, values in (("SGD momentum mu", sgd, SGD_MUS), ("Adam beta1", adam, ADAM_BETA1S)):
        print(f"\n{name}: tuned LR / best loss / best loss relative to no momentum")
        for b in BATCHES:
            base = table[(b, 0.0)][1]
            print(f"  B = {b:>3} ({N_EXAMPLES // b} updates)")
            for v in values:
                lr, best, edge = table[(b, v)]
                print(f"    {v:<6g} lr {lr:>9.5f}  loss {best:.3e}  x{best / base:.3f}" + ("  (LR at grid edge)" if edge else ""))

    best_sgd = {b: min(SGD_MUS, key=lambda mu: sgd[(b, mu)][1]) for b in BATCHES}
    best_adam = {b: min(ADAM_BETA1S, key=lambda v: adam[(b, v)][1]) for b in BATCHES}
    print("\nBenefit of momentum after tuning LR (best momentum loss / no-momentum loss):")
    for b in BATCHES:
        print(f"  B = {b:>3}: SGD mu=.9 x{sgd[(b, .9)][1] / sgd[(b, 0.0)][1]:.3f} "
              f"(best mu {best_sgd[b]:g}: x{sgd[(b, best_sgd[b])][1] / sgd[(b, 0.0)][1]:.3f});  "
              f"Adam beta1=.9 x{adam[(b, .9)][1] / adam[(b, 0.0)][1]:.3f} "
              f"(best beta1 {best_adam[b]:g}: x{adam[(b, best_adam[b])][1] / adam[(b, 0.0)][1]:.3f})")

    # Best loss vs. momentum coefficient, each LR-tuned; x is 1 - momentum so values near 1 spread out.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (name, table, values) in zip(axes, (("SGD, heavy-ball momentum", sgd, SGD_MUS),
                                                ("Adam, beta2 = .95", adam, ADAM_BETA1S))):
        for b, color in zip(BATCHES, ("C0", "C3")):
            base = table[(b, 0.0)][1]
            rel = [table[(b, v)][1] / base for v in values]
            x = [1 - v for v in values]
            ax.plot(x, rel, "o-", color=color, mec="k", label=f"B = {b} ({N_EXAMPLES // b} updates)")
            for xi, ri, v in zip(x, rel, values):
                if v in (0.0, 0.9) or v == (best_sgd if table is sgd else best_adam)[b]:
                    ax.annotate(f"{ri:.2f}", (xi, ri), textcoords="offset points", xytext=(4, 5), fontsize=7, color=color)
        ax.axhline(1, color="0.6", lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel="1 − momentum coefficient (left = more momentum)",
               ylabel="best loss / best loss without momentum", title=f"{name}: LR tuned at each point")
        ax.set_xticks([1 - v for v in values], labels=[f"{v:g}" for v in values], fontsize=7)
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    axes[0].set_xlabel(r"momentum $\mu$ (log scale in 1 − $\mu$)")
    axes[1].set_xlabel(r"$\beta_1$ (log scale in 1 − $\beta_1$)")
    fig.tight_layout()
    savefig(fig, "p31d_momentum.png")

    # Learning curves: best momentum setting vs. best no-momentum baseline, per optimizer and batch.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, b in zip(axes, BATCHES):
        for kind, table, best, color, name in (("sgd", sgd, best_sgd[b], "0.3", "SGD"),
                                               ("adam", adam, best_adam[b], "C3", "Adam")):
            for v, style in ((0.0, ":"), (best, "-")):
                lr = table[(b, v)][0]
                x, c = curve(kind, b, v, lr)
                label = (f"{name} no momentum" if v == 0 else
                         f"{name} {'mu' if kind == 'sgd' else 'beta1'} = {v:g}") + f" (lr {lr:.3g})"
                ax.plot(x, c, style, color=color, lw=1.6, label=label)
        ax.set(xscale="log", yscale="log", xlabel="examples processed", ylabel="mean loss",
               title=f"B = {b}: best momentum (solid) vs. best no-momentum (dotted)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p31d_curves.png")


if __name__ == "__main__":
    main()
