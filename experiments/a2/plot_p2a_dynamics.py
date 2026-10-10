"""Problem 2(a) follow-up: why does tuning WD help less at larger budgets?

Uses the supplied runs' validation-loss curves (the only per-step data they log):
  1. When during training does a larger WD win or lose against WD = .1 at the same LR?
  2. How many weight-decay timescales does a run contain? With decoupled WD the weights forget their
     past at rate eta_t * lambda per step, so a run spans n = lambda * sum_t eta_t ~= lambda * eta * T / 2
     timescales (warmup + linear decay average to half the peak LR). Compare n at the fitted optimum
     with n at the default (eta, lambda) = (.003, .1).
Writes plots/p2a_wd_dynamics.png and plots/p2a_wd_timescales.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.histories import val_curves
from experiments.a2.plot_p1a import fit_lr_law, savefig
from experiments.a2.plot_p2a import fit_paraboloid, grids

BATCH_TOKENS = 64 * 1024
DEFAULT_LR, DEFAULT_WD = .003, .1


def curve(curves, tokens, lr, wd):
    """(progress, val_loss) for one source run, or None."""
    for c in curves.values():
        if (c["tokens"], c["learning_rate"], c["weight_decay"]) == (tokens, lr, wd):
            return np.array(c["steps"]) / c["total_steps"], np.array(c["val_loss"])
    return None


def timescales(lr, wd, tokens):
    return lr * wd * (tokens // BATCH_TOKENS) / 2


def conditional_wd(fit, lr):
    """WD minimizing the fitted paraboloid at a fixed LR."""
    _, _, c, _, e, f = fit["coefs"]
    return float(np.exp(-(c + e * np.log(lr)) / (2 * f)))


def main():
    curves = val_curves()
    data = grids()
    budgets = np.array(list(data))
    fits = {t: fit_paraboloid(*data[t]) for t in budgets}

    # 1. Loss gap to WD = .1 over training, at fixed LR.
    lrs = (.0015, .003)
    fig, axes = plt.subplots(len(lrs), len(budgets), figsize=(4.2 * len(budgets), 3.4 * len(lrs)),
                             sharex=True)
    progress = np.linspace(0.02, 1, 300)
    print("Loss gap vs. WD = .1 at the same LR (negative = larger WD better)")
    for row, lr in enumerate(lrs):
        for ax, t in zip(axes[row], budgets):
            base = curve(curves, t, lr, .1)
            wds = sorted({c["weight_decay"] for c in curves.values() if c["tokens"] == t} - {.1})
            colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(wds)))
            for color, wd in zip(colors, wds):
                other = curve(curves, t, lr, wd)
                if other is None or base is None:
                    continue
                gap = np.interp(progress, *other) - np.interp(progress, *base)
                # Earliest point after which the larger WD stays ahead of WD .1 until the end.
                behind = np.flatnonzero(gap >= 0)
                if gap[-1] >= 0:
                    when = "never ahead at the end"
                elif behind.size == 0:
                    when = "ahead throughout"
                else:
                    when = f"ahead from {progress[behind[-1] + 1]:.0%} of training on"
                print(f"  lr {lr:<6g} {t / 1e6:7.1f}M wd {wd:<4g}: mid-run (50%) "
                      f"{np.interp(.5, progress, gap):+.4f}, final {gap[-1]:+.4f}, {when}")
                ax.plot(progress, gap, color=color, lw=1.6, label=f"WD {wd:g}  (final {gap[-1]:+.3f})")
            ax.axhline(0, color="0.5", lw=0.8)
            # Show the LR multiplier (warmup + linear decay) for reference.
            lr_mult = np.clip(1 - (progress - .01) / .99, 0, 1)
            ax.fill_between(progress, -0.08, -0.08 + 0.03 * lr_mult, color="0.85", lw=0)
            ax.set_ylim(-0.085, 0.16)
            ax.set_title(f"D = {t / 1e9:.4g}B, LR {lr:g}", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
            if row == len(lrs) - 1:
                ax.set_xlabel("fraction of training")
        axes[row][0].set_ylabel("val loss − val loss at WD .1")
    fig.suptitle("Larger WD loses mid-training and wins only during the LR decay "
                 "(grey band = LR schedule)", y=1.01)
    fig.tight_layout()
    savefig(fig, "p2a_wd_dynamics.png")

    # 2. Weight-decay timescales per run: optimum vs. default.
    n_default = np.array([timescales(DEFAULT_LR, DEFAULT_WD, t) for t in budgets])
    n_joint = np.array([timescales(fits[t]["lr"], fits[t]["wd"], t) for t in budgets])
    n_fixed = {lr: np.array([timescales(lr, conditional_wd(fits[t], lr), t) for t in budgets]) for lr in lrs}
    gain = np.array([data[t][2].min() for t in budgets])  # filled below
    p1_best = {}
    for c in curves.values():
        if c["weight_decay"] == .1:
            p1_best[c["tokens"]] = min(p1_best.get(c["tokens"], np.inf), c["final_val_loss"])
    gain = gain - np.array([p1_best[t] for t in budgets])

    print("\nWeight-decay timescales per run, n = lambda * eta * T / 2")
    print(f"{'D':>8}  {'default':>7}  {'joint opt':>9}  " + "  ".join(f"{f'opt @ LR {lr:g}':>13}" for lr in lrs)
          + f"  {'default/opt':>11}  {'WD gain':>8}")
    for i, t in enumerate(budgets):
        print(f"{t / 1e6:>7.1f}M  {n_default[i]:>7.2f}  {n_joint[i]:>9.2f}  "
              + "  ".join(f"{n_fixed[lr][i]:>13.2f}" for lr in lrs)
              + f"  {n_default[i] / n_joint[i]:>11.2f}  {gain[i]:>+8.4f}")
    for name, values in (("default", n_default), ("joint optimum", n_joint)):
        print(f"  n ({name}) ~ D^{fit_lr_law(budgets, values)[1]:.2f}")

    fig, (ax_n, ax_gain) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_n.plot(budgets, n_default, "s-", color="C0", label="default (LR .003, WD .1)")
    ax_n.plot(budgets, n_joint, "o-", color="C3", label="fitted joint optimum")
    for lr, marker in zip(lrs, ("^", "v")):
        ax_n.plot(budgets, n_fixed[lr], marker + "--", color="C3", alpha=0.5, ms=5,
                  label=f"best WD at fixed LR {lr:g}")
    ax_n.set(xscale="log", yscale="log", xlabel="training tokens D",
             ylabel=r"WD timescales per run, $n = \lambda\eta T/2$",
             title="The default under-decays at small budgets")
    n_ticks = [.25, .5, 1, 2, 4]
    ax_n.set_yticks(n_ticks, labels=[f"{v:g}" for v in n_ticks])
    ax_n.minorticks_off()
    ax_n.legend(fontsize=8)
    ratio = n_default / n_joint
    ax_gain.plot(ratio, -gain, "o-", color="k")
    for r, g, t in zip(ratio, -gain, budgets):
        ax_gain.annotate(f"{t / 1e9:.4g}B", (r, g), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax_gain.set(xscale="log", xlabel="default n / optimal n", ylabel="loss gain from tuning WD",
                title="Gain shrinks as the default approaches the optimal n")
    ratio_ticks = [.2, .3, .5, .7]
    ax_gain.set_xticks(ratio_ticks, labels=[f"{v:g}" for v in ratio_ticks])
    ax_gain.minorticks_off()
    fig.tight_layout()
    savefig(fig, "p2a_wd_timescales.png")


if __name__ == "__main__":
    main()
