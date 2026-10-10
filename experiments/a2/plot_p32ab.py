"""Problem 3.2(a, b-source): batch-size sweeps at 614.4M tokens and both hypotheses' B = 128/256 predictions.

Measurements (batch, lr, wd) -> final val loss come from: our A2 runs (tag a2-p32-D614m), the reused A1 runs
at B = 16/32 (p32_batch.REUSED_A1; same model, data prefix and AdamW math), and the supplied B = 64 sweeps
(P1a, P2a). Each loss curve is fit by a quadratic in log(x) through the best point and its two neighbours
(the edge pair if the best point is on the edge, flagged as extrapolated).

Hypothesis (i): WD .1, optimal LR ~ B^p.  Hypothesis (ii): LR .0015, optimal WD ~ B^q.
Writes plots/p32a_lr.png and plots/p32b_wd.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.p32_batch import EXPERIMENT_KEY, REUSED_A1, TOKENS
from experiments.a2.plot_p1a import savefig
from experiments.a2.provided_sweeps import load

BATCHES = (8, 16, 32, 64)
TARGETS = (128, 256)
FIXED_WD, FIXED_LR = .1, .0015


def measurements():
    """(batch, lr, wd) -> final val loss at 614.4M tokens."""
    out = {}
    for part in ("P1a", "P2a"):
        for r in load(part):
            if r["tokens"] == TOKENS:
                out[(r["batch_size"], r["learning_rate"], r["weight_decay"])] = r["final_val_loss"]
    a1 = {r.name: r for r in load_runs([], names=list(REUSED_A1.values()))}
    for (b, lr), name in REUSED_A1.items():
        out[(b, lr, FIXED_WD)] = a1[name].val_loss
    for r in load_runs([EXPERIMENT_KEY]):
        if r.state == "finished":
            out[(r.config["batch_size"], r.config["learning_rate"], r.config["weight_decay"])] = r.val_loss
    return out


def local_optimum(xs, losses):
    """Vertex of a quadratic in log(x) through the best point and its neighbours.

    Returns (x*, loss*, extrapolated, coefs) where extrapolated flags a best point on the sweep edge.
    """
    order = np.argsort(xs)
    xs, losses = np.asarray(xs)[order], np.asarray(losses)[order]
    i = int(np.argmin(losses))
    extrapolated = i in (0, len(xs) - 1)
    lo = min(max(i - 1, 0), len(xs) - 3)
    window = slice(lo, lo + 3)
    c2, c1, c0 = np.polyfit(np.log(xs[window]), losses[window], 2)
    if c2 <= 0:
        return float(xs[i]), float(losses[i]), True, (c0, c1, c2)
    x = -c1 / (2 * c2)
    return float(np.exp(x)), float(c0 + c1 * x + c2 * x * x), extrapolated, (c0, c1, c2)


def sweep(m, batch, axis):
    """Sorted (xs, losses) for the fixed-WD LR sweep (axis='lr') or fixed-LR WD sweep (axis='wd')."""
    pts = sorted((k[1] if axis == "lr" else k[2], v) for k, v in m.items()
                 if k[0] == batch and (k[2] == FIXED_WD if axis == "lr" else k[1] == FIXED_LR))
    return tuple(np.array(v) for v in zip(*pts))


def power_fit(batches, values):
    p, log_c = np.polyfit(np.log(batches), np.log(values), 1)
    return float(np.exp(log_c)), float(p), lambda b: np.exp(log_c) * np.asarray(b, dtype=float) ** p


def analyse(m, axis):
    fits = {b: local_optimum(*sweep(m, b, axis)) for b in BATCHES}
    c, p, predict = power_fit(BATCHES, [fits[b][0] for b in BATCHES])
    return fits, (c, p, predict)


def plot_sweeps(m, axis, fits, law, fname, xlabel, title, fixed_label):
    c, p, predict = law
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(BATCHES)))
    fig, (ax_curve, ax_opt) = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, b in zip(colors, BATCHES):
        xs, losses = sweep(m, b, axis)
        x_opt, l_opt, extrap, (c0, c1, c2) = fits[b]
        grid = np.geomspace(xs.min() / 1.3, xs.max() * 1.3, 200)
        ax_curve.plot(xs, losses, "o", color=color, mec="k", ms=6, label=f"B = {b}")
        ax_curve.plot(grid, c0 + c1 * np.log(grid) + c2 * np.log(grid) ** 2, color=color, lw=1, alpha=0.6)
        ax_curve.plot([x_opt], [l_opt], "*", color=color, mec="k", ms=14, mfc="white" if extrap else color)
    ax_curve.set(xscale="log", xlabel=xlabel, ylabel="final val loss", ylim=(2.912, 2.97 if axis == "lr" else 3.01),
                 title=f"{title} ({fixed_label}); stars = local fits, hollow = not bracketed")
    ax_curve.legend(fontsize=8)

    b_all = np.array(BATCHES + TARGETS)
    grid = np.geomspace(6, 320, 100)
    opt = np.array([fits[b][0] for b in BATCHES])
    extrap = np.array([fits[b][2] for b in BATCHES])
    ax_opt.plot(np.array(BATCHES)[~extrap], opt[~extrap], "o", color="C0", mec="k", ms=8, label="fitted optimum")
    if extrap.any():
        ax_opt.plot(np.array(BATCHES)[extrap], opt[extrap], "o", mfc="white", color="C0", mew=1.5, ms=8,
                    label="fitted optimum (best point on grid edge)")
    ax_opt.plot(grid, predict(grid), "--", color="C1", label=rf"fit $\propto B^{{{p:.2f}}}$")
    ax_opt.plot(TARGETS, predict(TARGETS), "X", color="C3", mec="k", ms=11, label="predictions B = 128, 256")
    ref = predict(64)
    for slope, style, name in ((0.5, ":", r"$\propto\sqrt{B}$"), (1.0, "-.", r"$\propto B$")):
        ax_opt.plot(grid, ref * (grid / 64) ** slope, style, color="0.6", lw=1, label=f"{name} through B = 64 fit")
    ax_opt.set(xscale="log", yscale="log", xlabel="batch size B (sequences)", ylabel=f"optimal {xlabel}",
               title=f"Optimal {xlabel} vs. batch ({fixed_label})")
    ax_opt.set_xticks(list(b_all), labels=[str(b) for b in b_all])
    ax_opt.minorticks_off()
    ax_opt.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, fname)


def main():
    m = measurements()
    lr_fits, lr_law = analyse(m, "lr")
    wd_fits, wd_law = analyse(m, "wd")

    print("(a) WD = .1: loss vs. LR")
    for b in BATCHES:
        xs, losses = sweep(m, b, "lr")
        x, l, e, _ = lr_fits[b]
        print(f"  B={b:>2}: " + "  ".join(f"{lr:g}:{loss:.4f}" for lr, loss in zip(xs, losses))
              + f"  -> lr* {x:.5f}, loss* {l:.4f}{' (best point on grid edge)' if e else ''}; best measured {losses.min():.4f}")
    print(f"  hypothesis (i): lr* = {lr_law[0]:.4g} * B^{lr_law[1]:.3f}")

    print("\n(b) LR = .0015: loss vs. WD")
    for b in BATCHES:
        xs, losses = sweep(m, b, "wd")
        x, l, e, _ = wd_fits[b]
        steps = TOKENS // (b * 1024)
        print(f"  B={b:>2}: " + "  ".join(f"{wd:g}:{loss:.4f}" for wd, loss in zip(xs, losses))
              + f"  -> wd* {x:.4f}, loss* {l:.4f}{' (best sampled point is the grid edge: optimum not bracketed)' if e else ''}; "
                f"n = lr*wd*T/2 = {FIXED_LR * x * steps / 2:.2f}")
    print(f"  hypothesis (ii): wd* = {wd_law[0]:.4g} * B^{wd_law[1]:.3f}")

    print("\nBest measured (lr, wd) per batch over both sweeps:")
    for b in BATCHES:
        k = min((k for k in m if k[0] == b), key=m.get)
        print(f"  B={b:>2}: lr {k[1]:g}, wd {k[2]:g} -> {m[k]:.4f}")

    print("\nPredictions for the targets:")
    for b in TARGETS:
        print(f"  B={b}: (i) WD .1, LR {float(lr_law[2](b)):.5f}   (ii) LR .0015, WD {float(wd_law[2](b)):.3f}")

    plot_sweeps(m, "lr", lr_fits, lr_law, "p32a_lr.png", "peak LR", "Loss vs. LR", "WD = .1")
    plot_sweeps(m, "wd", wd_fits, wd_law, "p32b_wd.png", "weight decay", "Loss vs. WD", "LR = .0015")


if __name__ == "__main__":
    main()
