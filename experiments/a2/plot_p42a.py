"""Problem 4.2(a, b): width transfer of Problem 1's best sampled LR (.003 at 153.6M tokens) under the course
baseline (SP) and muP, at widths 128 and 256, against Problem 1's supplied width-512 curve (shared: m = 1).

Fits each loss-LR curve with a quadratic in log LR through the best point and its neighbours (4.1 convention).
Writes plots/p42a_width_transfer.png.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.mup import EXPERIMENT_KEY, SOURCE_LR, TOKENS
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p32ab import local_optimum
from experiments.a2.provided_sweeps import load

PRESCRIPTIONS = {"sp": ("course baseline (SP)", "C3"), "mup": ("muP", "C0")}
SOURCE_WIDTH = 512


def curves():
    """{(prescription, width): {lr: final val loss}}; width 512 is Problem 1's supplied curve for both."""
    out = defaultdict(dict)
    for r in load_runs([EXPERIMENT_KEY]):
        if r.state != "finished":
            continue
        prescription = r.name.rsplit("-", 1)[-1]
        width = int(r.name.split("-w", 1)[1].split("-", 1)[0])
        for p in (PRESCRIPTIONS if width == SOURCE_WIDTH else (prescription,)):  # m = 1: shared
            out[(p, width)][r.config["learning_rate"]] = r.val_loss
    for row in load("P1a"):
        if row["tokens"] == TOKENS and row["batch_size"] == 64 and row["weight_decay"] == .1:
            for p in PRESCRIPTIONS:
                out[(p, SOURCE_WIDTH)][row["learning_rate"]] = row["final_val_loss"]
    return out


def fit(curve):
    lrs = np.array(sorted(curve))
    losses = np.array([curve[lr] for lr in lrs])
    lr_opt, loss_opt, edge, coefs = local_optimum(lrs, losses)
    return lrs, losses, lr_opt, loss_opt, edge, coefs


def main():
    c = {k: v for k, v in curves().items() if k[1] <= SOURCE_WIDTH}  # width 1024 is held out for 4.2(c)
    widths = sorted({w for _, w in c})
    fits = {k: fit(v) for k, v in c.items()}
    print(f"{'':>22} {'width':>5}  LR* (fit)  loss*   best sampled      loss at source LR {SOURCE_LR:g}")
    for p, (label, _) in PRESCRIPTIONS.items():
        for w in widths:
            lrs, losses, lr_opt, loss_opt, edge, _ = fits[(p, w)]
            i = int(np.argmin(losses))
            at_src = c[(p, w)].get(SOURCE_LR)
            print(f"{label:>22} {w:>5}  {lr_opt:9.4g}  {loss_opt:.4f}  {lrs[i]:g}: {losses[i]:.4f}   "
                  f"{at_src:.4f} (+{at_src - losses[i]:.4f} vs best sampled)" + ("  [edge]" if edge else ""))
        print("   " + "  ".join(f"w{w}: " + " ".join(f"{lr:g}:{l:.4f}" for lr, l in zip(*fits[(p, w)][:2]))
                               for w in widths))

    fig, axes = plt.subplots(1, len(widths) + 2, figsize=(5.2 * (len(widths) + 2), 4.6))
    for ax, w in zip(axes, widths):
        for p, (label, color) in PRESCRIPTIONS.items():
            lrs, losses, lr_opt, loss_opt, edge, (c0, c1, c2) = fits[(p, w)]
            ls = ":" if w == SOURCE_WIDTH and p == "mup" else "-"
            ax.plot(lrs, losses, "o" + ls, color=color, mec="k", ms=6, lw=1,
                    label=f"{label}: LR* {lr_opt:.2g}, loss* {loss_opt:.3f}" + (" (edge)" if edge else ""))
            ax.plot([lr_opt], [loss_opt], "*", color=color, mec="k", ms=15, mfc="white" if edge else color)
        ax.axvline(SOURCE_LR, color="0.5", ls="--", lw=1, label=f"source LR {SOURCE_LR:g}")
        all_lrs = sorted({lr for p in PRESCRIPTIONS for lr in fits[(p, w)][0]})
        ax.set(xscale="log", xlabel="peak LR", ylabel="final val loss",
               title=f"width {w}" + (" (Problem 1 source; m = 1, shared)" if w == SOURCE_WIDTH else f" (m = {w / 512:g})"))
        ax.set_xticks(all_lrs, labels=[f"{lr:g}" for lr in all_lrs], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.legend(fontsize=7)
    ax_lr, ax_loss = axes[-2], axes[-1]
    for p, (label, color) in PRESCRIPTIONS.items():
        lr_opt = [fits[(p, w)][2] for w in widths]
        loss_opt = [fits[(p, w)][3] for w in widths]
        at_src = [c[(p, w)][SOURCE_LR] for w in widths]
        ax_lr.plot(widths, lr_opt, "o-", color=color, mec="k", label=label)
        ax_loss.plot(widths, loss_opt, "o-", color=color, mec="k", label=f"{label}: fitted minimum")
        ax_loss.plot(widths, at_src, "s--", color=color, mfc="white", label=f"{label}: at source LR {SOURCE_LR:g}")
    ax_lr.axhline(SOURCE_LR, color="0.5", ls="--", lw=1, label="source LR")
    ax_lr.set(xscale="log", yscale="log", xlabel="width", ylabel="fitted optimal peak LR",
              title="Optimal LR vs. width")
    ax_lr.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_lr.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_loss.set(xscale="log", xlabel="width", ylabel="final val loss", title="Tuned vs. transferred loss")
    for ax in (ax_lr, ax_loss):
        ax.set_xticks(widths, labels=[str(w) for w in widths])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "p42a_width_transfer.png")


if __name__ == "__main__":
    main()
