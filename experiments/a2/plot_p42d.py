"""Problem 4.2(d): depth transfer at width 512 (reference depth 8, r = L / 8) for standard muP, Depth-muP, and
CompleteP at depths 4 and 16. Depth 8 is the shared source (prescriptions coincide at r = 1): Problem 1's curve
at 153.6M tokens plus our width-512 LR .00075 point. Fits each loss-LR curve with the 4.1/4.2(a) local quadratic.
Writes plots/p42d_depth_transfer.png.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.mup import BASE_DEPTH, DEPTH_KEY, SOURCE_LR
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p42a import curves as width_curves, fit

RULES = {"mup": ("standard muP", "C0"), "depth-mup": ("Depth-muP", "C2"), "completep": ("CompleteP", "C1")}


def curves():
    """{(rule, depth): {lr: final val loss}}; depth 8 is the shared width-512 source curve."""
    out = defaultdict(dict)
    for r in load_runs([DEPTH_KEY]):
        if r.state != "finished":
            continue
        rule = r.name.split(f"{DEPTH_KEY}-", 1)[1]
        depth = int(r.name.split("model-a2-d", 1)[1].split("-", 1)[0])
        out[(rule, depth)][r.config["learning_rate"]] = r.val_loss
    source = width_curves()[("mup", 512)]
    for rule in RULES:
        out[(rule, BASE_DEPTH)] = dict(source)
    return out


def main():
    c = curves()
    depths = sorted({d for _, d in c})
    fits = {k: fit(v) for k, v in c.items()}
    print(f"{'':>14} {'depth':>5}  LR* (fit)  loss*   best sampled      loss at source LR {SOURCE_LR:g}")
    for rule, (label, _) in RULES.items():
        for d in depths:
            lrs, losses, lr_opt, loss_opt, edge, _ = fits[(rule, d)]
            i = int(np.argmin(losses))
            at_src = c[(rule, d)][SOURCE_LR]
            print(f"{label:>14} {d:>5}  {lr_opt:9.4g}  {loss_opt:.4f}  {lrs[i]:g}: {losses[i]:.4f}   "
                  f"{at_src:.4f} (+{at_src - losses[i]:.4f})" + ("  [best point on grid edge]" if edge else ""))
        print("   " + "  ".join(f"L{d}: " + " ".join(f"{lr:g}:{l:.4f}" for lr, l in zip(*fits[(rule, d)][:2]))
                               for d in depths))

    fig, axes = plt.subplots(1, len(depths) + 2, figsize=(5.2 * (len(depths) + 2), 4.6))
    for ax, d in zip(axes, depths):
        for rule, (label, color) in RULES.items():
            if d == BASE_DEPTH and rule != "mup":
                continue
            lrs, losses, lr_opt, loss_opt, edge, _ = fits[(rule, d)]
            ax.plot(lrs, losses, "o-", color="0.3" if d == BASE_DEPTH else color, mec="k", ms=6, lw=1,
                    label=("shared source (r = 1)" if d == BASE_DEPTH else label)
                    + f": LR* {lr_opt:.2g}{' (edge)' if edge else ''}, loss* {loss_opt:.3f}")
            ax.plot([lr_opt], [loss_opt], "*", color="0.3" if d == BASE_DEPTH else color, mec="k", ms=15,
                    mfc="white" if edge else None)
        ax.axvline(SOURCE_LR, color="0.5", ls="--", lw=1, label=f"source LR {SOURCE_LR:g}")
        all_lrs = sorted({lr for r in RULES for lr in fits[(r, d)][0]})
        ax.set(xscale="log", xlabel="peak LR", ylabel="final val loss", title=f"depth {d} (r = {d / BASE_DEPTH:g})")
        ax.set_xticks(all_lrs, labels=[f"{lr:g}" for lr in all_lrs], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.legend(fontsize=7)
    ax_lr, ax_loss = axes[-2], axes[-1]
    for rule, (label, color) in RULES.items():
        lr_opt = [fits[(rule, d)][2] for d in depths]
        edge = [fits[(rule, d)][4] for d in depths]
        ax_lr.plot(depths, lr_opt, "-", color=color, label=label)
        for d, y, e in zip(depths, lr_opt, edge):
            ax_lr.plot([d], [y], "o", color=color, mec="k", mfc="white" if e else color)
        ax_loss.plot(depths, [fits[(rule, d)][3] for d in depths], "o-", color=color, mec="k",
                     label=f"{label}: fitted minimum")
        ax_loss.plot(depths, [c[(rule, d)][SOURCE_LR] for d in depths], "s--", color=color, mfc="white",
                     label=f"{label}: at source LR {SOURCE_LR:g}")
    ax_lr.axhline(SOURCE_LR, color="0.5", ls="--", lw=1)
    ax_lr.set(xscale="log", yscale="log", xlabel="depth L", ylabel="fitted optimal peak LR",
              title="Optimal LR vs. depth (hollow = best point on grid edge)")
    ax_lr.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_lr.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_loss.set(xscale="log", xlabel="depth L", ylabel="final val loss", title="Tuned vs. transferred loss")
    for ax in (ax_lr, ax_loss):
        ax.set_xticks(depths, labels=[str(d) for d in depths])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "p42d_depth_transfer.png")


if __name__ == "__main__":
    main()
