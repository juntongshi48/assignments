"""Problem 4.2(c): predict the held-out width 1024.

For each prescription, fit LR*(n) = A n^p to the 4.2(b) fitted optima at widths 128, 256, 512 and predict LR* at
1024; compare the predicted LR and the directly transferred source LR (.003) with the fitted local optimum of the
width-1024 sweep (local quadratic in log LR, as in 4.2(b)), and report each one's loss gap relative to the best
sampled width-1024 LR. Writes plots/p42c_heldout.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.mup import HELDOUT_WIDTH, SOURCE_LR
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p42a import PRESCRIPTIONS, curves, fit

SOURCE_WIDTHS = (128, 256, 512)


def main():
    c = curves()
    fig, axes = plt.subplots(1, 4, figsize=(22.5, 4.8))
    all_widths = (*SOURCE_WIDTHS, HELDOUT_WIDTH)
    print(f"{'':>22} {'law':>22} {'pred':>8} {'LR* 1024':>9}  pred/LR*  src/LR*  "
          f"best sampled      gap(pred)  gap(src)")
    for p, (label, color) in PRESCRIPTIONS.items():
        opt = np.array([fit(c[(p, w)])[2] for w in SOURCE_WIDTHS])
        slope, icpt = np.polyfit(np.log(SOURCE_WIDTHS), np.log(opt), 1)
        law = lambda n: np.exp(icpt) * np.asarray(n, dtype=float) ** slope
        pred = float(law(HELDOUT_WIDTH))
        lrs, losses, lr_opt, loss_opt, edge, (c0, c1, c2) = fit(c[(p, HELDOUT_WIDTH)])
        best = int(np.argmin(losses))
        pred_lr = min(lrs, key=lambda lr: abs(np.log(lr / pred)))  # the run launched at the (rounded) prediction
        gap_pred = c[(p, HELDOUT_WIDTH)][pred_lr] - losses[best]
        gap_src = c[(p, HELDOUT_WIDTH)][SOURCE_LR] - losses[best]
        print(f"{label:>22} {f'{np.exp(icpt):.3g} n^{slope:.3f}':>22} {pred:8.5f} {lr_opt:9.5f}  {pred / lr_opt:8.2f} "
              f"{SOURCE_LR / lr_opt:8.2f}  {lrs[best]:g}: {losses[best]:.4f}   {gap_pred:+.4f}   {gap_src:+.4f}"
              + ("  [edge]" if edge else ""))
        print("   w1024: " + "  ".join(f"{lr:g}:{l:.4f}" for lr, l in zip(lrs, losses)))

        ax = axes[0 if p == "sp" else 1]
        ax.plot(lrs, losses, "o-", color=color, mec="k", ms=6, lw=1, label="width 1024 sweep")
        g = np.geomspace(lrs[max(best - 1, 0)], lrs[min(best + 1, len(lrs) - 1)], 50)
        ax.plot(g, c0 + c1 * np.log(g) + c2 * np.log(g) ** 2, "--", color="0.4", lw=1)
        ax.plot([lr_opt], [loss_opt], "*", color="gold", mec="k", ms=16, label=f"fitted local optimum {lr_opt:.2g}")
        ax.axvline(pred, color=color, ls=":", lw=1.5, label=f"power-law prediction {pred:.2g} (gap {gap_pred:+.4f})")
        ax.axvline(SOURCE_LR, color="0.5", ls="--", lw=1, label=f"transferred {SOURCE_LR:g} (gap {gap_src:+.4f})")
        ax.set(xscale="log", xlabel="peak LR", ylabel="final val loss", title=f"{label}, held-out width {HELDOUT_WIDTH}")
        ax.set_xticks(lrs, labels=[f"{lr:.2g}" for lr in lrs], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.legend(fontsize=7)

        grid = np.geomspace(100, 1300, 50)
        axes[2].plot(SOURCE_WIDTHS, opt, "o", color=color, mec="k", label=f"{label}: fitted LR* (128-512)")
        axes[2].plot(grid, law(grid), "-", color=color, lw=1, label=f"{label}: {np.exp(icpt):.3g} n^{slope:.2f}")
        axes[2].plot([HELDOUT_WIDTH], [lr_opt], "*", color=color, mec="k", ms=15, label=f"{label}: measured LR* at 1024")

        # Tuned vs. transferred loss across all widths (as in p42a's last panel), plus the prediction at 1024.
        tuned = [fit(c[(p, w)])[3] for w in all_widths]
        at_src = [c[(p, w)][SOURCE_LR] for w in all_widths]
        axes[3].plot(all_widths, tuned, "o-", color=color, mec="k", label=f"{label}: fitted minimum")
        axes[3].plot(all_widths, at_src, "s--", color=color, mfc="white", label=f"{label}: at source LR {SOURCE_LR:g}")
        axes[3].plot([HELDOUT_WIDTH], [c[(p, HELDOUT_WIDTH)][pred_lr]], "D", color=color, mec="k", ms=7,
                     label=f"{label}: at predicted LR {pred_lr:.2g} (1024)")
    axes[2].axhline(SOURCE_LR, color="0.5", ls="--", lw=1, label=f"transferred {SOURCE_LR:g}")
    axes[2].set(xscale="log", yscale="log", xlabel="width", ylabel="optimal peak LR",
                title="Width trend fitted on 128-512, tested at 1024")
    axes[2].set_xticks((*SOURCE_WIDTHS, HELDOUT_WIDTH), labels=[str(w) for w in (*SOURCE_WIDTHS, HELDOUT_WIDTH)])
    axes[2].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axes[2].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    axes[2].yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    axes[2].legend(fontsize=7)
    axes[3].axvspan(768, 1300, color="0.92", zorder=0)
    axes[3].set(xscale="log", xlabel="width", ylabel="final val loss",
                title="Tuned vs. transferred loss (shaded: held-out width)")
    axes[3].set_xticks(all_widths, labels=[str(w) for w in all_widths])
    axes[3].set_xlim(100, 1300)
    axes[3].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axes[3].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    savefig(fig, "p42c_heldout.png")


if __name__ == "__main__":
    main()
