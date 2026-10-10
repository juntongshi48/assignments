"""Problem 4.1(c): depth transfer of the base LR for standard muP, Depth-muP, and CompleteP after five updates.

Width 64 (reference width 64), reference depth 2 (one shared source sweep: the prescriptions coincide at r = 1).
Downloads p41d-*.json from the Modal volume into cache/p41d/ and fits each loss-LR curve as in 4.1(a).
Writes plots/p41c_depth_transfer.png.
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p41a import download, plot_curve

CACHE_NAME = "p41d"
PRESCRIPTIONS = {"mup": ("standard muP", "C0"), "depth-mup": ("Depth-muP", "C2"), "completep": ("CompleteP", "C1")}


FIT_RANGE = (.003, .1)  # fit one quadratic in log LR through every sampled point in this range (7 points)


def fit(curve, fit_range=FIT_RANGE):
    """(lrs, losses, LR*, loss at LR*, edge, (c0, c1, c2)) from a quadratic in log LR over fit_range.

    edge flags a vertex outside the fitted range or a best in-range point on the range boundary."""
    lrs = np.array(sorted(curve))
    losses = np.array([curve[lr] for lr in lrs])
    use = np.isfinite(losses) & (lrs >= fit_range[0] * .999) & (lrs <= fit_range[1] * 1.001)
    x, y = np.log(lrs[use]), losses[use]
    c2, c1, c0 = np.polyfit(x, y, 2)
    best = int(np.argmin(y))
    if c2 <= 0:
        return lrs, losses, float(lrs[use][best]), float(y[best]), True, (c0, c1, c2)
    v = -c1 / (2 * c2)
    edge = not (x.min() <= v <= x.max()) or best in (0, len(y) - 1)
    return lrs, losses, float(np.exp(v)), float(c0 + c1 * v + c2 * v * v), edge, (c0, c1, c2)


def load():
    """{(prescription, depth): {lr: final val loss (inf if diverged)}}; depth 2 is copied to every prescription."""
    cache = download(CACHE_NAME, "p41d-")
    curves, records = defaultdict(dict), {}
    for path in sorted(cache.glob("p41d-*.json")):
        r = json.loads(path.read_text())
        if path.name.endswith(".diverged.json"):
            curves[(r["policy"], r["depth"])][r["base_lr"]] = np.inf
            continue
        policy, depth = r["config"].get("policy"), r["config"]["depth"]
        policy = policy or path.stem.split("-L")[0].removeprefix("p41d-")
        curves[(policy, depth)][r["base_lr"]] = r["history"][-1]["val_loss"]
        records[(policy, depth, r["base_lr"])] = r
    for policy in PRESCRIPTIONS:  # r = 1: the prescriptions coincide at the reference depth
        if ("mup", 2) in curves:
            curves[(policy, 2)] = curves[("mup", 2)]
    return curves, records


def main():
    curves, _ = load()
    depths = sorted({d for _, d in curves})
    fits = {}
    for policy, (label, _) in PRESCRIPTIONS.items():
        for depth in depths:
            if (policy, depth) not in curves:
                continue
            lrs, losses, lr_opt, loss_opt, edge, _ = fits[(policy, depth)] = fit(curves[(policy, depth)])
            pts = "  ".join(f"{lr:g}:{'div' if not np.isfinite(l) else f'{l:.3f}'}" for lr, l in zip(lrs, losses))
            print(f"{label:>12} L={depth:<5} {pts}\n{'':>20}-> LR* {lr_opt:.4g} (quadratic over {FIT_RANGE[0]:g}-{FIT_RANGE[1]:g}), best measured "
                  f"{np.nanmin(np.where(np.isfinite(losses), losses, np.nan)):.4f} at {lrs[np.nanargmin(np.where(np.isfinite(losses), losses, np.nan))]:g}"
                  + ("  (best point on grid edge)" if edge else ""))

    shown = [d for d in depths if d != 2]
    fig, axes = plt.subplots(1, len(shown) + 3, figsize=(5.2 * (len(shown) + 3), 4.6))
    plot_curve(axes[0], *fits[("mup", 2)], "0.3", fit_range=FIT_RANGE)
    axes[0].set_title(f"depth 2 (shared source, r = 1)\nLR* = {fits[('mup', 2)][2]:.2g}, "
                      f"best measured = {np.min(fits[('mup', 2)][1]):.3f}", fontsize=10)
    axes[0].set_ylabel("val loss after 5 updates")
    for ax, depth in zip(axes[1:], shown):
        for policy, (label, color) in PRESCRIPTIONS.items():
            if (policy, depth) not in fits:
                continue
            lrs, losses, lr_opt, loss_opt, edge, _ = fits[(policy, depth)]
            ok = np.isfinite(losses)
            ax.plot(lrs[ok], losses[ok], "o-", color=color, mec="k", ms=6, lw=1,
                    label=f"{label}: LR* {lr_opt:.2g}{' (edge)' if edge else ''}, best {np.nanmin(losses[ok]):.3f}")
            ax.plot([lr_opt], [loss_opt], "*", color=color, mec="k", ms=15, mfc="white" if edge else color)
            c0, c1, c2 = fits[(policy, depth)][5]
            g = np.geomspace(*FIT_RANGE, 60)
            ax.plot(g, c0 + c1 * np.log(g) + c2 * np.log(g) ** 2, "--", color=color, lw=1)
        all_lrs = sorted({lr for p in PRESCRIPTIONS if (p, depth) in fits for lr in fits[(p, depth)][0]})
        ax.set(xscale="log", xlabel="base LR", title=f"depth {depth} (r = {depth // 2})")
        ax.set_xticks(all_lrs, labels=[f"{lr:.2g}" for lr in all_lrs], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.legend(fontsize=8)
    ax_lr, ax_loss = axes[-2], axes[-1]
    for policy, (label, color) in PRESCRIPTIONS.items():
        ds = [d for d in depths if (policy, d) in fits]
        lr_opt = [fits[(policy, d)][2] for d in ds]
        # Best measured loss: on a x3 LR grid the 3-point parabola's minimum undershoots every sampled point.
        loss_opt = [np.nanmin(np.where(np.isfinite(fits[(policy, d)][1]), fits[(policy, d)][1], np.nan)) for d in ds]
        edge = [fits[(policy, d)][4] for d in ds]
        ax_lr.plot(ds, lr_opt, "-", color=color, label=label)
        for d, y, e in zip(ds, lr_opt, edge):
            ax_lr.plot([d], [y], "o", color=color, mec="k", mfc="white" if e else color)
        ax_loss.plot(ds, loss_opt, "o-", color=color, mec="k", label=label)
    ax_lr.set(xscale="log", yscale="log", xlabel="depth L", ylabel="fitted optimal base LR",
              title="Optimal base LR vs. depth (hollow = best point on grid edge)")
    ax_lr.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_lr.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
    ax_loss.set(xscale="log", xlabel="depth L", ylabel="best measured val loss", title="Best measured loss vs. depth")
    for ax in (ax_lr, ax_loss):
        ax.set_xticks(depths, labels=[str(d) for d in depths])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p41c_depth_transfer.png")


if __name__ == "__main__":
    main()
