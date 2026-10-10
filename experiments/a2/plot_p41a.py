"""Problem 4.1(a): width transfer of the base LR for Kaiming vs. standard muP after five updates.

Downloads the p41w-*.json outputs from the Modal volume into cache/p41w/, then fits each loss-LR curve with a
quadratic in log(LR) through the best point and its neighbours (flagged when the best point is on the grid edge).
Diverged runs (nonfinite loss) count as missing points on the high-LR side.
Writes plots/p41a_width_transfer.png.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p32ab import local_optimum
from modal_utils import MODAL_ENVIRONMENT, VOLUME_NAME

CACHE = Path(__file__).resolve().parent / "cache" / "p41w"
POLICIES = {"kaiming": ("Kaiming", "C3"), "mup": ("standard muP", "C0")}


def download(cache_name="p41w", prefix="p41w-"):
    """Fetch every `prefix`*.json output not yet cached (the volume is the source of truth); returns the cache dir."""
    cache = CACHE.parent / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["modal", "volume", "ls", VOLUME_NAME, "/a2-stress", "--env", MODAL_ENVIRONMENT],
                             capture_output=True, text=True, check=True).stdout.split()
    for remote in listing:
        name = Path(remote).name
        if name.startswith(prefix) and name.endswith(".json") and not (cache / name).exists():
            subprocess.run(["modal", "volume", "get", VOLUME_NAME, f"/a2-stress/{name}", str(cache / name),
                            "--env", MODAL_ENVIRONMENT], check=True, capture_output=True)
    return cache


def load():
    """{(policy, width): {lr: final val loss (inf if diverged)}} and the full records."""
    curves, records = defaultdict(dict), {}
    for path in sorted(CACHE.glob("p41w-*.json")):
        r = json.loads(path.read_text())
        if path.name.endswith(".diverged.json"):
            curves[(r["policy"], r["width"])][r["base_lr"]] = np.inf
            continue
        _, policy, w, _ = path.stem.split("-", 3)
        key = (policy, int(w[1:]))
        curves[key][r["base_lr"]] = r["history"][-1]["val_loss"]
        records[(*key, r["base_lr"])] = r
    return curves, records


def fit(curve):
    lrs = np.array(sorted(curve))
    losses = np.array([curve[lr] for lr in lrs])
    ok = np.isfinite(losses)
    lr_opt, loss_opt, edge, coefs = local_optimum(lrs[ok], losses[ok])
    return lrs, losses, lr_opt, loss_opt, edge, coefs


def main():
    download()
    curves, _ = load()
    widths = sorted({w for _, w in curves})
    fits = {}
    for (policy, width), curve in sorted(curves.items()):
        lrs, losses, lr_opt, loss_opt, edge, _ = fits[(policy, width)] = fit(curve)
        pts = "  ".join(f"{lr:g}:{'div' if not np.isfinite(l) else f'{l:.3f}'}" for lr, l in zip(lrs, losses))
        print(f"{POLICIES[policy][0]:>12} w={width:<5} {pts}\n{'':>20}-> LR* {lr_opt:.4g}, loss* {loss_opt:.4f}"
              + ("  (best point on grid edge)" if edge else ""))

    fig, axes = plt.subplots(len(POLICIES), len(widths) + 1, figsize=(5.2 * (len(widths) + 1), 4.4 * len(POLICIES)))
    for row, (policy, (label, color)) in enumerate(POLICIES.items()):
        for ax, width in zip(axes[row], widths):
            if (policy, width) not in fits:
                ax.set_visible(False)
                continue
            optimum = plot_curve(ax, *fits[(policy, width)], color)
            ax.set_title(f"{label}, width {width}\n{optimum}", fontsize=10)
            ax.set_ylabel("val loss after 5 updates")
    plot_summary(axes[0][-1], axes[1][-1], fits, widths)
    fig.suptitle("Loss after 5 updates vs. base LR (star: fitted optimum from a quadratic in log LR through the best "
                 "point and its neighbours, dashed; triangles: off-scale points)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p41a_width_transfer.png")


def plot_curve(ax, lrs, losses, lr_opt, loss_opt, edge, coefs, color, fit_range=None):
    """Sampled LRs as ticks, local quadratic fit, optimum star; off-scale points as labelled arrows at the top."""
    ok = np.isfinite(losses)
    top = min(np.nanmax(losses[ok]), loss_opt + 2.5) + .1
    shown = ok & (losses <= top)
    ax.plot(lrs[shown], losses[shown], "o-", color=color, mec="k", ms=6, lw=1)
    for lr, loss in zip(lrs[~shown], losses[~shown]):
        ax.plot([lr], [top - .03], "^", color=color, mec="k", ms=8)
        ax.annotate("diverged" if not np.isfinite(loss) else f"{loss:.1f}", (lr, top - .03),
                    textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    c0, c1, c2 = coefs
    i = int(np.argmin(np.where(ok, losses, np.inf)))
    window = lrs[max(i - 1, 0):i + 2] if fit_range is None else np.array(fit_range)
    if c2 > 0:
        g = np.geomspace(window.min(), window.max(), 50)
        ax.plot(g, c0 + c1 * np.log(g) + c2 * np.log(g) ** 2, color="0.4", lw=1, ls="--")
    ax.plot([lr_opt], [loss_opt], "*", color="gold", mec="k", ms=17, zorder=4)
    ax.set(xscale="log", xlabel="base LR", ylim=(loss_opt - .1, top))
    ax.set_xticks(lrs, labels=[f"{lr:.2g}" for lr in lrs], rotation=45, fontsize=7)
    ax.minorticks_off()
    return f"LR* = {lr_opt:.2g}, loss* = {loss_opt:.3f}" + (" (edge)" if edge else "")


def plot_summary(ax_lr, ax_loss, fits, widths):
    w = np.array(widths, dtype=float)
    for policy, (label, color) in POLICIES.items():
        ws = np.array([x for x in widths if (policy, x) in fits], dtype=float)
        lr_opt = np.array([fits[(policy, int(x))][2] for x in ws])
        loss_opt = np.array([fits[(policy, int(x))][3] for x in ws])
        for ax, ys, fmt in ((ax_lr, lr_opt, "{:.2g}"), (ax_loss, loss_opt, "{:.3f}")):
            ax.plot(ws, ys, "o-", color=color, mec="k", label=label)
            for x, y in zip(ws, ys):
                ax.annotate(fmt.format(y), (x, y), textcoords="offset points", fontsize=8, color=color,
                            xytext=(-8, 6) if x == ws[-1] else (6, 6), ha="right" if x == ws[-1] else "left")
        if policy == "kaiming":
            for slope, style in ((1, ":"), (1.5, "-.")):
                ax_lr.plot(w, lr_opt[0] * (w / w[0]) ** -slope, style, color="0.5", lw=1,
                           label=rf"$\propto m^{{-{slope:g}}}$ through Kaiming at {widths[0]}")
    ax_lr.set(xscale="log", yscale="log", xlabel="width", ylabel="fitted optimal base LR",
              title="Optimal base LR vs. width")
    ax_loss.set(xscale="log", xlabel="width", ylabel="fitted minimum val loss", title="Minimum loss vs. width")
    for ax in (ax_lr, ax_loss):
        ax.set_xticks(widths, labels=[str(x) for x in widths])
        ax.minorticks_off()
        ax.legend(fontsize=8)


if __name__ == "__main__":
    main()
