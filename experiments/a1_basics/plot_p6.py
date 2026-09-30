"""Problem 6 analysis: parameter / gradient / activation RMS across depth and training,
and how hyperparameters move them (ratio to the default run over training).
Stats come from module_rms_logging (every 100 steps; gradients are pre-clip).
Writes plots/p6_*.png and prints a start/mid/end ratio table."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.plot_p5 import label
from experiments.a1_basics.results import history, load_runs, savefig

LAYERS = range(8)
PER_LAYER = {
    "residual stream (block output)": "logging/rms/model.layers.{i}/activation",
    "query projection output (pre QK-norm)": "logging/rms/model.layers.{i}.self_attn.q_proj/activation",
    "MLP down_proj gradient": "logging/rms/model.layers.{i}.mlp.down_proj/gradient",
    "MLP up_proj parameter": "logging/rms/model.layers.{i}.mlp.up_proj/parameter",
}
GLOBAL = {
    "global parameter RMS": "logging/rms/global/parameter",
    "global gradient RMS (pre-clip)": "logging/rms/global/gradient",
    "final residual (layer 8 output)": "logging/rms/model.layers.7/activation",
    "query proj. output, layer 8": "logging/rms/model.layers.7.self_attn.q_proj/activation",
    "logits (lm_head output)": "logging/rms/lm_head/activation",
}
FAMILIES = {
    "Learning rate": ["lr0.001", "lr0.01", "lr0.03"],
    "Weight decay": ["wd0", "wd0.01", "wd0.3", "wd1"],
    "Warmup / schedule": ["warmup0", "warmup0.1", "warmup0.1 wsd0.0", "warmup0.1 cos"],
    "Batch size / Adam": ["bs16", "bs256", "b10", "b20.99"],
    "Clipping / QK-norm": ["clip0.1", "clipNone", "lr0.01 clipNone", "no-qknorm"],
}
SNAPSHOTS = (0.0, 0.1, 0.5, 1.0)


def progress(h, run):
    return (h["optimizer_step"] + 1) / float(run.config["total_steps"])


def plot_depth_and_time(base):
    keys = [template.format(i=i) for template in PER_LAYER.values() for i in LAYERS]
    h = history(base, keys)
    p = progress(h, base)
    fig, axes = plt.subplots(1, len(PER_LAYER) + 1, figsize=(24, 4.6))
    snap_colors = mpl.colormaps["viridis"](np.linspace(0, 0.9, len(SNAPSHOTS)))
    print("P6(a) default d8: per-layer RMS at start / 10% / 50% / end")
    for ax, (title, template) in zip(axes, PER_LAYER.items()):
        for color, frac in zip(snap_colors, SNAPSHOTS):
            index = int(np.argmin(np.abs(p - frac)))
            values = [h[template.format(i=i)][index] for i in LAYERS]
            ax.plot([i + 1 for i in LAYERS], values, "o-", color=color, label=f"step {h['optimizer_step'][index]}")
        print(f"  {title:<40} " + " | ".join(
            " ".join(f"{h[template.format(i=i)][int(np.argmin(np.abs(p - frac)))]:.3g}" for i in (0, 3, 7))
            for frac in SNAPSHOTS) + "   (layers 1,4,8 per snapshot)")
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[-1]
    g = history(base, list(GLOBAL.values()))
    for title, key in GLOBAL.items():
        ax.plot(progress(g, base), g[key], lw=1.5, label=title)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of training")
    ax.set_title("global statistics over training", fontsize=10)
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("P6(a): default d8 run across depth and training (RMS, log scale)")
    fig.tight_layout()
    savefig(fig, "p6a_depth_time.png")
    plt.close(fig)


def plot_interventions(base, by_label):
    keys = list(GLOBAL.values())
    base_h = history(base, keys)
    base_p = progress(base_h, base)
    fig, axes = plt.subplots(len(GLOBAL), len(FAMILIES), figsize=(4.4 * len(FAMILIES), 3.0 * len(GLOBAL)), sharex=True)
    print("\nP6(c): statistic / default-run statistic at 10% | 50% | 100% of training")
    print(f"{'intervention':<18} " + "  ".join(f"{name[:22]:>24}" for name in GLOBAL))
    for col, (family, labels) in enumerate(FAMILIES.items()):
        colors = mpl.colormaps["plasma"](np.linspace(0, 0.85, len(labels)))
        for color, name in zip(colors, labels):
            run = by_label.get(name)
            if run is None:
                print(f"  missing: {name}")
                continue
            h = history(run, keys)
            p = progress(h, run)
            cells = []
            for row, (title, key) in enumerate(GLOBAL.items()):
                ratio = np.interp(base_p, p, h[key]) / base_h[key]
                axes[row, col].plot(base_p, ratio, color=color, lw=1.4, label=name)
                picks = [ratio[int(np.argmin(np.abs(base_p - frac)))] for frac in (0.1, 0.5, 1.0)]
                cells.append(" ".join(f"{value:7.2f}" for value in picks))
            print(f"{name:<18} " + "  ".join(f"{cell:>24}" for cell in cells))
        for row, title in enumerate(GLOBAL):
            ax = axes[row, col]
            ax.axhline(1.0, color="black", lw=0.8)
            ax.set_yscale("log")
            ax.grid(True, ls=":", alpha=0.3)
            if row == 0:
                ax.set_title(family)
                ax.legend(frameon=False, fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"{title}\n/ default", fontsize=9)
            if row == len(GLOBAL) - 1:
                ax.set_xlabel("fraction of training")
    fig.suptitle("P6(b,c): how interventions move internal statistics (ratio to the default run; 1 = unchanged)")
    fig.tight_layout()
    savefig(fig, "p6c_interventions.png")
    plt.close(fig)


def main():
    runs = [run for run in load_runs(["p1a-v1", "p1c-v1", "p5-v1", "p6-v1"]) if run.state == "finished" and run.depth == 8]
    by_label = {}
    for run in runs:
        by_label.setdefault(label(run), run)
    base = by_label["default"]
    plot_depth_and_time(base)
    plot_interventions(base, by_label)


if __name__ == "__main__":
    main()
