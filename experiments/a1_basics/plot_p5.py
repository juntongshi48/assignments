"""Problem 5 analysis: loss-curve macrostructure and microstructure across all d8 runs.

Macro: 201-step moving average of the per-step train loss.
Micro: residual r_t = L_t - moving average; roughness = SD(r_t) over 30-70% of training,
and spikes = steps with r_t above 5 robust SDs. Minibatch sampling alone scales roughness
by ~1/sqrt(batch size), so roughness is also shown rescaled to bs64.
Writes plots/p5_*.png and prints a roughness table."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import history, load_runs, savefig
from train import TrainConfig

WINDOW = 201
MID = (0.3, 0.7)
LABEL_KEYS = ("learning_rate", "batch_size", "warmup_percent", "weight_decay", "lr_schedule",
              "grad_norm", "beta1", "beta2", "qk_norm")
SHORT = {"learning_rate": "lr", "batch_size": "bs", "warmup_percent": "warmup", "weight_decay": "wd",
         "lr_schedule": "", "grad_norm": "clip", "beta1": "b1", "beta2": "b2", "qk_norm": "qknorm"}


def label(run):
    defaults = TrainConfig()
    parts = []
    for key in LABEL_KEYS:
        value, default = run.config.get(key), getattr(defaults, key)
        if value is None and default is None or value == default:
            continue
        if isinstance(value, float) and isinstance(default, float) and np.isclose(value, default):
            continue
        parts.append(f"{SHORT[key]}{value}" if key != "qk_norm" else ("no-qknorm" if not value else ""))
    return " ".join(parts) or "default"


def moving_average(values, window=WINDOW):
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def micro_stats(steps, losses, total_steps):
    residual = losses - moving_average(losses)
    progress = (steps + 1) / total_steps
    mid = (progress >= MID[0]) & (progress <= MID[1])
    r = residual[mid]
    robust_sd = 1.4826 * np.median(np.abs(r - np.median(r)))
    return dict(roughness=float(r.std()), spikes=int((residual[progress > 0.05] > 5 * robust_sd).sum()),
                residual=residual, progress=progress)


def main():
    runs = [run for run in load_runs(["p1a-v1", "p1b-v1", "p1c-v1", "p5-v1", "p6-v1"])
            if run.state == "finished" and run.depth == 8]
    stats = {}
    for run in runs:
        h = history(run, ["train_loss"])
        steps, losses = h["optimizer_step"], h["train_loss"]
        total = float(run.config["total_steps"])
        stats[run.name] = dict(run=run, label=label(run), steps=steps, losses=losses,
                               smooth=moving_average(losses), **micro_stats(steps, losses, total))
    by_label = {entry["label"]: entry for entry in stats.values()}

    families = {
        "Learning rate (bs64)": ["lr0.0003", "lr0.001", "default", "lr0.01", "lr0.03"],
        "Batch size (lr 3e-3)": ["bs16", "bs32", "default", "bs128", "bs256"],
        "Adam betas": ["b10", "b10.5", "default", "b10.95", "b20.99"],  # beta1=0 is logged as int 0
        "Schedule (warmup 0.1)": ["warmup0.1", "warmup0.1 cos", "warmup0.1 wsd0.2", "warmup0.1 wsd0.0"],
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for col, (title, labels) in enumerate(families.items()):
        colors = mpl.colormaps["viridis"](np.linspace(0, 0.9, len(labels)))
        for color, name in zip(colors, labels):
            entry = by_label.get(name)
            if entry is None:
                print(f"  missing in family {title!r}: {name}")
                continue
            progress = (entry["steps"] + 1) / float(entry["run"].config["total_steps"])
            axes[0, col].plot(progress, entry["smooth"], color=color, lw=1.6, label=f"{name} ({entry['run'].val_loss:.3f})")
            # Micro: raw loss in a 300-step window at 50% of training, mean-removed.
            mid = np.searchsorted(progress, 0.5)
            sl = slice(mid, mid + 300)
            axes[1, col].plot(np.arange(300)[: len(entry["losses"][sl])], entry["losses"][sl] - entry["smooth"][sl],
                              color=color, lw=0.8, alpha=0.85,
                              label=f"{name} (rough {entry['roughness']:.3f})")
        axes[0, col].set_ylim(2.85, 3.8)
        axes[0, col].set_title(f"Macro: {title}")
        axes[0, col].set_xlabel("fraction of training")
        axes[0, col].set_ylabel(f"train loss ({WINDOW}-step mean)")
        axes[1, col].set_title("Micro: raw - smoothed, 300 steps at 50%")
        axes[1, col].set_xlabel("steps")
        axes[1, col].set_ylabel("residual")
        for ax in axes[:, col]:
            ax.grid(True, ls=":", alpha=0.3)
            ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "p5_macro_micro.png")
    plt.close(fig)

    rows = sorted(stats.values(), key=lambda entry: entry["roughness"])
    print(f"\nRoughness over {MID[0]:.0%}-{MID[1]:.0%} of training (SD of raw - {WINDOW}-step mean)")
    print(f"{'config':<32} {'bs':>4} {'rough':>7} {'rough*sqrt(bs/64)':>18} {'spikes':>6} {'val':>7}")
    for entry in rows:
        bs = int(entry["run"].config["batch_size"])
        print(f"{entry['label']:<32} {bs:>4} {entry['roughness']:7.4f} {entry['roughness'] * np.sqrt(bs / 64):18.4f} "
              f"{entry['spikes']:6d} {entry['run'].val_loss:7.4f}")

    fig, ax = plt.subplots(figsize=(11, 0.28 * len(rows) + 1.5))
    names = [entry["label"] for entry in rows]
    normalized = [entry["roughness"] * np.sqrt(int(entry["run"].config["batch_size"]) / 64) for entry in rows]
    ax.barh(names, [entry["roughness"] for entry in rows], color="#9ecae1", label="roughness")
    ax.scatter(normalized, names, color="#d62728", zorder=3, s=18, label="roughness x sqrt(bs/64)")
    ax.set_xlabel("SD of per-step train loss around its moving average (30-70% of training)")
    ax.set_title("Loss-curve microstructure across all d8 runs")
    ax.grid(True, axis="x", ls=":", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "p5_roughness.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
