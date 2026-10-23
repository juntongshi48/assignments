"""Problem 2(a): four LR schedules (constant after warmup, linear, cosine, WSD 20%) at three peak LRs.

Reads the SCHEDULE_RUNS and LR_SWEEP_RUNS of p2_river_valley from W&B (finished runs are cached
under cache/p2/). Figure 1, one column per peak LR: learning rate, training loss (smoothed), and
validation loss against training tokens. Figure 2: final validation loss against peak LR per
schedule; the lowest is the reference run for part (b). Peak LR 7.5e-4 has only constant and WSD.
Writes plots/p2a_schedules.png and plots/p2a_final_vs_lr.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p2_river_valley import (
    EMA_DECAYS,
    EXPERIMENT_KEY,
    LOW_LR_RUNS,
    LR_SWEEP_RUNS,
    SCHEDULE_RUNS,
)
from experiments.a3_optimization.plot_p1a import savefig
from train import training_run_name

CACHE = Path(__file__).resolve().parent / "cache" / "p2"
TOKENS_PER_STEP = SCHEDULE_RUNS[0].batch_size * 1024
KEYS = ("learning_rate", "train_loss", "val_loss", *(f"val_ema{d:g}_loss" for d in EMA_DECAYS))
SCHEDULES = {"wsd0.0": "constant", "linear": "linear", "cos": "cosine", "wsd0.2": "WSD (20% decay)"}
COLORS = dict(zip(SCHEDULES, ("C3", "C0", "C2", "C1")))
RUNS = SCHEDULE_RUNS + LR_SWEEP_RUNS + LOW_LR_RUNS  # LOW_LR_RUNS: constant and WSD only


def fetch(configs=RUNS, keys=KEYS):
    """{run_name: dict(state, config, history={key: (steps, values)})}; finished runs are cached."""
    import wandb

    from utils import WANDB_ENTITY, WANDB_PROJECT

    CACHE.mkdir(parents=True, exist_ok=True)
    names = {training_run_name(c) for c in configs}
    out = {}
    for name in names:
        path = CACHE / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    missing = names - set(out)
    if missing:
        api = wandb.Api(timeout=120)
        latest = {}
        for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [EXPERIMENT_KEY]}}):
            if run.name in missing and (run.name not in latest or run.created_at > latest[run.name].created_at):
                latest[run.name] = run
        for name, run in latest.items():
            history = {k: ([], []) for k in keys}
            for row in run.scan_history(keys=None):
                step = row.get("optimizer_step")
                if step is None:
                    continue
                for k in keys:
                    if row.get(k) is not None:
                        history[k][0].append(step + 1)
                        history[k][1].append(row[k])
            out[name] = dict(state=run.state, history=history)
            if run.state == "finished":
                (CACHE / f"{name}.json").write_text(json.dumps(out[name]))
    return out


def curve(data, config, key):
    entry = data.get(training_run_name(config))
    if entry is None:
        return None, None, None
    steps, values = entry["history"][key]
    return np.array(steps, dtype=float) * TOKENS_PER_STEP / 1e6, np.array(values), entry["state"]


def smooth(values, window=100):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def by_lr():
    out = {}
    for c in RUNS:
        out.setdefault(c.learning_rate, []).append(c)
    return dict(sorted(out.items()))


def main():
    data = fetch()
    groups = by_lr()
    fig, axes = plt.subplots(3, len(groups), figsize=(5.2 * len(groups), 11), sharex=True, squeeze=False)
    for j, (lr, configs) in enumerate(groups.items()):
        for c in configs:
            color, label = COLORS[c.lr_schedule], SCHEDULES[c.lr_schedule]
            tok, lrs, state = curve(data, c, "learning_rate")
            if tok is None:
                continue
            suffix = "" if state == "finished" else f" ({state})"
            axes[0, j].plot(tok, lrs, color=color, label=label + suffix)
            tok, loss, _ = curve(data, c, "train_loss")
            axes[1, j].plot(tok[99:] if len(tok) >= 100 else tok, smooth(loss), color=color, lw=1)
            tok, val, _ = curve(data, c, "val_loss")
            axes[2, j].plot(tok, val, color=color, marker="." if state != "finished" else None)
        axes[0, j].set(title=f"peak LR {lr:g}", ylabel="learning rate")
        axes[0, j].legend(fontsize=8)
        axes[1, j].set(ylabel="training loss (100-step mean)", ylim=(2.85, 3.6))
        axes[2, j].set(ylabel="validation loss", ylim=(2.88, 3.5), xlabel="training tokens (M)")
        for i in range(3):
            axes[i, j].grid(alpha=0.3)
    fig.suptitle("P2(a): learning-rate schedules, d8, 614M tokens (batch 64); one column per peak LR")
    fig.tight_layout()
    savefig(fig, "p2a_schedules.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    best = None
    print(f"{'schedule':>16} " + " ".join(f"{lr:>9g}" for lr in groups))
    for schedule, label in SCHEDULES.items():
        lrs, finals, row = [], [], []
        for lr, configs in groups.items():
            c = next((c for c in configs if c.lr_schedule == schedule), None)
            if c is None:
                row.append(f"{'-':>9}")
                continue
            entry = data.get(training_run_name(c))
            if entry is None or entry["state"] != "finished":
                row.append(f"{'running':>9}")
                continue
            final = entry["history"]["val_loss"][1][-1]
            lrs.append(lr); finals.append(final); row.append(f"{final:9.4f}")
            if best is None or final < best[0]:
                best = (final, c)
        ax.plot(lrs, finals, "o-", color=COLORS[schedule], label=label)
        print(f"{label:>16} " + " ".join(row))
    if best:
        ax.plot(best[1].learning_rate, best[0], "k*", ms=14, zorder=5,
                label=f"reference: {SCHEDULES[best[1].lr_schedule]}, LR {best[1].learning_rate:g}")
        print(f"reference run: {training_run_name(best[1])}  val {best[0]:.4f}")
    ax.set(xscale="log", xlabel="peak learning rate", ylabel="final validation loss",
           title="P2(a): final validation loss vs. peak LR")
    ax.set_xticks(list(groups), [f"{lr:g}" for lr in groups])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "p2a_final_vs_lr.png")


if __name__ == "__main__":
    main()
