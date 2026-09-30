from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a1_basics.p1_hyperparameters import PART_A_KEY, PART_A_SWEEPS
from experiments.lr_tuning.plot_lr_tuning import (
    created_at_sort_key,
    finite_float,
    metric_trajectory,
    run_created_at,
    set_style,
)
from train import TrainConfig
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
CONTEXT_LENGTH = 1024
CMAP = mpl.colormaps["viridis"]
LABELS = {
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "warmup_percent": "Warmup fraction",
    "weight_decay": "Weight decay",
}


def latest_runs_by_value(experiment_key, sweeps):
    """Return {hparam: {value: run}}, keeping the newest run per value.

    The baseline run is shared by every sweep at its default value.
    """
    defaults = TrainConfig()
    api = wandb.Api()
    latest = {name: {} for name in sweeps}
    for run in api.runs(PROJECT_PATH, filters={"tags": {"$in": [experiment_key]}}):
        last_step = finite_float(run.summary.get("optimizer_step"))
        total_steps = finite_float(run.config.get("total_steps"))
        if last_step is not None and total_steps is not None and last_step >= total_steps:
            # A crashed run left open in a reused container can absorb the next job's logs.
            print(f"  skipping {run.name} ({run.id}): logged step {last_step:g} >= total_steps {total_steps:g}")
            continue
        tags = set(run.tags or [])
        if "baseline" in tags:
            names = list(sweeps)
        else:
            names = [name for name in sweeps if f"sweep-{name}" in tags]
        for name in names:
            value = finite_float(run.config.get(name))
            if value is None:
                continue
            expected = [v for v in sweeps[name] if np.isclose(v, value)]
            if not expected or ("baseline" not in tags and expected[0] == getattr(defaults, name)):
                continue
            key = created_at_sort_key(run_created_at(run))
            current = latest[name].get(expected[0])
            if current is None or key > current[0]:
                latest[name][expected[0]] = (key, run)
    return {name: {v: run for v, (_, run) in runs.items()} for name, runs in latest.items()}


def plot_sweeps(experiment_key, sweeps, out_path):
    runs = latest_runs_by_value(experiment_key, sweeps)
    defaults = TrainConfig()
    set_style()
    fig, axes = plt.subplots(2, len(sweeps), figsize=(4.2 * len(sweeps), 7.5))

    for col, (name, values) in enumerate(sweeps.items()):
        ax_curve, ax_final = axes[0, col], axes[1, col]
        norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))
        finals = []
        for value in values:
            run = runs[name].get(value)
            if run is None:
                print(f"  missing: {name}={value}")
                continue
            steps, losses = metric_trajectory(run, "val_loss")
            if not losses.size:
                print(f"  no val_loss yet: {name}={value} ({run.state})")
                continue
            batch_size = int(run.config.get("batch_size", defaults.batch_size))
            tokens = (steps + 1) * batch_size * CONTEXT_LENGTH
            ax_curve.plot(tokens, losses, color=CMAP(norm(value)), lw=1.6, label=f"{value:g}")
            finals.append((value, float(losses[-1]), run.state))
            print(f"{name}={value:g} val_loss={losses[-1]:.4f} state={run.state} name={run.name}")

        ax_curve.set_xscale("log")
        ax_curve.set_ylim(2.85, 4.0)
        ax_curve.set_xlabel("Training tokens")
        ax_curve.set_ylabel("Validation loss")
        ax_curve.set_title(LABELS.get(name, name))
        ax_curve.grid(True, linestyle=":", alpha=0.3)
        ax_curve.legend(frameon=False, fontsize=8.5)

        if finals:
            xs, ys, states = zip(*finals)
            ax_final.plot(xs, ys, color="#777777", lw=1.0, alpha=0.5)
            ax_final.scatter(xs, ys, c=[CMAP(norm(x)) for x in xs], s=70, edgecolor="#111111", zorder=5)
            finished = [(x, y) for x, y, state in finals if state == "finished"]
            if finished:
                best_x, best_y = min(finished, key=lambda item: item[1])
                ax_final.scatter([best_x], [best_y], s=150, facecolors="none", edgecolor="#e74c3c", lw=2, zorder=6)
            for x, y, state in finals:
                label = f"{y:.3f}" + ("" if state == "finished" else f"\n({state})")
                ax_final.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color="#555555")
        ax_final.axvline(getattr(defaults, name), color="#999999", ls="--", lw=0.8)
        ax_final.set_xscale("log")
        ax_final.set_xticks(list(values))
        ax_final.set_xticklabels([f"{v:g}" for v in values], rotation=20)
        ax_final.minorticks_off()
        ax_final.set_xlabel(LABELS.get(name, name))
        ax_final.set_ylabel("Final validation loss")
        ax_final.margins(y=0.15)
        ax_final.grid(True, linestyle=":", alpha=0.3)

    fig.suptitle(f"{experiment_key}: one-at-a-time sweeps around d8 default (compute-matched)")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    print(plot_sweeps(PART_A_KEY, PART_A_SWEEPS, PLOT_DIR / f"{PART_A_KEY}_sweeps.png").resolve())


if __name__ == "__main__":
    main()
