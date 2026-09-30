from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from experiments.a1_basics.p1_hyperparameters import (
    PART_A_KEY,
    PART_B_KEY,
    PART_C2_WARMUP,
    PART_C_KEY,
    PART_C_SCHEDULES,
    PART_C_WARMUPS,
)
from experiments.lr_tuning.plot_lr_tuning import (
    created_at_sort_key,
    finite_float,
    metric_trajectory,
    run_created_at,
    set_style,
)
from lr_schedules import build_scheduler
from train import TrainConfig
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
KEYS = ("learning_rate", "batch_size", "warmup_percent", "weight_decay", "lr_schedule", "grad_norm")
SCHEDULE_LABELS = {
    "linear": "linear",
    "cos": "cosine",
    "wsd0.0": "constant (wsd0.0)",
    "wsd0.1": "wsd0.1",
    "wsd0.2": "wsd0.2",
    "wsd0.5": "wsd0.5",
}
SCHEDULE_COLORS = {
    "linear": "#1f77b4",
    "cos": "#ff7f0e",
    "wsd0.0": "#2ca02c",
    "wsd0.1": "#9467bd",
    "wsd0.2": "#d62728",
    "wsd0.5": "#8c564b",
}
LOSS_CMAP = mpl.colormaps["viridis_r"]
HEATMAP_RANGE = 0.08
TOTAL_STEPS = 9375


def normalize(name, value):
    if name == "lr_schedule":
        return "linear" if value == "lin" else value
    if name == "grad_norm" and value is None:
        return None
    return finite_float(value)


def collect_runs():
    """Return {config tuple over KEYS: run} for every finished p1 run (newest wins)."""
    defaults = TrainConfig()
    newest = {}
    runs = wandb.Api().runs(PROJECT_PATH, filters={"tags": {"$in": [PART_A_KEY, PART_B_KEY, PART_C_KEY]}})
    for run in runs:
        last_step = finite_float(run.summary.get("optimizer_step"))
        total_steps = finite_float(run.config.get("total_steps"))
        if last_step is not None and total_steps is not None and last_step >= total_steps:
            print(f"  skipping {run.name} ({run.id}): logged step {last_step:g} >= total_steps {total_steps:g}")
            continue
        if run.state != "finished":
            print(f"  not finished: {run.name} ({run.state})")
            continue
        values = tuple(normalize(k, run.config.get(k, getattr(defaults, k))) for k in KEYS)
        key = created_at_sort_key(run_created_at(run))
        if values not in newest or key > newest[values][0]:
            newest[values] = (key, run)
    return {values: run for values, (_, run) in newest.items()}


def find(runs, **overrides):
    defaults = TrainConfig()
    target = [normalize(k, overrides.get(k, getattr(defaults, k))) for k in KEYS]
    for values, run in runs.items():
        if all(
            v == t if isinstance(t, str) or t is None or v is None else np.isclose(v, t)
            for v, t in zip(values, target)
        ):
            return run
    return None


def final_loss(run):
    return None if run is None else finite_float(run.summary.get("val_loss"))


def heatmap(ax, runs, row_name, row_values, col_name, col_values, fixed, vmin, title):
    losses = np.full((len(row_values), len(col_values)), np.nan)
    for i, r in enumerate(row_values):
        for j, c in enumerate(col_values):
            loss = final_loss(find(runs, **{row_name: r, col_name: c}, **fixed))
            if loss is not None:
                losses[i, j] = loss
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmin + HEATMAP_RANGE)
    cmap = LOSS_CMAP.copy()
    cmap.set_bad("#e6e6e6")
    ax.imshow(np.ma.masked_invalid(losses), cmap=cmap, norm=norm, aspect="auto")
    best = np.unravel_index(np.nanargmin(losses), losses.shape) if np.isfinite(losses).any() else None
    for i in range(len(row_values)):
        for j in range(len(col_values)):
            value = losses[i, j]
            text = f"{value:.3f}" if np.isfinite(value) else "missing"
            dark = np.isfinite(value) and norm(value) > 0.55
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5,
                    color="white" if dark else "#111111",
                    fontweight="bold" if best == (i, j) else "normal")
    if best is not None:
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor="#e74c3c", lw=2))

    def fmt(name, v):
        return SCHEDULE_LABELS.get(v, v) if name == "lr_schedule" else ("off" if v is None else f"{v:g}")

    ax.set_xticks(range(len(col_values)), [fmt(col_name, v) for v in col_values])
    ax.set_yticks(range(len(row_values)), [fmt(row_name, v) for v in row_values])
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title)
    return losses


def lr_curve(lr_schedule, warmup_percent, peak=1.0):
    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=peak)
    scheduler = build_scheduler(optimizer, lr_schedule, int(TOTAL_STEPS * warmup_percent), TOTAL_STEPS)
    lrs = []
    for _ in range(TOTAL_STEPS):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.array(lrs)


def plot_curves(ax, runs, schedules, warmup_percent, title):
    """Validation loss per schedule (left axis) with its LR multiplier (right axis, dashed)."""
    ax_lr = ax.twinx()
    for lr_schedule in schedules:
        run = find(runs, lr_schedule=lr_schedule, warmup_percent=warmup_percent)
        color = SCHEDULE_COLORS[lr_schedule]
        ax_lr.plot(np.arange(TOTAL_STEPS), lr_curve(lr_schedule, warmup_percent), color=color, ls="--", lw=1, alpha=0.5)
        if run is None:
            continue
        steps, losses = metric_trajectory(run, "val_loss")
        ax.plot(steps + 1, losses, color=color, lw=1.7,
                label=f"{SCHEDULE_LABELS[lr_schedule]} ({losses[-1]:.3f})")
    ax.set_ylim(2.88, 3.4)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation loss")
    ax_lr.set_ylabel("LR / peak LR (dashed)")
    ax_lr.set_ylim(0, 1.6)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def main():
    runs = collect_runs()
    set_style()
    vmin = min(loss for loss in (final_loss(r) for r in runs.values()) if loss is not None)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (1) schedule x warmup at LR 3e-3.
    table = heatmap(axes[0, 0], runs, "lr_schedule", PART_C_SCHEDULES, "warmup_percent", PART_C_WARMUPS,
                    {}, vmin, "Schedule x warmup (LR 3e-3)")
    print("\nschedule x warmup", PART_C_WARMUPS)
    for s, row in zip(PART_C_SCHEDULES, table):
        print(f"  {s:<7}", "  ".join(f"{v:.4f}" for v in row))

    # (2) schedule x peak LR at warmup 0.1.
    lrs = (1e-3, 3e-3, 1e-2)
    schedules = ("linear", "cos", "wsd0.2")
    table = heatmap(axes[0, 1], runs, "lr_schedule", schedules, "learning_rate", lrs,
                    {"warmup_percent": PART_C2_WARMUP}, vmin, f"Schedule x peak LR (warmup {PART_C2_WARMUP:g})")
    print("\nschedule x LR", lrs)
    for s, row in zip(schedules, table):
        print(f"  {s:<7}", "  ".join(f"{v:.4f}" for v in row))

    # (3) gradient clipping x LR at default warmup.
    table = heatmap(axes[0, 2], runs, "grad_norm", (1.0, None), "learning_rate", (3e-3, 1e-2, 3e-2),
                    {}, vmin, "Gradient clipping x LR (linear, warmup 0.01)")
    print("\nclip x LR (3e-3, 1e-2, 3e-2)")
    for g, row in zip(("clip1.0", "noclip"), table):
        print(f"  {g:<7}", "  ".join(f"{v:.4f}" if np.isfinite(v) else "   -   " for v in row))

    # (4) WSD decay fraction at warmup 0.1, with linear/cosine as reference lines.
    ax = axes[1, 0]
    fractions, losses = [], []
    for fraction in (0.0, 0.1, 0.2, 0.5):
        loss = final_loss(find(runs, lr_schedule=f"wsd{fraction}", warmup_percent=PART_C2_WARMUP))
        if loss is not None:
            fractions.append(fraction)
            losses.append(loss)
    ax.plot(fractions, losses, "o-", color=SCHEDULE_COLORS["wsd0.2"], lw=1.8, label="wsd (stable, then linear decay)")
    for x, y in zip(fractions, losses):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
    for reference in ("linear", "cos"):
        loss = final_loss(find(runs, lr_schedule=reference, warmup_percent=PART_C2_WARMUP))
        if loss is not None:
            ax.axhline(loss, color=SCHEDULE_COLORS[reference], ls=":", lw=1.4, label=f"{SCHEDULE_LABELS[reference]} ({loss:.3f})")
    ax.set_xlabel("Decay fraction (share of training spent decaying)")
    ax.set_ylabel("Final validation loss")
    ax.set_title(f"WSD decay length (LR 3e-3, warmup {PART_C2_WARMUP:g})")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    print("\nwsd decay fraction", dict(zip(fractions, [round(l, 4) for l in losses])))

    # (5) and (6) loss curves with LR schedule overlaid.
    plot_curves(axes[1, 1], runs, PART_C_SCHEDULES, PART_C2_WARMUP, f"Loss curves by schedule (warmup {PART_C2_WARMUP:g})")
    plot_curves(axes[1, 2], runs, PART_C_SCHEDULES, 0.0, "Loss curves by schedule (no warmup)")

    fig.suptitle(f"{PART_C_KEY}: LR schedules and their interactions (d8, compute-matched)")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"{PART_C_KEY}_schedules.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
