from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a1_basics.p1_hyperparameters import (
    PART_A_KEY,
    PART_A_SWEEPS,
    PART_B_GRIDS,
    PART_B_KEY,
)
from experiments.lr_tuning.plot_lr_tuning import (
    created_at_sort_key,
    finite_float,
    run_created_at,
    set_style,
)
from train import TrainConfig
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
HPARAMS = tuple(PART_A_SWEEPS)
LABELS = {
    "learning_rate": "Learning rate",
    "batch_size": "Batch size",
    "warmup_percent": "Warmup fraction",
    "weight_decay": "Weight decay",
}
LOSS_CMAP = mpl.colormaps["viridis_r"]
LINE_CMAP = mpl.colormaps["plasma"]
# Color scale spans best loss .. best + this; worse cells saturate.
HEATMAP_RANGE = 0.08


def collect_runs():
    """Return {hparam values tuple: (state, final val loss)} over parts (a) and (b).

    Keeps the newest run per configuration and skips runs whose logged steps exceed
    their configured total (a crashed run that absorbed another job's logs).
    """
    defaults = TrainConfig()
    api = wandb.Api()
    newest = {}
    for run in api.runs(PROJECT_PATH, filters={"tags": {"$in": [PART_A_KEY, PART_B_KEY]}}):
        last_step = finite_float(run.summary.get("optimizer_step"))
        total_steps = finite_float(run.config.get("total_steps"))
        if last_step is not None and total_steps is not None and last_step >= total_steps:
            print(f"  skipping {run.name} ({run.id}): logged step {last_step:g} >= total_steps {total_steps:g}")
            continue
        values = tuple(
            finite_float(run.config.get(name, getattr(defaults, name))) for name in HPARAMS
        )
        key = created_at_sort_key(run_created_at(run))
        if values not in newest or key > newest[values][0]:
            newest[values] = (key, run.state, finite_float(run.summary.get("val_loss")))
    return {values: (state, loss) for values, (_, state, loss) in newest.items()}


def lookup(runs, **overrides):
    defaults = TrainConfig()
    target = [float(overrides.get(name, getattr(defaults, name))) for name in HPARAMS]
    for values, result in runs.items():
        if all(v is not None and np.isclose(v, t) for v, t in zip(values, target)):
            return result
    return None


def grid_losses(runs, x_name, x_values, y_name, y_values):
    """Return (losses[y, x] with NaN for missing/unfinished, states[y][x])."""
    losses = np.full((len(y_values), len(x_values)), np.nan)
    states = [["missing"] * len(x_values) for _ in y_values]
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            result = lookup(runs, **{x_name: x, y_name: y})
            if result is None:
                continue
            state, loss = result
            states[i][j] = state
            if state == "finished" and loss is not None:
                losses[i, j] = loss
    return losses, states


def plot_heatmap(ax, losses, states, x_name, x_values, y_name, y_values, vmin):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmin + HEATMAP_RANGE)
    cmap = LOSS_CMAP.copy()
    cmap.set_bad("#e6e6e6")  # missing / unfinished cells
    ax.imshow(np.ma.masked_invalid(losses), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(y_values)):
        row = losses[i]
        best_j = int(np.nanargmin(row)) if np.isfinite(row).sum() >= 2 else None
        for j in range(len(x_values)):
            if np.isfinite(losses[i, j]):
                text = f"{losses[i, j]:.3f}"
                dark = norm(losses[i, j]) > 0.55
            else:
                text, dark = states[i][j], False
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5,
                    color="white" if dark else "#111111",
                    fontweight="bold" if j == best_j else "normal")
        if best_j is not None:
            ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#e74c3c", lw=2))
    ax.set_xticks(range(len(x_values)), [f"{v:g}" for v in x_values])
    ax.set_yticks(range(len(y_values)), [f"{v:g}" for v in y_values])
    ax.set_xlabel(LABELS[x_name])
    ax.set_ylabel(LABELS[y_name])
    ax.set_title(f"{LABELS[x_name]} x {LABELS[y_name]}\n(red box: best LR per row)")


def plot_lines(ax, losses, x_name, x_values, y_name, y_values):
    norm = mpl.colors.LogNorm(vmin=min(y_values), vmax=max(y_values))
    for i, y in enumerate(y_values):
        mask = np.isfinite(losses[i])
        if not mask.any():
            continue
        xs = np.array(x_values)[mask]
        color = LINE_CMAP(0.85 * norm(y))
        ax.plot(xs, losses[i][mask], "o-", color=color, lw=1.6, label=f"{y:g}")
        best = int(np.argmin(losses[i][mask]))
        ax.scatter([xs[best]], [losses[i][mask][best]], s=140, facecolors="none", edgecolor=color, lw=2)
    ax.set_xscale("log")
    ax.set_xticks(list(x_values), [f"{v:g}" for v in x_values])
    ax.minorticks_off()
    ax.set_xlabel(LABELS[x_name])
    ax.set_ylabel("Final validation loss")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(title=LABELS[y_name], frameon=False, fontsize=8.5, title_fontsize=9)


def plot_lr_wd_product(runs, out_path):
    """LR x WD grid against lr*wd: collapse onto one curve means only the product matters."""
    x_values, y_values = PART_B_GRIDS[("learning_rate", "weight_decay")]
    losses, _ = grid_losses(runs, "learning_rate", x_values, "weight_decay", y_values)
    norm = mpl.colors.LogNorm(vmin=min(x_values), vmax=max(x_values))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for j, lr in enumerate(x_values):
        mask = np.isfinite(losses[:, j])
        products = lr * np.array(y_values)[mask]
        ax.plot(products, losses[:, j][mask], "o-", color=LINE_CMAP(0.85 * norm(lr)), lw=1.6, label=f"LR {lr:g}")
    ax.set_xscale("log")
    ax.set_xlabel("lr x weight decay (per-step decay)")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Does only lr x wd matter?")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    runs = collect_runs()
    set_style()
    grids = list(PART_B_GRIDS.items())
    fig, axes = plt.subplots(2, len(grids), figsize=(5.2 * len(grids), 8.5))
    finished = [loss for state, loss in runs.values() if state == "finished" and loss is not None]
    vmin = min(finished)
    for col, ((x_name, y_name), (x_values, y_values)) in enumerate(grids):
        losses, states = grid_losses(runs, x_name, x_values, y_name, y_values)
        print(f"\n{LABELS[x_name]} x {LABELS[y_name]} (rows: {y_name}, cols: {x_name})")
        for y, row, row_states in zip(y_values, losses, states):
            cells = [f"{loss:.4f}" if np.isfinite(loss) else f"{state:>6}" for loss, state in zip(row, row_states)]
            print(f"  {y_name}={y:<6g} " + "  ".join(cells))
        plot_heatmap(axes[0, col], losses, states, x_name, x_values, y_name, y_values, vmin)
        plot_lines(axes[1, col], losses, x_name, x_values, y_name, y_values)
    fig.suptitle(f"{PART_B_KEY}: LR co-variation grids (d8, compute-matched; part (a) runs fill default row/column)")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = PLOT_DIR / f"{PART_B_KEY}_grids.png"
    fig.savefig(grid_path)
    plt.close(fig)
    product_path = plot_lr_wd_product(runs, PLOT_DIR / f"{PART_B_KEY}_lr_wd_product.png")
    print(grid_path.resolve())
    print(product_path.resolve())


if __name__ == "__main__":
    main()
