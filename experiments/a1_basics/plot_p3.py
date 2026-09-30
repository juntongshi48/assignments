"""Problem 3 analysis: run-to-run variation by source, and how hyperparameters change it.
Writes plots/p3_*.png and prints SD tables."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p3_measuring_variation import P3A_KEY, P3B_KEY, P3C_KEY
from experiments.a1_basics.results import history, load_runs, savefig

BASELINE_42 = "model-d8-lr0.003-tok614M-p1a-v1"  # seeds 42/42, nondeterministic, H100
SMOKE_42 = "model-d8-lr0.003-tok614M-modal"  # identical config, earlier H100 run
DET_REFS = (
    "model-d8-lr0.003-tok614M-deterministic-deterministic-reference-1",
    "model-d8-lr0.003-tok614M-deterministic-deterministic-reference-2",
)
SEED_42_BY_GROUP = {
    "lr0.01": "model-d8-lr0.01-tok614M-p1a-v1",
    "bs16": "model-d8-lr0.003-bs16-tok614M-p1a-v1",
    "d9": "model-d9-lr0.003-tok614M-p2a-v1",
}
EXTRA_NAMES = [BASELINE_42, SMOKE_42, *DET_REFS, *SEED_42_BY_GROUP.values()]


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def groups_a_b(by_name, runs):
    tagged = lambda tag: [run for run in runs if run.has(tag)]
    ref = by_name[DET_REFS[0]]
    return {
        "all sources\n(seeds 1,2,3,42)": [*tagged("all-sources"), by_name[BASELINE_42]],
        "init only\n(det., model seed)": [*tagged("init-only"), ref],
        "data order only\n(det., data seed)": [*tagged("data-order-only"), ref],
        "GPU nondeterminism\n(H100, seed 42 x3)": [by_name[SMOKE_42], by_name[BASELINE_42],
                                                    *[run for run in tagged("H100") if run.has(P3B_KEY)]],
        "GPU nondeterminism\n(A100, seed 42 x2)": [run for run in tagged("A100") if run.has(P3B_KEY)],
        "deterministic refs\n(H100, identical)": [by_name[name] for name in DET_REFS],
    }


def groups_c(by_name, runs, standard):
    groups = {"standard (d8)": standard}
    for group, name in SEED_42_BY_GROUP.items():
        groups[group] = [*[run for run in runs if run.has(P3C_KEY) and run.has(f"group-{group}")], by_name[name]]
    return groups


def strip(ax, groups, title, center=False):
    for index, (label, members) in enumerate(groups.items()):
        losses = np.array([run.val_loss for run in members])
        shown = losses - losses.mean() if center else losses
        jitter = np.linspace(-0.12, 0.12, len(shown))
        ax.scatter(index + jitter, shown, s=45, zorder=4, edgecolor="black", lw=0.5)
        ax.annotate(f"SD {sd(losses) * 1e3:.2f}e-3\nn={len(losses)}", (index, shown.max()),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(groups)), list(groups), fontsize=8.5)
    ax.set_ylabel("final val loss - group mean" if center else "final validation loss")
    ax.set_title(title)
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    ax.margins(y=0.25)


def spread_over_training(ax, groups):
    """Across-run SD of validation loss at each eval step (log scale)."""
    for label, members in groups.items():
        curves = [history(run, ["val_loss"]) for run in members]
        n = min(len(curve["val_loss"]) for curve in curves)
        progress = (curves[0]["optimizer_step"][:n] + 1) / float(members[0].config["total_steps"])
        stack = np.stack([curve["val_loss"][:n] for curve in curves])
        spread = stack.std(axis=0, ddof=1)
        ax.plot(progress, np.maximum(spread, 1e-7), lw=1.6, label=label.replace("\n", " "))
    ax.set_yscale("log")
    ax.set_xlabel("fraction of training (identical runs plotted at 1e-7)")
    ax.set_ylabel("SD of val loss across runs")
    ax.set_title("How run-to-run spread evolves during training")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5)


def main():
    runs = [run for run in load_runs([P3A_KEY, P3B_KEY, P3C_KEY], names=EXTRA_NAMES) if run.state == "finished"]
    by_name = {run.name: run for run in runs}
    missing = [name for name in EXTRA_NAMES if name not in by_name]
    if missing:
        raise SystemExit(f"missing reference runs: {missing}")

    ab = groups_a_b(by_name, runs)
    print("Determinism check (reference-1 vs reference-2):")
    for key in ("val_loss", "train_loss"):
        # Train and eval metrics are logged in separate wandb.log calls, so fetch separately.
        refs = [history(by_name[name], [key]) for name in DET_REFS]
        diff = np.abs(refs[0][key] - refs[1][key])
        print(f"  max |diff| {key}: {diff.max():.3e} over {len(diff)} logged steps")

    print("\nP3(a)/(b): final val loss by source of variation")
    total = sd([run.val_loss for run in next(iter(ab.values()))])
    for label, members in ab.items():
        losses = [run.val_loss for run in members]
        print(f"  {label.replace(chr(10), ' '):<44} n={len(losses)} mean={np.mean(losses):.4f} "
              f"SD={sd(losses):.2e} var fraction of all-sources={sd(losses) ** 2 / total ** 2:6.2f}  "
              + " ".join(f"{loss:.4f}" for loss in losses))
    h100 = [run.val_loss for run in ab["GPU nondeterminism\n(H100, seed 42 x3)"]]
    a100 = [run.val_loss for run in ab["GPU nondeterminism\n(A100, seed 42 x2)"]]
    print(f"  A100 mean - H100 mean = {np.mean(a100) - np.mean(h100):+.4f}")

    c = groups_c(by_name, runs, ab["all sources\n(seeds 1,2,3,42)"])
    print("\nP3(c): does variability change? (paired seeds 1,2,3,42)")
    base_sd = sd([run.val_loss for run in c["standard (d8)"]])
    for label, members in c.items():
        losses = [run.val_loss for run in members]
        print(f"  {label:<14} mean={np.mean(losses):.4f} SD={sd(losses):.2e} SD ratio vs standard={sd(losses) / base_sd:5.2f}  "
              + " ".join(f"{loss:.4f}" for loss in losses))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    strip(axes[0], {k: v for k, v in ab.items() if "identical" not in k}, "P3(a)/(b): final loss by source of variation")
    spread_over_training(axes[1], {k: v for k, v in ab.items() if "A100" not in k})
    fig.tight_layout()
    savefig(fig, "p3ab_sources.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strip(axes[0], c, "P3(c): spread around each group's mean", center=True)
    spread_over_training(axes[1], c)
    fig.tight_layout()
    savefig(fig, "p3c_hyperparameters.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
