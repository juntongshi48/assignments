"""Problem 2 analysis: scaling-law fits, pre-registered d8/d9 predictions, slope changes,
and screens for power-law breakage. Writes plots/p2{a,b,c}_*.png and prints tables."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p2_scaling_law_reliability import (
    P2A_KEY,
    P2A_VARIANTS,
    P2B_KEY,
    P2C_INTERVENTIONS,
    P2C_KEY,
)
from experiments.a1_basics.results import body_compute, fit_power_law, history, load_runs, savefig

TOKENS = 600_000 * 1024
D8_COMPUTE = body_compute(8, TOKENS)
DEPTHS = np.arange(4, 10)
PRE_REGISTER_DEPTHS = (4, 5, 6, 7)
FAR_DEPTH = 20
# d8 points reused from Problem 1 (see p2_scaling_law_reliability.P2A_REUSED_D8).
REUSED_D8 = {
    "baseline": "model-d8-lr0.003-tok614M-p1a-v1",
    "lr0.03": "model-d8-lr0.03-tok614M-p1a-v1",
    "constant": "model-d8-lr0.003-tok614M-warmup0.0-wsd0.0-p1c-v1",
}
COLORS = {
    "baseline": "#1f77b4",
    "constant": "#2ca02c",
    "dropout0.2": "#9467bd",
    "lr0.03": "#d62728",
    "lr-width-scaled": "#ff7f0e",
    "repeat10x": "#8c564b",
    "no-qk-norm": "#e377c2",
    "no-warmup": "#7f7f7f",
    "lr0.03-no-clip": "#bcbd22",
}


def rel_compute(depth, tokens=TOKENS):
    return body_compute(depth, tokens) / D8_COMPUTE


def tag_value(run, prefix):
    return next((tag[len(prefix):] for tag in run.tags if tag.startswith(prefix)), None)


def finished(runs):
    return [run for run in runs if run.state == "finished" and run.val_loss is not None]


def part_a(runs_by_name):
    """variant -> {depth: loss}."""
    curves = {variant: {} for variant in P2A_VARIANTS}
    for run in runs_by_name.values():
        variant = tag_value(run, "variant-")
        if run.has(P2A_KEY) and variant:
            curves[variant][run.depth] = run.val_loss
    for variant, name in REUSED_D8.items():
        if name in runs_by_name:
            curves[variant][8] = runs_by_name[name].val_loss
    return curves


def plot_part_a(curves):
    fig, (ax, ax_res) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    far_x = rel_compute(FAR_DEPTH)
    grid = np.geomspace(rel_compute(4), far_x, 200)
    print("\nP2(a): fit L = a*C^-alpha + eps on d4-d7 (pre-registration) and on d4-d9 (final)")
    print(f"{'variant':<11} {'alpha':>6} {'eps':>6} | {'pred d8':>8} {'act d8':>7} {'pred d9':>8} {'act d9':>7} | {'d20 (d4-7)':>10} {'d20 (d4-9)':>10}")
    residuals = {}
    for variant, points in curves.items():
        depths = sorted(points)
        pre = [d for d in depths if d in PRE_REGISTER_DEPTHS]
        if len(pre) < 3:
            continue
        color = COLORS[variant]
        fit_pre = fit_power_law([rel_compute(d) for d in pre], [points[d] for d in pre])
        fit_all = fit_power_law([rel_compute(d) for d in depths], [points[d] for d in depths])
        ax.plot(grid, fit_pre["predict"](grid), color=color, ls="--", lw=1.2, alpha=0.8)
        ax.scatter([rel_compute(d) for d in pre], [points[d] for d in pre], color=color, s=40, zorder=4,
                   label=f"{variant} (alpha={fit_pre['alpha']:.2f})")
        held = [d for d in depths if d not in PRE_REGISTER_DEPTHS]
        ax.scatter([rel_compute(d) for d in held], [points[d] for d in held], facecolors="white",
                   edgecolors=color, s=55, lw=1.8, zorder=5)
        residuals[variant] = {d: points[d] - float(fit_pre["predict"](rel_compute(d))) for d in held}
        print(f"{'':<11} points: " + "  ".join(f"d{d}={points[d]:.4f}" for d in depths))
        cells = []
        for d in (8, 9):
            pred = float(fit_pre["predict"](rel_compute(d)))
            cells += [f"{pred:8.4f}", f"{points[d]:7.4f}" if d in points else f"{'-':>7}"]
        print(f"{variant:<11} {fit_pre['alpha']:6.3f} {fit_pre['eps']:6.3f} | " + " ".join(cells)
              + f" | {float(fit_pre['predict'](far_x)):10.4f} {float(fit_all['predict'](far_x)):10.4f}")
    ax.axvline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Body compute relative to d8 (6ND, fixed 614M tokens)")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Scaling d4-d9; dashed = fit on d4-d7 only, hollow = held out (d8, d9)")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    secondary = ax.secondary_xaxis("top", functions=(lambda c: 8 * np.cbrt(c), lambda d: (d / 8) ** 3))
    secondary.set_xticks([4, 6, 8, 10, 14, 20])
    secondary.set_xlabel("depth")

    for index, (variant, by_depth) in enumerate(residuals.items()):
        for d, residual in by_depth.items():
            ax_res.bar(index + (0.2 if d == 9 else -0.2), residual, width=0.38, color=COLORS[variant],
                       alpha=1.0 if d == 9 else 0.5, edgecolor="black", lw=0.5)
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_xticks(range(len(residuals)), list(residuals), rotation=15)
    ax_res.set_ylabel("actual - predicted (nats)")
    ax_res.set_title("Pre-registration error (light: d8, dark: d9)")
    ax_res.grid(True, axis="y", ls=":", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "p2a_scaling.png")
    plt.close(fig)


def plot_part_b(runs_by_name, curves):
    fig, (ax_model, ax_data) = plt.subplots(1, 2, figsize=(13, 4.8))
    baseline = curves["baseline"]
    width_scaled = {8: baseline.get(8)}
    data = {64: {600_000: baseline.get(6)}, 128: {}}
    for run in runs_by_name.values():
        if not run.has(P2B_KEY):
            continue
        if run.has("lr-width-scaled"):
            width_scaled[run.depth] = run.val_loss
        for batch_size in (64, 128):
            if run.has(f"data-scaling-bs{batch_size}"):
                data[batch_size][int(run.config["num_train_sequences"])] = run.val_loss

    print("\nP2(b) model axis: fixed LR 0.003 vs LR = 0.024/depth (shared eps from baseline fit)")
    base_depths = sorted(baseline)
    base_fit = fit_power_law([rel_compute(d) for d in base_depths], [baseline[d] for d in base_depths])
    grid = np.geomspace(rel_compute(4), rel_compute(FAR_DEPTH), 200)
    for label, points in (("baseline", baseline), ("lr-width-scaled", width_scaled)):
        depths = sorted(d for d in points if points[d] is not None)
        xs = [rel_compute(d) for d in depths]
        fit = fit_power_law(xs, [points[d] for d in depths], floors=[base_fit["eps"]])
        ax_model.scatter(xs, [points[d] for d in depths], color=COLORS[label], s=40, zorder=4,
                         label=f"{label} (alpha={fit['alpha']:.3f})")
        ax_model.plot(grid, fit["predict"](grid), color=COLORS[label], ls="--", lw=1.1)
        print(f"  {label:<16} alpha={fit['alpha']:.3f} rmse={fit['rmse']:.4f}  "
              + "  ".join(f"d{d}={points[d]:.4f}" for d in depths)
              + f"  d20={float(fit['predict'](rel_compute(FAR_DEPTH))):.4f}")
    ax_model.set_xscale("log")
    ax_model.set_xlabel("Body compute relative to d8")
    ax_model.set_ylabel("Final validation loss")
    ax_model.set_title(f"Model-size axis (eps fixed at {base_fit['eps']:.3f})")
    ax_model.grid(True, ls=":", alpha=0.3)
    ax_model.legend(frameon=False, fontsize=9)

    print("\nP2(b) data axis at d6: bs64 vs bs128 (shared eps from bs64 fit)")
    bs64 = data[64]
    fit64 = fit_power_law([n * 1024 for n in sorted(bs64)], [bs64[n] for n in sorted(bs64)])
    token_grid = np.geomspace(1.2e8, 2e9, 200)
    for batch_size, color in ((64, "#1f77b4"), (128, "#d62728")):
        points = data[batch_size]
        budgets = sorted(n for n in points if points[n] is not None)
        tokens = [n * 1024 for n in budgets]
        fit = fit_power_law(tokens, [points[n] for n in budgets], floors=[fit64["eps"]])
        ax_data.scatter(tokens, [points[n] for n in budgets], color=color, s=40, zorder=4,
                        label=f"bs{batch_size} (beta={fit['alpha']:.3f})")
        ax_data.plot(token_grid, fit["predict"](token_grid), color=color, ls="--", lw=1.1)
        print(f"  bs{batch_size:<4} beta={fit['alpha']:.3f} rmse={fit['rmse']:.4f}  "
              + "  ".join(f"{n * 1024 / 1e6:.0f}M={points[n]:.4f}" for n in budgets))
    ax_data.set_xscale("log")
    ax_data.set_xlabel("Training tokens (d6)")
    ax_data.set_ylabel("Final validation loss")
    ax_data.set_title(f"Data axis at d6 (eps fixed at {fit64['eps']:.3f})")
    ax_data.grid(True, ls=":", alpha=0.3)
    ax_data.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, "p2b_slopes.png")
    plt.close(fig)


def plot_part_c(runs_by_name, curves):
    baseline = curves["baseline"]
    lr003 = curves["lr0.03"]
    screens = {name: {} for name in P2C_INTERVENTIONS}
    screen_runs = {name: {} for name in P2C_INTERVENTIONS}
    for run in runs_by_name.values():
        name = tag_value(run, "intervention-")
        if run.has(P2C_KEY) and name in screens:
            screens[name][run.depth] = run.val_loss
            screen_runs[name][run.depth] = run

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_scale, ax_gap, ax_repeat, ax_noclip = axes.flat
    depths = sorted(baseline)
    ax_scale.plot([rel_compute(d) for d in depths], [baseline[d] for d in depths], "o-", color=COLORS["baseline"], label="baseline")
    ax_scale.plot([rel_compute(d) for d in sorted(lr003)], [lr003[d] for d in sorted(lr003)], "o-", color=COLORS["lr0.03"], alpha=0.5, label="lr0.03 (part a)")
    def chord_offset(points):
        """Middle point's deviation from the chord through the endpoints (L vs log C).

        A power law is convex here, so compare against the baseline's own offset.
        """
        ds = sorted(points)
        if len(ds) != 3:
            return None
        logs = np.log([rel_compute(d) for d in ds])
        return points[ds[1]] - np.interp(logs[1], [logs[0], logs[2]], [points[ds[0]], points[ds[2]]])

    screen_depths = sorted({d for points in screens.values() for d in points})
    print("\nP2(c) screen: loss, gap to reference curve (baseline; lr0.03 for lr0.03-no-clip), and")
    print(f"  mid-point offset from the log-C chord (baseline itself: "
          f"{chord_offset({d: baseline[d] for d in screen_depths}):+.4f})")
    for name, points in screens.items():
        ds = sorted(points)
        reference = lr003 if name == "lr0.03-no-clip" else baseline
        ax_scale.plot([rel_compute(d) for d in ds], [points[d] for d in ds], "s--", color=COLORS[name], label=name)
        gaps = [points[d] - reference[d] for d in ds if d in reference]
        ax_gap.plot([d for d in ds if d in reference], gaps, "s-", color=COLORS[name], label=name)
        offset = chord_offset(points)
        cells = [f"d{d}={points[d]:.4f}" + (f" (gap {points[d] - reference[d]:+.4f})" if d in reference else " (gap -)")
                 for d in ds]
        print(f"  {name:<15} " + "  ".join(cells) + (f"  offset {offset:+.4f}" if offset is not None else ""))
    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("Body compute relative to d8")
    ax_scale.set_ylabel("Final validation loss")
    ax_scale.set_title("Screen: interventions at d4 / d6 / d9")
    ax_scale.grid(True, ls=":", alpha=0.3)
    ax_scale.legend(frameon=False, fontsize=8)
    ax_gap.axhline(0, color="black", lw=0.8)
    ax_gap.set_xlabel("depth")
    ax_gap.set_ylabel("loss - reference loss (nats)")
    ax_gap.set_title("Gap vs depth (parallel = constant offset; trend/bend = scale dependence)")
    ax_gap.grid(True, ls=":", alpha=0.3)
    ax_gap.legend(frameon=False, fontsize=8)

    for ax, name, title in ((ax_repeat, "repeat10x", "10x repeated data: val loss over training"),
                            (ax_noclip, "lr0.03-no-clip", "LR 0.03 without clipping: val loss over training")):
        norm = mpl.colors.Normalize(vmin=3, vmax=10)
        for d, run in sorted(screen_runs[name].items()):
            h = history(run, ["val_loss"])
            ax.plot(h["optimizer_step"], h["val_loss"], color=mpl.colormaps["viridis"](norm(d)), lw=1.5, label=f"d{d}")
        ax.set_ylim(2.8, 4.2)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation loss")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "p2c_screen.png")
    plt.close(fig)


def main():
    runs = load_runs([P2A_KEY, P2B_KEY, P2C_KEY], names=list(REUSED_D8.values()))
    for run in runs:
        if run.state != "finished":
            print(f"  not finished: {run.name} ({run.state})")
    runs_by_name = {run.name: run for run in finished(runs)}
    curves = part_a(runs_by_name)
    plot_part_a(curves)
    plot_part_b(runs_by_name, curves)
    plot_part_c(runs_by_name, curves)


if __name__ == "__main__":
    main()
