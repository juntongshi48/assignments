"""Problem 2(c) analysis: does the product law give an effective recipe at 2.4576B tokens?

Compares the product-law WD at peak LR .003 against baseline (i), the best 153.6M pair reused without
retuning, and baseline (ii), the best of WD {.05, .1, .2} at LR .003 (WD .1 is the supplied P1b run).
Writes plots/p2c_joint_rule.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.p2c_joint_rule import BASELINE_I, BASELINE_II, EXPERIMENT_KEY, PREDICTED, TARGET_TOKENS
from experiments.a2.plot_p1a import by_budget, fit_lr_law, savefig
from experiments.a2.plot_p2a import fit_paraboloid, grids
from experiments.a2.provided_sweeps import load


def measurements():
    """(lr, wd) -> final val loss at 2.4576B: finished target runs plus the supplied (.003, .1) run."""
    out = {(float(r.config["learning_rate"]), float(r.config["weight_decay"])): r.val_loss
           for r in load_runs([EXPERIMENT_KEY]) if r.state == "finished" and r.val_loss is not None}
    lrs, losses = by_budget(load("P1b"))[TARGET_TOKENS]
    out[(.003, .1)] = float(losses[np.isclose(lrs, .003)][0])
    return out


def main():
    data = grids()
    budgets = np.array(list(data))
    fits = {t: fit_paraboloid(*data[t]) for t in budgets}
    a, alpha, predict = fit_lr_law(budgets, [fits[t]["lr"] * fits[t]["wd"] for t in budgets])
    print(f"Product law (153.6M-1.2288B): eta*lambda = {a:.4g} * D^{alpha:.3f} -> "
          f"{float(predict(TARGET_TOKENS)):.3e} at 2.4576B; WD at LR .003 = {float(predict(TARGET_TOKENS)) / .003:.4f}")

    m = measurements()
    needed = [PREDICTED, BASELINE_I, *BASELINE_II]
    missing = [p for p in needed if p not in m]
    print(f"\nRuns at 2.4576B: {len(needed) - len(missing)}/{len(needed)} new runs finished"
          + (f"  (waiting on {missing})" if missing else ""))
    for (lr, wd), loss in sorted(m.items()):
        print(f"  lr {lr:<6g} wd {wd:<6g} loss {loss:.4f}")
    if missing:
        return

    pred = m[PREDICTED]
    base_i = m[BASELINE_I]
    sweep = {wd: m[(.003, wd)] for wd in (.05, .1, .2)}
    best_wd = min(sweep, key=sweep.get)
    print(f"\nPrediction (lr .003, wd {PREDICTED[1]:g}): {pred:.4f}")
    print(f"Baseline (i)  (lr {BASELINE_I[0]:g}, wd {BASELINE_I[1]:g}): {base_i:.4f}   gap {pred - base_i:+.4f}")
    print(f"Baseline (ii) best of WD {{.05, .1, .2}}: wd {best_wd:g} {sweep[best_wd]:.4f}   "
          f"gap {pred - sweep[best_wd]:+.4f}")

    wds = np.array(sorted({wd for (lr, wd) in m if lr == .003}))
    losses = np.array([m[(.003, wd)] for wd in wds])
    c2, c1, c0 = np.polyfit(np.log(wds), losses, 2)
    wd_opt = float(np.exp(-c1 / (2 * c2))) if c2 > 0 else float("nan")
    print(f"Quadratic in log(WD) over {len(wds)} points at LR .003: fitted optimal WD {wd_opt:.4f} "
          f"(prediction {PREDICTED[1]:g}, ratio {PREDICTED[1] / wd_opt:.2f})")

    fig, (ax_wd, ax_law) = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = np.geomspace(wds.min() / 1.3, wds.max() * 1.3, 200)
    ax_wd.plot(grid, c0 + c1 * np.log(grid) + c2 * np.log(grid) ** 2, color="0.4", lw=1.4,
               label="quadratic fit in log(WD)")
    ax_wd.scatter([w for w in wds if w != PREDICTED[1]], [m[(.003, w)] for w in wds if w != PREDICTED[1]],
                  color="C2", s=60, edgecolor="k", zorder=3, label="baseline (ii): WD .05/.1/.2")
    ax_wd.scatter([PREDICTED[1]], [pred], marker="X", color="C3", s=130, edgecolor="k", zorder=4,
                  label=f"product-law WD {PREDICTED[1]:g}")
    if np.isfinite(wd_opt):
        ax_wd.axvline(wd_opt, color="gold", lw=1.5, label=f"fitted optimal WD {wd_opt:.3f}")
    # Baseline (i) sits far above this range, so state it in text instead of squashing the axis.
    ax_wd.text(0.02, 0.97, f"baseline (i): LR {BASELINE_I[0]:g}, WD {BASELINE_I[1]:g} = {base_i:.4f}\n"
                           f"(off-scale, +{base_i - pred:.4f} above the prediction)",
               transform=ax_wd.transAxes, va="top", fontsize=8, color="C0")
    ax_wd.set(xscale="log", xlabel="weight decay (peak LR .003)", ylabel="final val loss",
              title="D = 2.4576B: product-law WD vs. baselines")
    ax_wd.set_xticks([.05, .1, .2], labels=[".05", ".1", ".2"])
    ax_wd.minorticks_off()
    ax_wd.legend(fontsize=8)

    d_grid = np.geomspace(budgets.min() / 1.3, TARGET_TOKENS * 1.3, 100)
    ax_law.plot(budgets, [fits[t]["lr"] * fits[t]["wd"] for t in budgets], "o", color="C4", ms=8, mec="k",
                label="fitted optima (fit budgets)")
    ax_law.plot(d_grid, predict(d_grid), "--", color="C4", label=rf"product law $\propto D^{{{alpha:.3f}}}$")
    ax_law.plot([TARGET_TOKENS], [float(predict(TARGET_TOKENS))], "X", color="C3", ms=11, mec="k",
                label="prediction at 2.4576B")
    if np.isfinite(wd_opt):
        ax_law.plot([TARGET_TOKENS], [.003 * wd_opt], "*", color="gold", ms=16, mec="k",
                    label="measured: .003 x fitted optimal WD")
    ax_law.set(xscale="log", yscale="log", xlabel="training tokens D", ylabel=r"$\eta\lambda$",
               title="Product law extrapolated one doubling")
    ticks = [2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3]
    ax_law.set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax_law.minorticks_off()
    ax_law.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p2c_joint_rule.png")


if __name__ == "__main__":
    main()
