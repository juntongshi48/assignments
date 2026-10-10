"""Problem 3.2(b) targets: test both hypotheses' predictions at B = 128 and 256 (614.4M tokens).

Hypothesis (i): WD .1 with the fitted LR(B); hypothesis (ii): LR .0015 with the fitted WD(B). Each was run at
its prediction and x/2, x2 around it. Reports each bracket's fitted optimum vs. the prediction and the loss at
each prediction vs. the best sampled loss at that batch (over both brackets).
Writes plots/p32b_targets.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.p32_batch import PREDICTED, round2
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p32ab import FIXED_LR, FIXED_WD, local_optimum, measurements

TARGETS = tuple(PREDICTED)
HYPOTHESES = {"i": ("lr", "LR at WD .1"), "ii": ("wd", "WD at LR .0015")}


def bracket(m, batch, axis):
    pts = sorted((k[1] if axis == "lr" else k[2], v) for k, v in m.items()
                 if k[0] == batch and (k[2] == FIXED_WD if axis == "lr" else k[1] == FIXED_LR))
    return tuple(np.array(v) for v in zip(*pts)) if pts else (np.array([]), np.array([]))


def main():
    m = measurements()
    missing = [(b, h) for b in TARGETS for h, (axis, _) in HYPOTHESES.items() if len(bracket(m, b, axis)[0]) < 3]
    if missing:
        print(f"waiting on target runs: {missing}")
        return

    print(f"{'B':>4} {'hyp':>4}  {'predicted':>9}  {'fitted opt':>10}  {'pred/fit':>8}  {'loss@pred':>9}  "
          f"{'best in bracket':>15}  {'best at this B':>14}  {'gap to best at B':>16}")
    summary = {}
    for b in TARGETS:
        best_at_b = min(v for k, v in m.items() if k[0] == b)
        for h, (axis, _) in HYPOTHESES.items():
            xs, losses = bracket(m, b, axis)
            pred = round2(PREDICTED[b][axis])
            x_opt, _, edge, coefs = local_optimum(xs, losses)
            loss_pred = float(losses[np.isclose(xs, pred)][0])
            summary[(b, h)] = loss_pred - best_at_b
            concave = coefs[2] <= 0
            opt = f"<= {x_opt:.4g}" if concave else f"{x_opt:.4g}"
            ratio = f">= {pred / x_opt:.2f}" if concave else f"{pred / x_opt:.2f}"
            note = ("  (loss rises monotonically: no minimum in bracket, optimum at or below lowest point)" if concave
                    else "  (best point on bracket edge: vertex not bracketed by data)" if edge else "")
            print(f"{b:>4} {h:>4}  {pred:>9g}  {opt:>10}  {ratio:>8}  {loss_pred:>9.4f}  "
                  f"{losses.min():>15.4f}  {best_at_b:>14.4f}  {loss_pred - best_at_b:>+16.4f}" + note)
    for b in TARGETS:
        better = min(HYPOTHESES, key=lambda h: summary[(b, h)])
        print(f"B = {b}: hypothesis ({better}) transfers better "
              f"(gap {summary[(b, 'i')]:+.4f} for (i) vs {summary[(b, 'ii')]:+.4f} for (ii))")

    fig, axes = plt.subplots(len(TARGETS), len(HYPOTHESES), figsize=(12, 8.2))
    for row, b in enumerate(TARGETS):
        best_at_b = min(v for k, v in m.items() if k[0] == b)
        for col, (h, (axis, label)) in enumerate(HYPOTHESES.items()):
            ax = axes[row][col]
            xs, losses = bracket(m, b, axis)
            x_opt, l_opt, edge, (c0, c1, c2) = local_optimum(xs, losses)
            grid = np.geomspace(xs.min() / 1.3, xs.max() * 1.3, 200)
            ax.plot(grid, c0 + c1 * np.log(grid) + c2 * np.log(grid) ** 2, color="0.5", lw=1)
            ax.plot(xs, losses, "o", color="C0", mec="k", ms=7, label="measured")
            pred = round2(PREDICTED[b][axis])
            ax.plot([pred], [losses[np.isclose(xs, pred)][0]], "X", color="C3", mec="k", ms=12,
                    label=f"prediction {pred:g}")
            if c2 <= 0:
                ax.plot([x_opt], [l_opt], "*", color="white", mec="k", ms=16,
                        label=f"no minimum in bracket: optimum <= {x_opt:.3g}")
            else:
                ax.plot([x_opt], [l_opt], "*", color="white" if edge else "gold", mec="k", ms=16,
                        label=f"fitted optimum {x_opt:.3g}" + (" (not bracketed)" if edge else ""))
            ax.axhline(best_at_b, color="C2", ls="--", lw=1, label=f"best at B = {b} (either hypothesis)")
            ax.set(xscale="log", xlabel=label.split(" at ")[0], ylabel="final val loss",
                   title=f"B = {b}, hypothesis ({h}): {label}")
            ax.set_xticks(list(xs), labels=[f"{x:g}" for x in xs])
            ax.minorticks_off()
            ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "p32b_targets.png")


if __name__ == "__main__":
    main()
