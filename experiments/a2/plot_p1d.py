"""Problem 1(d) analysis: Hyperball vs. AdamW LR scaling at the three P1(a) budgets.

The Hyperball sweep spans .0003-.03, far wider than a quadratic in log(LR) can describe (the
low-LR side is steep), so each budget is fit locally over a fixed window bracketing the minimum.
Other windows are refit to show how much the optima depend on that choice.
Writes plots/p1d_hyperball.png and prints the optima, both power laws, and the 1.2288B prediction.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import by_budget, fit_lr_law, fit_optimum, savefig
from experiments.a2.plot_p1b import curve_loss
from experiments.a2.provided_sweeps import load

TARGET_TOKENS = 1_228_800_000
WINDOW = (.006, .01, .015, .03)
ALT_WINDOWS = {"3 pt (.006-.015)": (.006, .01, .015), "5 pt (.003-.03)": (.003, .006, .01, .015, .03)}


def restrict(lrs, losses, window):
    keep = np.isin(lrs, window)
    return lrs[keep], losses[keep]


def main():
    hb = by_budget(load("P1d"))
    budgets = np.array(list(hb))
    fits = {t: fit_optimum(*restrict(*hb[t], WINDOW)) for t in budgets}
    hb_opts = np.array([fits[t][1] for t in budgets])
    hb_law = fit_lr_law(budgets, hb_opts)

    aw = by_budget(load("P1a"))
    aw_opts = np.array([fit_optimum(*aw[t])[1] for t in budgets])
    aw_law = fit_lr_law(budgets, aw_opts)

    print(f"Hyperball local fits over LR window {WINDOW}:")
    print(f"{'D (tokens)':>14}  {'lr*':>8}  {'loss*':>7}  {'rms resid':>9}  {'best sampled':>15}  "
          + "  ".join(f"{name:>16}" for name in ALT_WINDOWS))
    for tokens in budgets:
        lrs, losses = hb[tokens]
        coefs, lr_opt, loss_opt = fits[tokens]
        wl, wy = restrict(lrs, losses, WINDOW)
        rms = np.sqrt(np.mean((wy - curve_loss(coefs, wl)) ** 2))
        best = int(np.argmin(losses))
        alts = [fit_optimum(*restrict(lrs, losses, w))[1] for w in ALT_WINDOWS.values()]
        print(f"{tokens:>14,}  {lr_opt:>8.5f}  {loss_opt:>7.4f}  {rms:>9.4f}  "
              f"{lrs[best]:>6g} ({losses[best]:.4f})  " + "  ".join(f"{x:>16.5f}" for x in alts))

    print()
    for name, (a, alpha, predict), opts in (("Hyperball", hb_law, hb_opts), ("AdamW", aw_law, aw_opts)):
        resid = np.log(opts) - np.log(predict(budgets))
        print(f"{name:>9}: lr*(D) = {a:.4g} * D^{alpha:+.3f}   rms log-resid {np.sqrt(np.mean(resid**2)):.3f}"
              f"   -> lr*(1.2288B) = {float(predict(TARGET_TOKENS)):.5f}")
    print("Window sensitivity of the Hyperball prediction at 1.2288B:")
    for name, window in ALT_WINDOWS.items():
        opts = [fit_optimum(*restrict(*hb[t], window))[1] for t in budgets]
        _, alpha, predict = fit_lr_law(budgets, opts)
        print(f"  {name:>16}: alpha = {alpha:+.3f}, lr*(1.2288B) = {float(predict(TARGET_TOKENS)):.5f}")

    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(budgets)))
    fig, (ax_curve, ax_law) = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = np.geomspace(min(WINDOW) / 1.3, max(WINDOW) * 1.3, 200)
    for color, tokens in zip(colors, budgets):
        lrs, losses = hb[tokens]
        coefs, lr_opt, loss_opt = fits[tokens]
        ax_curve.plot(grid, curve_loss(coefs, grid), color=color, lw=1.5)
        ax_curve.plot(lrs, losses, "o", color=color, alpha=0.35, ms=5)
        wl, wy = restrict(lrs, losses, WINDOW)
        ax_curve.scatter(wl, wy, color=color, zorder=3, label=f"D = {tokens / 1e6:.1f}M")
        ax_curve.scatter([lr_opt], [loss_opt], color=color, marker="*", s=180, edgecolor="k", zorder=4)
    ax_curve.set(xscale="log", xlabel="Hyperball peak LR", ylabel="final validation loss", ylim=(2.88, 3.5),
                 title="Hyperball loss vs. LR (faded = outside fit window; stars = optima)")
    ax_curve.legend()

    d_grid = np.geomspace(budgets.min() / 1.3, TARGET_TOKENS * 1.3, 100)
    for name, (a, alpha, predict), opts, color in (("Hyperball", hb_law, hb_opts, "C3"),
                                                     ("AdamW", aw_law, aw_opts, "C0")):
        # Normalize by the 153.6M optimum so both optimizers share one axis despite a ~5x LR offset.
        ref = opts[0]
        ax_law.plot(d_grid, predict(d_grid) / ref, "--", color=color, lw=1.2,
                    label=rf"{name}: $\alpha = {alpha:+.3f}$")
        ax_law.scatter(budgets, opts / ref, color=color, s=60, edgecolor="k", zorder=3)
        ax_law.scatter([TARGET_TOKENS], [predict(TARGET_TOKENS) / ref], color=color, marker="X", s=90,
                       edgecolor="k", zorder=4)
    ax_law.axhline(1, color="0.6", lw=0.8, ls=":")
    ax_law.set(xscale="log", yscale="log", xlabel="training tokens D",
               ylabel="optimal LR / optimal LR at 153.6M",
               title="Relative drift of the optimal LR (X = 1.2288B prediction)")
    ax_law.legend()
    fig.tight_layout()
    savefig(fig, "p1d_hyperball.png")
    target_analysis(float(hb_law[2](TARGET_TOKENS)), float(aw_law[2](TARGET_TOKENS)))


def target_measurements():
    """lr -> final val loss for finished 1.2288B Hyperball runs (cancelled stubs are skipped)."""
    from experiments.a1_basics.results import load_runs
    from experiments.a2.p1d_hyperball import EXPERIMENT_KEY
    runs = [r for r in load_runs([EXPERIMENT_KEY]) if r.state == "finished" and r.val_loss is not None]
    return dict(sorted((float(r.config["learning_rate"]), r.val_loss) for r in runs))


def target_analysis(lr_pred, aw_pred):
    from experiments.a2.p1d_hyperball import NEARBY_LRS, PREDICTED_LR
    measured = target_measurements()
    missing = sorted({PREDICTED_LR, *NEARBY_LRS} - set(measured))
    print(f"\n1.2288B Hyperball target runs finished: {len(measured)}/4"
          + (f"  (waiting on lr = {', '.join(f'{lr:g}' for lr in missing)})" if missing else ""))
    for lr, loss in measured.items():
        print(f"  lr = {lr:<7g} loss = {loss:.4f}")
    if len(measured) < 3:
        return
    lrs, losses = map(np.array, zip(*measured.items()))
    coefs, lr_opt, loss_opt = fit_optimum(lrs, losses)
    best_lr = min(measured, key=measured.get)
    aw_lrs, aw_losses = by_budget(load("P1b"))[TARGET_TOKENS]
    aw_coefs, aw_opt, aw_loss_opt = fit_optimum(aw_lrs, aw_losses)
    aw_best = float(np.min(aw_losses))
    print(f"Target fit over {len(lrs)} LRs: lr* = {lr_opt:.5f}, loss* = {loss_opt:.4f}")
    print(f"Prediction {lr_pred:.5f} (launched {PREDICTED_LR:g}): pred/fit = {PREDICTED_LR / lr_opt:.3f}")
    if PREDICTED_LR in measured:
        print(f"Best sampled: lr = {best_lr:g}, loss {measured[best_lr]:.4f}; prediction's loss "
              f"{measured[PREDICTED_LR]:.4f} (gap {measured[PREDICTED_LR] - measured[best_lr]:+.4f})")
    print(f"AdamW at 1.2288B: P1(a) prediction {aw_pred:.5f}, fitted lr* = {aw_opt:.5f} "
          f"(pred/fit = {aw_pred / aw_opt:.3f}), loss* = {aw_loss_opt:.4f}, "
          f"loss at prediction (read off fit) = {float(curve_loss(aw_coefs, aw_pred)):.4f}")
    print(f"AdamW best sampled at 1.2288B: {aw_best:.4f} (Hyperball best minus AdamW best: "
          f"{measured[best_lr] - aw_best:+.4f})")

    # (name, sampled lrs, losses, fit coefs, lr*, loss*, predicted lr, loss at prediction, measured?)
    models = [
        ("Hyperball", lrs, losses, coefs, lr_opt, loss_opt, PREDICTED_LR,
         measured.get(PREDICTED_LR, float(curve_loss(coefs, PREDICTED_LR))), PREDICTED_LR in measured, "C3"),
        ("AdamW", aw_lrs, aw_losses, aw_coefs, aw_opt, aw_loss_opt, aw_pred,
         float(curve_loss(aw_coefs, aw_pred)), False, "C0"),
    ]
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, m_lrs, m_losses, m_coefs, m_opt, m_loss_opt, m_pred, m_pred_loss, ran, color in models:
        grid = np.geomspace(m_lrs.min() / 1.3, m_lrs.max() * 1.3, 200)
        fit = curve_loss(m_coefs, grid)
        pred_style = dict(color=color, marker="X", s=120, edgecolor="k", zorder=4,
                          label=f"{name} prediction {m_pred:.4f}" + ("" if ran else " (on fit, not run)"))
        if not ran:
            pred_style.update(facecolor="white", edgecolor=color, linewidths=2)
        ax_abs.plot(grid, fit, color=color, lw=1.5)
        ax_abs.scatter(m_lrs, m_losses, color=color, zorder=3, label=f"{name} measured")
        ax_abs.scatter([m_opt], [m_loss_opt], color=color, marker="*", s=220, edgecolor="k", zorder=5,
                       label=f"{name} fitted optimum {m_opt:.4f}")
        ax_abs.scatter([m_pred], [m_pred_loss], **pred_style)
        ax_rel.plot(grid / m_opt, fit - m_loss_opt, color=color, lw=1.5, label=f"{name} fit")
        ax_rel.scatter(m_lrs / m_opt, m_losses - m_loss_opt, color=color, zorder=3)
        ax_rel.scatter([m_pred / m_opt], [m_pred_loss - m_loss_opt],
                       **{**pred_style, "label": f"{name} prediction: {m_pred / m_opt:.3f}x optimum"})
    ticks = [.0015, .003, .006, .0105, .015, .021]
    ax_abs.set(xscale="log", xlabel="peak LR", ylabel="final validation loss",
               title="AdamW vs. Hyperball at D = 1.2288B")
    ax_abs.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax_abs.minorticks_off()
    ax_abs.legend(fontsize=8, ncol=2, loc="upper center")
    ax_rel.axvline(1, color="0.6", lw=0.8, ls=":")
    ax_rel.axhline(0, color="0.6", lw=0.8, ls=":")
    rel_ticks = [.4, .5, .7, 1, 1.4, 2]
    ax_rel.set(xscale="log", xlabel="peak LR / fitted optimal LR", ylabel="loss above fitted minimum",
               title="Prediction accuracy and curve width, each model at its own optimum")
    ax_rel.set_xticks(rel_ticks, labels=[f"{t:g}x" for t in rel_ticks])
    ax_rel.minorticks_off()
    ax_rel.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p1d_target.png")


if __name__ == "__main__":
    main()
