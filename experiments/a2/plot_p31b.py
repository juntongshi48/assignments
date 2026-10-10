"""Problem 3.1(b): RMSProp and Adam on the 2D noisy quadratic, LR tuned separately per batch.

For each optimizer: fit eta*(B) = c B^p over B = 1..64, predict B = 256 and test it (the predicted LR is
added to the B = 256 sweep, on the same noise draws), and extend through B = 512. Plain SGD's exact optimum
is shown for reference. Writes plots/p31b_adaptive_scaling.png and plots/p31b_adaptive_curves.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.nqm import H_2D, INIT_VAR_2D, power_law, sgd_exact, tune
from experiments.a2.plot_p1a import savefig

SOURCE = (1, 2, 4, 8, 16, 32, 64)
EXTENDED = (128, 256, 512)
TARGET = 256
N_SAMPLES = 4096
COARSE_LRS = np.geomspace(1e-5, 3.0, 45)


def batch_study(optimizer, *, h=H_2D, init_var=INIT_VAR_2D, noise_var=1.0, n_samples=N_SAMPLES,
                coarse_lrs=COARSE_LRS):
    """Tune LR at every batch; fit the source power law; test the B = 256 prediction.

    Source optima on the edge of the LR grid are excluded from the fit. Returns dict(opt={B: (lr*, best
    loss, at_edge)}, curves={B: (lrs, loss)}, fit=(c, p, predict), n_fit, lr_pred, loss_pred, loss_tuned)
    where the last two are measured on the same B = 256 sweep.
    """
    kw = dict(h=h, init_var=init_var, noise_var=noise_var, n_samples=n_samples)
    opt, curves = {}, {}
    for b in SOURCE:
        lrs, loss, _, lr_opt, best, edge = tune(optimizer, b, coarse_lrs, seed=b, **kw)
        opt[b], curves[b] = (lr_opt, best, edge), (lrs, loss)
    usable = [b for b in SOURCE if not opt[b][2]]
    fit = power_law(usable, [opt[b][0] for b in usable])
    lr_pred = float(fit[2](TARGET))
    for b in EXTENDED:
        lrs, loss, _, lr_opt, best, edge = tune(optimizer, b, coarse_lrs, seed=b,
                                                extra_lrs=[lr_pred] if b == TARGET else [], **kw)
        opt[b], curves[b] = (lr_opt, best, edge), (lrs, loss)
    lrs, loss = curves[TARGET]
    loss_pred = float(loss[int(np.argmin(np.abs(lrs - lr_pred)))])
    return dict(opt=opt, curves=curves, fit=fit, n_fit=len(usable), lr_pred=lr_pred, loss_pred=loss_pred,
                loss_tuned=opt[TARGET][1])


def sgd_reference(batches, **kw):
    fine = np.geomspace(1e-5, 0.9999 * 2 / kw.get("h", H_2D).max(), 3000)
    out = {}
    for b in batches:
        losses = np.array([sgd_exact(lr, b, **kw) for lr in fine])
        out[b] = (float(fine[np.argmin(losses)]), float(losses.min()))
    return out


def report(name, study):
    c, p, predict = study["fit"]
    print(f"\n{name}: eta*(B) = {c:.3e} * B^{p:.3f} over B = 1..64")
    print(f"{'B':>4}  {'eta*':>9}  {'eta*/sqrt(B)':>12}  {'fit pred':>9}  {'best loss':>10}")
    for b, (lr_opt, best, edge) in study["opt"].items():
        print(f"{b:>4}  {lr_opt:>9.5f}  {lr_opt / np.sqrt(b):>12.2e}  {float(predict(b)):>9.5f}  {best:>10.3e}"
              + ("  (argmin at grid edge)" if edge else ""))
    tuned = study["opt"][TARGET][0]
    print(f"B = {TARGET}: predicted LR {study['lr_pred']:.5f} vs tuned {tuned:.5f} (ratio {study['lr_pred'] / tuned:.2f}); "
          f"loss at predicted {study['loss_pred']:.3e} vs tuned {study['loss_tuned']:.3e} "
          f"(ratio {study['loss_pred'] / study['loss_tuned']:.2f})")


def main():
    studies = {name: batch_study(name) for name in ("rmsprop", "adam")}
    for name, study in studies.items():
        report(name.upper() if name == "rmsprop" else name.capitalize(), study)
    batches = SOURCE + EXTENDED
    sgd = sgd_reference(batches)

    colors = {"sgd": "0.4", "rmsprop": "C0", "adam": "C3"}
    labels = {"sgd": "SGD (exact)", "rmsprop": "RMSProp", "adam": "Adam"}
    b_arr = np.array(batches)
    b_grid = np.geomspace(1, 512, 100)
    src = np.isin(b_arr, SOURCE)
    fig, (ax_lr, ax_norm, ax_loss) = plt.subplots(1, 3, figsize=(16, 4.6))
    ax_lr.plot(b_arr, [sgd[b][0] for b in batches], ":", color=colors["sgd"], label="SGD exact optimum (p = 1)")
    ax_loss.plot(b_arr, [sgd[b][1] for b in batches], ":", color=colors["sgd"], label="SGD (exact)")
    for name, study in studies.items():
        c, p, predict = study["fit"]
        lr_opt = np.array([study["opt"][b][0] for b in batches])
        color = colors[name]
        for ax, values, scale in ((ax_lr, lr_opt, 1), (ax_norm, lr_opt / np.sqrt(b_arr), np.sqrt(b_grid))):
            ax.plot(b_arr[src], values[src], "o", color=color, mec="k", ms=7,
                    label=f"{labels[name]} optimum (fit)" if ax is ax_lr else None)
            ax.plot(b_arr[~src], values[~src], "o", mfc="white", color=color, mew=1.5, ms=7,
                    label=f"{labels[name]} optimum (held out)" if ax is ax_lr else None)
            ax.plot(b_grid, predict(b_grid) / scale, "--", color=color, lw=1.2,
                    label=rf"{labels[name]} fit $\propto B^{{{p:.3f}}}$")
        ax_lr.plot([TARGET], [study["lr_pred"]], "X", color=color, mec="k", ms=10)
        ax_loss.plot(b_arr, [study["opt"][b][1] for b in batches], "o-", color=color, mec="k", ms=6,
                     label=f"{labels[name]} (tuned LR)")
        ax_loss.plot([TARGET], [study["loss_pred"]], "X", color=color, mec="k", ms=10,
                     label=f"{labels[name]} at predicted LR (B = 256)")
    for ax in (ax_lr, ax_norm, ax_loss):
        ax.set(xscale="log", yscale="log", xlabel="batch size B")
        ax.set_xticks(batches, labels=[str(b) for b in batches])
        ax.legend(fontsize=7)
    ax_lr.set(ylabel="optimal LR", title="Optimal LR vs. batch (X = prediction at B = 256)")
    ax_norm.set(ylabel=r"optimal LR / $\sqrt{B}$", title=r"Optimal LR / $\sqrt{B}$ (flat = square-root scaling)")
    ax_loss.set(ylabel="best final loss", title="Best loss after tuning LR")
    fig.tight_layout()
    savefig(fig, "p31b_adaptive_scaling.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(batches)))
    for ax, (name, study) in zip(axes, studies.items()):
        for color, b in zip(cmap, batches):
            lrs, loss = study["curves"][b]
            ax.plot(lrs, loss, color=color, lw=1.4, label=f"B = {b}")
            ax.plot([study["opt"][b][0]], [study["opt"][b][1]], "o", color=color, mec="k", ms=5)
        ax.set(xscale="log", yscale="log", ylim=(1e-4, 3), xlabel="learning rate", title=f"{labels[name]}: loss vs. LR")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(r"$L_N(B, \eta)$")
    fig.tight_layout()
    savefig(fig, "p31b_adaptive_curves.png")


if __name__ == "__main__":
    main()
