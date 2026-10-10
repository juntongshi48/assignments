"""Problem 3.1(c): how gradient-noise scale and curvature change the adaptive-optimizer batch rule.

Noise covariance sigma^2 I with sigma in {1, 10, 100, 300}, on the 2D quadratic (H = diag(1, 10)) and the
scalar quadratic f = w^2 / 2 with w0 ~ N(0, 2). For RMSProp and Adam: source fit over B = 1..64 and the
B = 256 prediction test (plot_p31b.batch_study). SGD's exact optimum gives the reference exponent.
Studies run in parallel and are cached in experiments/a2/cache/p31c.json.
Writes plots/p31c_exponents.png and plots/p31c_optima.png.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.nqm import H_2D, INIT_VAR_2D, power_law, sgd_exact
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p31b import EXTENDED, SOURCE, TARGET, batch_study

SIGMAS = (1, 10, 100, 300)
MODELS = {"2D": (H_2D, INIT_VAR_2D), "scalar": (np.array([1.0]), np.array([2.0]))}
OPTIMIZERS = ("rmsprop", "adam")
N_SAMPLES = 2048
COARSE_LRS = np.geomspace(1e-7, 3.0, 60)
CACHE = Path(__file__).resolve().parent / "cache" / "p31c.json"


def _run(job):
    model, sigma, optimizer = job
    h, init_var = MODELS[model]
    s = batch_study(optimizer, h=h, init_var=init_var, noise_var=float(sigma) ** 2, n_samples=N_SAMPLES,
                    coarse_lrs=COARSE_LRS)
    return job, dict(opt={str(b): v for b, v in s["opt"].items()}, c=s["fit"][0], p=s["fit"][1],
                     n_fit=s["n_fit"], lr_pred=s["lr_pred"], loss_pred=s["loss_pred"], loss_tuned=s["loss_tuned"])


def studies():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = [(m, s, o) for m in MODELS for s in SIGMAS for o in OPTIMIZERS if f"{m}|{s}|{o}" not in cache]
    if jobs:
        print(f"running {len(jobs)} studies in parallel")
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
            for (m, s, o), result in pool.map(_run, jobs):
                cache[f"{m}|{s}|{o}"] = result
                print(f"  done {m} sigma={s} {o}: p = {result['p']:.3f}")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def sgd_exponent(model, sigma):
    h, init_var = MODELS[model]
    fine = np.geomspace(1e-7, 0.9999 * 2 / h.max(), 3000)
    opt = [fine[np.argmin([sgd_exact(lr, b, h=h, init_var=init_var, noise_var=sigma**2) for lr in fine])]
           for b in SOURCE]
    return power_law(SOURCE, opt)[1]


def sgd_best_loss(model, sigma, batches):
    h, init_var = MODELS[model]
    fine = np.geomspace(1e-7, 0.9999 * 2 / h.max(), 3000)
    return {b: min(sgd_exact(lr, b, h=h, init_var=init_var, noise_var=sigma**2) for lr in fine) for b in batches}


def plot_best_losses(cache):
    """Best final loss after tuning LR vs. batch: SGD (exact), RMSProp and Adam, per model and noise scale."""
    batches = SOURCE + EXTENDED
    fig, axes = plt.subplots(len(MODELS), len(SIGMAS), figsize=(18, 8), sharex=True)
    print("\nBest loss relative to SGD (adaptive / SGD) at B = 1, 16, 64, 128, 256, 512; SGD absolute in brackets")
    for row, m in enumerate(MODELS):
        h, init_var = MODELS[m]
        for col, s in enumerate(SIGMAS):
            ax = axes[row][col]
            sgd = sgd_best_loss(m, s, batches)
            ax.plot(batches, [sgd[b] for b in batches], "s-", color="0.4", ms=5, label="SGD (exact)")
            for o, color, name in (("rmsprop", "C0", "RMSProp"), ("adam", "C3", "Adam")):
                best = [cache[f"{m}|{s}|{o}"]["opt"][str(b)][1] for b in batches]
                ax.plot(batches, best, "o-", color=color, mec="k", ms=5, label=name)
                print(f"  {m:>6} sigma={s:<3} {name:>7}: "
                      + "  ".join(f"{best[batches.index(b)] / sgd[b]:5.2f}" for b in (1, 16, 64, 128, 256, 512)))
            b_c = s**2 / (h**2 * init_var).max()
            if 1 <= b_c <= 512:
                ax.axvline(b_c, color="C2", lw=1, ls=":", label=r"$B_c$: signal = noise at init")
            ax.set(xscale="log", yscale="log", title=rf"{m}, $\sigma$ = {s}")
            # Give every panel at least a 3x range so small differences are not stretched to fill it.
            values = [line.get_ydata() for line in ax.get_lines() if len(line.get_ydata()) == len(batches)]
            lo, hi = float(np.min(values)), float(np.max(values))
            half = np.sqrt(max(hi / lo, 3.0)) * 1.15
            ax.set_ylim(np.sqrt(lo * hi) / half, np.sqrt(lo * hi) * half)
            y0, y1 = ax.get_ylim()
            ticks = [m * 10.0**k for k in range(int(np.floor(np.log10(y0))), int(np.ceil(np.log10(y1))) + 1)
                     for m in (1, 2, 3, 5) if y0 <= m * 10.0**k <= y1]
            ax.set_yticks(ticks, labels=[f"{t:g}" for t in ticks], fontsize=7)
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.set_xticks(batches, labels=[str(b) for b in batches], fontsize=7)
            ax.minorticks_off()
            ax.legend(fontsize=7)
        axes[row][0].set_ylabel("best final loss (LR tuned)")
    for ax in axes[-1]:
        ax.set_xlabel("batch size B")
    fig.suptitle("Best final loss after tuning LR: SGD vs. RMSProp vs. Adam", y=1.01)
    fig.tight_layout()
    savefig(fig, "p31c_best_loss.png")


def main():
    cache = studies()
    plot_best_losses(cache)
    sgd_p = {(m, s): sgd_exponent(m, s) for m in MODELS for s in SIGMAS}

    print(f"\n{'model':>6} {'sigma':>5} {'opt':>8}  {'p (B=1..64)':>11}  {'n fit':>5}  {'pred/tuned LR @256':>18}  "
          f"{'loss pred/tuned':>15}  {'SGD p':>6}  {'B_c = sigma^2/(h^2 var0)':>24}")
    for m in MODELS:
        h, init_var = MODELS[m]
        # Batch below which noise sigma^2 / B exceeds the largest initial squared gradient h^2 var0.
        b_c = np.array(SIGMAS) ** 2 / (h**2 * init_var).max()
        for s, bc in zip(SIGMAS, b_c):
            for o in OPTIMIZERS:
                r = cache[f"{m}|{s}|{o}"]
                tuned = r["opt"][str(TARGET)][0]
                edges = [b for b in SOURCE if r["opt"][str(b)][2]]
                print(f"{m:>6} {s:>5} {o:>8}  {r['p']:>11.3f}  {r['n_fit']:>5}  {r['lr_pred'] / tuned:>18.2f}  "
                      f"{r['loss_pred'] / r['loss_tuned']:>15.3f}  {sgd_p[(m, s)]:>6.3f}  {bc:>24.1f}"
                      + (f"  edge at B={edges}" if edges else ""))

    styles = {"2D": "-", "scalar": "--"}
    colors = {"rmsprop": "C0", "adam": "C3"}
    fig, (ax_p, ax_lr, ax_loss) = plt.subplots(1, 3, figsize=(16, 4.4))
    for m in MODELS:
        for o in OPTIMIZERS:
            rs = [cache[f"{m}|{s}|{o}"] for s in SIGMAS]
            label = f"{'RMSProp' if o == 'rmsprop' else 'Adam'}, {m}"
            ax_p.plot(SIGMAS, [r["p"] for r in rs], "o" + styles[m], color=colors[o], mec="k", label=label)
            ax_lr.plot(SIGMAS, [r["lr_pred"] / r["opt"][str(TARGET)][0] for r in rs], "o" + styles[m],
                       color=colors[o], mec="k", label=label)
            ax_loss.plot(SIGMAS, [r["loss_pred"] / r["loss_tuned"] for r in rs], "o" + styles[m],
                         color=colors[o], mec="k", label=label)
        ax_p.plot(SIGMAS, [sgd_p[(m, s)] for s in SIGMAS], "s" + styles[m], color="0.5", ms=5,
                  label=f"SGD (exact), {m}")
    for y, text in ((1.0, "linear"), (0.5, "square root")):
        ax_p.axhline(y, color="0.7", lw=0.8, ls=":")
        ax_p.text(1.05, y + 0.015, text, fontsize=8, color="0.4")
    ax_lr.axhline(1, color="0.7", lw=0.8)
    ax_loss.axhline(1, color="0.7", lw=0.8)
    for ax, ylabel, title in ((ax_p, "fitted exponent p", "Source exponent (B = 1..64) vs. noise scale"),
                              (ax_lr, "predicted / tuned LR at B = 256", "LR transfer to B = 256"),
                              (ax_loss, "loss at predicted / tuned LR", "Loss cost of the transferred LR")):
        ax.set(xscale="log", xlabel=r"gradient-noise scale $\sigma$", ylabel=ylabel, title=title)
        ax.set_xticks(SIGMAS, labels=[str(s) for s in SIGMAS])
        ax.minorticks_off()
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "p31c_exponents.png")

    batches = SOURCE + EXTENDED
    fig, axes = plt.subplots(len(MODELS), len(OPTIMIZERS), figsize=(12, 8), sharex=True)
    sigma_colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(SIGMAS)))
    for row, m in enumerate(MODELS):
        for col, o in enumerate(OPTIMIZERS):
            ax = axes[row][col]
            for color, s in zip(sigma_colors, SIGMAS):
                r = cache[f"{m}|{s}|{o}"]
                lr = np.array([r["opt"][str(b)][0] for b in batches])
                norm = lr / np.sqrt(batches) / (lr[0] / 1.0)
                ax.plot(batches, norm, "o-", color=color, mec="k", ms=5, label=rf"$\sigma$ = {s} (p = {r['p']:.2f})")
            ax.set(xscale="log", yscale="log",
                   title=f"{'RMSProp' if o == 'rmsprop' else 'Adam'}, {m}: optimal LR / sqrt(B), normalized at B = 1")
            ax.set_xticks(batches, labels=[str(b) for b in batches])
            ax.minorticks_off()
            ax.legend(fontsize=7)
            ax.axvline(64, color="0.7", lw=0.8, ls=":")
        axes[row][0].set_ylabel(r"$\eta^*(B) / \sqrt{B}$, relative to B = 1")
    for ax in axes[-1]:
        ax.set_xlabel("batch size B (dotted: end of source range)")
    fig.tight_layout()
    savefig(fig, "p31c_optima.png")


if __name__ == "__main__":
    main()
