"""Problem 3.2(c): momentum ablation at B = 8 and 256 (614.4M tokens), each batch's best LR-WD pair, beta2 = .95.

Final val loss vs. beta1 (beta1 = .9 is the reused 3.2(a, b) run), and the change relative to the beta1 = 0
control, next to the NQM's 3.1(d) Adam result (LR retuned per beta1, sigma = 1, B = 16 and 256).
Writes plots/p32c_momentum.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.p32c_momentum import BEST_PAIR, BETA1S, EXPERIMENT_KEY
from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p32ab import measurements

SEED_STD = 0.0017  # A1 replicate runs at 614M
ALL_BETA1S = tuple(sorted(BETA1S + (.9,)))
# NQM 3.1(d), Adam, beta2 = .95, LR retuned per beta1: best loss relative to beta1 = 0.
NQM_RATIO = {
    16: {0: 1, .5: .959, .8: .933, .9: .910, .95: .857, .98: 1.085, .99: 13.305},
    256: {0: 1, .5: .243, .8: .147, .9: 1.120, .95: 4.137, .98: 8.422, .99: 10.552},
}


def losses():
    """batch -> {beta1: final val loss} at each batch's fixed LR-WD pair."""
    m = measurements()
    out = {b: {.9: m[(b, p["lr"], p["wd"])]} for b, p in BEST_PAIR.items()}
    for r in load_runs([EXPERIMENT_KEY]):
        if r.state == "finished":
            out[r.config["batch_size"]][r.config["beta1"]] = r.val_loss
    return out


def main():
    data = losses()
    missing = [(b, beta1) for b in BEST_PAIR for beta1 in ALL_BETA1S if beta1 not in data[b]]
    if missing:
        print(f"waiting on: {missing}")
    print(f"{'beta1':>6}" + "".join(f"  B={b} (lr {p['lr']:g}, wd {p['wd']:g}): loss, vs beta1=0"
                                    for b, p in BEST_PAIR.items()))
    for beta1 in ALL_BETA1S:
        row = f"{beta1:>6g}"
        for b in BEST_PAIR:
            loss, base = data[b].get(beta1), data[b].get(0.0)
            row += f"  {'':>14}{loss:>8.4f}" if loss is not None else f"  {'':>14}{'-':>8}"
            row += f"  {loss - base:>+8.4f}" if loss is not None and base is not None else f"  {'-':>8}"
        print(row)
    for b in BEST_PAIR:
        if 0.0 in data[b] and len(data[b]) > 1:
            best = min(data[b], key=data[b].get)
            print(f"B = {b}: best beta1 {best:g} ({data[b][best]:.4f}); momentum benefit "
                  f"{data[b][0.0] - data[b][best]:+.4f} vs beta1 = 0 (seed noise ~{SEED_STD})")

    fig, (ax_loss, ax_delta, ax_nqm) = plt.subplots(1, 3, figsize=(17, 4.8))
    ticks = list(ALL_BETA1S)
    pos = {beta1: i for i, beta1 in enumerate(ticks)}  # categorical x: beta1 values are clustered near 1
    for b, color in zip(BEST_PAIR, ("C0", "C3")):
        xs = [x for x in ticks if x in data[b]]
        ys = np.array([data[b][x] for x in xs])
        p = BEST_PAIR[b]
        ax_loss.plot([pos[x] for x in xs], ys, "o-", color=color, mec="k",
                     label=f"B = {b} (lr {p['lr']:g}, wd {p['wd']:g})")
        if 0.0 in data[b]:
            ax_delta.plot([pos[x] for x in xs], ys - data[b][0.0], "o-", color=color, mec="k", label=f"B = {b}")
        nqm_b = 16 if b == 8 else b
        ax_nqm.plot([pos[x] for x in ticks], [NQM_RATIO[nqm_b][x] for x in ticks], "s--", color=color, mec="k",
                    label=f"NQM B = {nqm_b} ({8192 // nqm_b} updates)")
    ax_delta.axhspan(-2 * SEED_STD, 2 * SEED_STD, color="0.85", label=r"$\pm 2\times$ seed std")
    ax_delta.axhline(0, color="0.4", lw=1)
    ax_nqm.axhline(1, color="0.4", lw=1)
    ax_loss.set(ylabel="final val loss", title="LM: loss vs. beta1 (LR, WD fixed; beta2 = .95)")
    ax_delta.set(ylabel=r"loss $-$ loss at beta1 = 0", title="LM: change vs. no-momentum control")
    ax_nqm.set(yscale="log", ylabel="best loss / best loss at beta1 = 0",
               title="NQM 3.1(d), Adam (LR retuned per beta1)")
    ax_nqm.set_yticks([.15, .3, 1, 3, 10], labels=["0.15", "0.3", "1", "3", "10"])
    ax_nqm.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    for ax in (ax_loss, ax_delta, ax_nqm):
        ax.set_xticks(range(len(ticks)), labels=[f"{x:g}" for x in ticks])
        ax.axvline(pos[.95] + (np.sqrt(.95) - .95) / (.98 - .95), color="0.6", ls=":", lw=1)
        ax.set_xlabel(r"beta1 (dotted: $\sqrt{\beta_2}$)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p32c_momentum.png")


if __name__ == "__main__":
    main()
