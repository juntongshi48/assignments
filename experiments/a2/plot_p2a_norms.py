"""Problem 2(a) follow-up: does WD = .1 control the weight norm at small budgets?

Uses our A1 d6 runs at the default LR .003, WD .1, batch 64 at four budgets (the same budgets as P2a).
They log per-module parameter RMS every 100 steps. The RMS of the weight-decayed matrices (all linear
weights, including the readout) is rebuilt from the per-module values weighted by parameter count;
embeddings receive no WD and serve as a control.
Writes plots/p2a_weight_norms.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import history, load_runs
from experiments.a2.plot_p1a import savefig
from model_config import LMConfig
from modeling import AutoregressiveLM

RUNS = {  # tokens -> A1 run name (d6, LR .003, WD .1, batch 64)
    153_600_000: "model-d6-lr0.003-tok154M-p2b-v1",
    307_200_000: "model-d6-lr0.003-tok307M-p2b-v1",
    614_400_000: "model-d6-lr0.003-tok614M-p2a-v1",
    1_228_800_000: "model-d6-lr0.003-tok1.23B-p2b-v1",
}
LR, WD = .003, .1
# Matched d8 pair at 614.4M from A1 (LR .003, batch 64): WD = 0 control vs. the WD .1 baseline.
D8_RUNS = {0.0: "model-d8-lr0.003-tok614M-wd0.0-p6-v1", 0.1: "model-d8-lr0.003-tok614M-modal"}
D6_CONFIG = LMConfig("d6", 4096, 1024, 384, 1344, 6, 6, 6)
D8_CONFIG = LMConfig("d8", 4096, 1024, 512, 1792, 8, 8, 8)


def decayed_matrix_sizes(config=D6_CONFIG):
    """Module name -> parameter count for every linear weight of the model (all receive WD)."""
    model = AutoregressiveLM(config)
    return {name.removesuffix(".weight"): p.numel() for name, p in model.named_parameters()
            if p.ndim == 2 and "embed" not in name}


def load_curves(runs, sizes):
    """Finished runs only: tokens -> RMS curves, WD, n = lr*wd*T/2 and final val loss."""
    keys = [f"logging/rms/{m}/parameter" for m in sizes] + ["logging/rms/model.embed_tokens/parameter"]
    curves = {}
    for run in runs:
        if run.state != "finished":
            continue
        h = history(run, keys)
        decayed = np.sqrt(sum(n * h[f"logging/rms/{m}/parameter"] ** 2 for m, n in sizes.items())
                          / sum(sizes.values()))
        total, wd = run.config["total_steps"], run.config["weight_decay"]
        curves[run.config["num_train_sequences"] * 1024] = dict(
            steps=h["optimizer_step"], total=total, wd=wd, n=LR * wd * total / 2, val_loss=run.val_loss,
            decayed=decayed, embed=h["logging/rms/model.embed_tokens/parameter"])
    return dict(sorted(curves.items()))


def print_table(title, curves):
    print(f"\n{title}")
    print(f"{'D':>8}  {'WD':>5}  {'n':>5}  {'start':>6}  {'peak (at)':>14}  {'end':>6}  {'end/peak':>8}  "
          f"{'embed end':>9}  {'val loss':>8}")
    for tokens, c in curves.items():
        i = int(np.argmax(c["decayed"]))
        print(f"{tokens / 1e6:>7.1f}M  {c['wd']:>5g}  {c['n']:>5.2f}  {c['decayed'][0]:>6.4f}  "
              f"{c['decayed'][i]:.4f} ({c['steps'][i] / c['total']:>3.0%})  {c['decayed'][-1]:>6.4f}  "
              f"{c['decayed'][-1] / c['decayed'][i]:>8.2f}  {c['embed'][-1]:>9.4f}  {c['val_loss']:>8.4f}")


def plot_better_wd(base, better):
    """Same panels as the WD .1 figure, with each budget's better-WD run (solid) over its WD .1 run (dashed)."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(base)))
    fig, (ax_frac, ax_step, ax_shape) = plt.subplots(1, 3, figsize=(16, 4.4))
    for color, (tokens, c) in zip(colors, base.items()):
        frac = c["steps"] / c["total"]
        ax_frac.plot(frac, c["decayed"], "--", color=color, lw=1.2, alpha=0.6)
        ax_step.plot(c["steps"], c["decayed"], "--", color=color, lw=1.2, alpha=0.6)
        if tokens not in better:
            continue
        b = better[tokens]
        label = f"D = {tokens / 1e9:.4g}B: WD {b['wd']:g} (n = {b['n']:.2f})"
        ax_frac.plot(b["steps"] / b["total"], b["decayed"], color=color, lw=1.8, label=label)
        ax_step.plot(b["steps"], b["decayed"], color=color, lw=1.8, label=label)
    for series, marker, name in ((base, "s", "WD .1"), (better, "o", "better WD")):
        n = [c["n"] for c in series.values()]
        shape = [c["decayed"][-1] / c["decayed"].max() for c in series.values()]
        ax_shape.plot(n, shape, marker, ms=8, mec="k", label=name)
    ax_frac.set(xlabel="fraction of training", ylabel="RMS of weight-decayed matrices",
                title="Better WD (solid) vs. WD .1 (dashed)")
    ax_step.set(xscale="log", xlabel="optimizer step", ylabel="RMS of weight-decayed matrices",
                title="Same curves vs. optimizer step")
    ax_shape.set(xscale="log", xlabel=r"WD timescales per run, $n = \lambda\eta T/2$",
                 ylabel="final RMS / peak RMS", title="How far WD pulls the norm back, vs. n")
    for ax in (ax_frac, ax_step, ax_shape):
        ax.legend(fontsize=8)
    fig.suptitle("A1 d6 runs at LR .003 with each budget's better WD", y=1.02)
    fig.tight_layout()
    savefig(fig, "p2a_weight_norms_better_wd.png")


def main():
    from experiments.a2.p2a_norm_followup import EXPERIMENT_KEY
    sizes = decayed_matrix_sizes()
    curves = load_curves(load_runs([], names=list(RUNS.values())), sizes)
    print_table("WD = .1 (A1 d6 budget series)", curves)
    better = load_curves(load_runs([EXPERIMENT_KEY]), sizes)
    if better:
        print_table("Better WD per budget (p2a_norm_followup)", better)
        plot_better_wd(curves, better)
    else:
        print(f"\nNo finished {EXPERIMENT_KEY} runs yet; skipping the better-WD figure.")

    d8_sizes = decayed_matrix_sizes(D8_CONFIG)
    d8_runs = {run.name: run for run in load_runs([], names=list(D8_RUNS.values()))}
    d8 = {wd: next(iter(load_curves([d8_runs[name]], d8_sizes).values())) for wd, name in D8_RUNS.items()}
    print("\nd8 at 614.4M (WD = 0 control vs. WD .1)")
    for wd, c in d8.items():
        i = int(np.argmax(c["decayed"]))
        print(f"  WD {wd:<4g} n = {c['n']:.2f}  start {c['decayed'][0]:.4f}  peak {c['decayed'][i]:.4f} "
              f"({c['steps'][i] / c['total']:.0%})  end {c['decayed'][-1]:.4f}  "
              f"end/peak {c['decayed'][-1] / c['decayed'][i]:.2f}  end/start {c['decayed'][-1] / c['decayed'][0]:.2f}  "
              f"val loss {c['val_loss']:.4f}")

    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(curves)))
    fig, (ax_frac, ax_step, ax_emb, ax_d8) = plt.subplots(1, 4, figsize=(21, 4.4))
    for color, (tokens, c) in zip(colors, curves.items()):
        frac = c["steps"] / c["total"]
        label = f"D = {tokens / 1e9:.4g}B (n = {c['n']:.2f})"
        ax_frac.plot(frac, c["decayed"], color=color, lw=1.8, label=label)
        ax_step.plot(c["steps"], c["decayed"], color=color, lw=1.8, label=label)
        ax_emb.plot(frac, c["embed"], color=color, lw=1.8, label=label)
    ax_frac.set(xlabel="fraction of training", ylabel="RMS of weight-decayed matrices",
                title="Weight-decayed matrices vs. progress")
    ax_step.set(xscale="log", xlabel="optimizer step", ylabel="RMS of weight-decayed matrices",
                title="Same curves vs. optimizer step")
    ax_emb.set(xlabel="fraction of training", ylabel="embedding RMS",
               title="Control: embeddings (no WD) never turn around")
    for wd, style in ((0.0, dict(color="C3")), (0.1, dict(color="C0"))):
        c = d8[wd]
        ax_d8.plot(c["steps"] / c["total"], c["decayed"], lw=1.8, **style,
                   label=f"WD {wd:g} (n = {c['n']:.2f}), val {c['val_loss']:.4f}")
    ax_d8.set(xlabel="fraction of training", ylabel="RMS of weight-decayed matrices",
              title="Real control: d8 at 0.6144B, WD 0 vs. WD .1")
    for ax in (ax_frac, ax_step, ax_emb, ax_d8):
        ax.legend(fontsize=8)
    fig.suptitle("A1 runs at LR .003. Panels 1-3: d6 at WD .1 across budgets; only longer runs let WD pull the "
                 "norm back down. Panel 4: d8 at 0.6144B, WD 0 vs. WD .1", y=1.02)
    fig.tight_layout()
    savefig(fig, "p2a_weight_norms.png")


if __name__ == "__main__":
    main()
