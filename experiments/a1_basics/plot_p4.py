"""Problem 4 analysis: how tiny perturbations amplify in deterministic training.

Every P4 run is compared with Problem 3's deterministic reference-1 (same config, no
perturbation; reference-2 is identical to it bit for bit). Divergence measures:
  parameter distance  ||theta - theta_ref|| / ||theta_ref||, from the 64 logged random
                      projections (param_fingerprint/*), at every eval step;
  |delta train loss|  per step, as a 51-step moving average of |L - L_ref|;
  |delta val loss|    at every eval step.
Writes plots/p4_*.png and prints a table."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p4_amplification import P4_KEY
from experiments.a1_basics.perturbation import FINGERPRINT_DIM
from experiments.a1_basics.results import history, load_runs, savefig

REFERENCES = (
    "model-d8-lr0.003-tok614M-deterministic-deterministic-reference-1",
    "model-d8-lr0.003-tok614M-deterministic-deterministic-reference-2",
)
FINGERPRINT_KEYS = [f"logging/param_fingerprint/{i:02d}" for i in range(FINGERPRINT_DIM)] + ["logging/param_fingerprint/norm"]
FLOOR = 1e-12


def describe(run):
    if run.name == REFERENCES[1]:
        return dict(kind="reference-2", scale=0.0, after=0)
    if run.config.get("perturb_one_token"):
        return dict(kind="one token", scale=None, after=1)
    suffix = run.name.split("-noise")[-1]  # e.g. "1e-06-after2000"
    scale, after = suffix.split("-after")
    return dict(kind="param noise", scale=float(scale), after=int(after))


def divergence(run, ref):
    fp, fp_ref = history(run, ["val_loss", *FINGERPRINT_KEYS]), history(ref, ["val_loss", *FINGERPRINT_KEYS])
    n = min(len(fp["optimizer_step"]), len(fp_ref["optimizer_step"]))
    proj = np.stack([fp[key][:n] for key in FINGERPRINT_KEYS[:-1]])
    proj_ref = np.stack([fp_ref[key][:n] for key in FINGERPRINT_KEYS[:-1]])
    param = np.sqrt(np.mean((proj - proj_ref) ** 2, axis=0)) / fp_ref["logging/param_fingerprint/norm"][:n]
    val = np.abs(fp["val_loss"][:n] - fp_ref["val_loss"][:n])
    tr, tr_ref = history(run, ["train_loss"]), history(ref, ["train_loss"])
    m = min(len(tr["train_loss"]), len(tr_ref["train_loss"]))
    train = np.convolve(np.abs(tr["train_loss"][:m] - tr_ref["train_loss"][:m]), np.ones(51) / 51, mode="same")
    return dict(eval_steps=fp["optimizer_step"][:n], param=param, val=val,
                train_steps=tr["optimizer_step"][:m], train=train)


def growth_rate(steps, values, start, span=1500):
    """Slope of log10(divergence) per 1000 steps over [start + 100, start + span]."""
    mask = (steps >= start + 100) & (steps <= start + span) & (values > FLOOR)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(steps[mask], np.log10(values[mask]), 1)[0] * 1000)


def main():
    runs = [run for run in load_runs([P4_KEY], names=list(REFERENCES)) if run.state == "finished"]
    by_name = {run.name: run for run in runs}
    ref = by_name[REFERENCES[0]]
    targets = [run for run in runs if run.name != REFERENCES[0]]
    if len(targets) == 1:
        print("Only reference-2 available yet; P4 runs have not finished.")

    results = []
    for run in targets:
        info = describe(run)
        div = divergence(run, ref)
        realized = None
        if info["kind"] == "param noise":
            realized_h = history(run, ["logging/perturbation/realized_relative_norm"])
            realized = float(realized_h["logging/perturbation/realized_relative_norm"][0]) if len(realized_h["optimizer_step"]) else None
        results.append(dict(run=run, **info, **div, realized=realized,
                            rate=growth_rate(div["eval_steps"], div["param"], info["after"])))

    print(f"{'perturbation':<26} {'realized':>9} {'param dist @+100':>16} {'@end':>9} {'growth/1k steps':>15} {'|dval| end':>11} {'|dval| max':>11}")
    for r in sorted(results, key=lambda r: (r["kind"], r["after"], r["scale"] or 0)):
        name = r["kind"] if r["kind"] != "param noise" else f"noise {r['scale']:.0e} after {r['after']}"
        idx = int(np.searchsorted(r["eval_steps"], r["after"] + 100))
        idx = min(idx, len(r["param"]) - 1)
        print(f"{name:<26} {r['realized'] if r['realized'] is not None else float('nan'):9.2e} "
              f"{r['param'][idx]:16.2e} {r['param'][-1]:9.2e} {r['rate']:15.2f} {r['val'][-1]:11.2e} {r['val'].max():11.2e}")

    noise = [r for r in results if r["kind"] == "param noise"]
    others = [r for r in results if r["kind"] != "param noise"]
    scales = sorted({r["scale"] for r in noise})
    afters = sorted({r["after"] for r in noise})
    colors = dict(zip(scales, mpl.colormaps["viridis"](np.linspace(0, 0.9, max(len(scales), 1)))))
    styles = dict(zip(afters, ["-", "--", "-.", ":", (0, (1, 3))]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    for r in noise:
        kw = dict(color=colors[r["scale"]], ls=styles[r["after"]], lw=1.5, label=f"{r['scale']:.0e} after {r['after']}")
        axes[0].plot(r["eval_steps"], np.maximum(r["param"], FLOOR), **kw)
        axes[1].plot(r["train_steps"], np.maximum(r["train"], FLOOR), **kw)
        axes[2].plot(r["eval_steps"], np.maximum(r["val"], FLOOR), **kw)
    for r in others:
        kw = dict(color="#d62728" if r["kind"] == "one token" else "black", lw=2, label=r["kind"])
        axes[0].plot(r["eval_steps"], np.maximum(r["param"], FLOOR), **kw)
        axes[1].plot(r["train_steps"], np.maximum(r["train"], FLOOR), **kw)
        axes[2].plot(r["eval_steps"], np.maximum(r["val"], FLOOR), **kw)
    for ax, title in zip(axes, ("parameter distance ||dtheta|| / ||theta||", "|delta train loss| (51-step mean)",
                                "|delta val loss|")):
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.3)
        for after in afters:
            ax.axvline(after, color="#cccccc", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.suptitle(f"P4: divergence from the deterministic reference (values at {FLOOR:g} are exactly identical)")
    fig.tight_layout()
    savefig(fig, "p4_divergence.png")
    plt.close(fig)

    if noise:
        fig, (ax_final, ax_remaining) = plt.subplots(1, 2, figsize=(13, 4.8))
        for after in afters:
            subset = sorted((r for r in noise if r["after"] == after), key=lambda r: r["scale"])
            ax_final.plot([r["scale"] for r in subset], [max(r["val"][-1], FLOOR) for r in subset], "o-", label=f"after {after}")
        ax_final.set_xscale("log")
        ax_final.set_yscale("log")
        ax_final.set_xlabel("perturbation relative size")
        ax_final.set_ylabel("|final val loss - reference|")
        ax_final.set_title("Final effect vs perturbation size")
        ax_final.grid(True, ls=":", alpha=0.3)
        ax_final.legend(frameon=False)
        for scale in scales:
            subset = sorted((r for r in noise if r["scale"] == scale), key=lambda r: r["after"])
            ax_remaining.plot([ref.config["total_steps"] - r["after"] for r in subset],
                              [r["param"][-1] for r in subset], "o-", color=colors[scale], label=f"size {scale:.0e}")
        ax_remaining.set_xscale("log")
        ax_remaining.set_yscale("log")
        ax_remaining.set_xlabel("steps remaining after the perturbation")
        ax_remaining.set_ylabel("final parameter distance")
        ax_remaining.set_title("Final divergence vs time left to amplify")
        ax_remaining.grid(True, ls=":", alpha=0.3)
        ax_remaining.legend(frameon=False)
        fig.tight_layout()
        savefig(fig, "p4_size_and_time.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
