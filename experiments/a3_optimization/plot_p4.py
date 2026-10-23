"""Problem 4: the Hessian of the d8 model, from the fetched results (p4_inside_the_hessian fetch).

(a) plots/p4a_curvature.png  family / global lambda_max, tr(H^2)/d, and lambda_min vs. tokens
    plots/p4a_density.png    eigenvalue density (Lanczos + Gaussian smoothing) at 0, 10M, 100M, 614M
    plots/p4a_blocks.png     share of ||H||_F^2 inside family / layer blocks vs. a random partition,
                             and per-layer curvature vs. depth
(b) plots/p4b_sharpness.png  function-preserving rescalings, and the training changes
(c) plots/p4c_subspace.png   SGDM continued with the full / top-k / top-k-removed update
"""

from __future__ import annotations

import math

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization import hessian as hs
from experiments.a3_optimization import p4_inside_the_hessian as p
from experiments.a3_optimization.hessian_checkpoints import STEPS
from experiments.a3_optimization.hessian_jobs import load_result
from experiments.a3_optimization.plot_p1a import savefig
from experiments.a3_optimization.plot_p3ab import plain_log_ticks, plain_number

FAMILIES = ("embedding", "attention", "qk_norm", "mlp", "norm", "head")
FAMILY_COLORS = dict(zip(FAMILIES, ("C1", "C0", "C9", "C2", "C4", "C3")))
INIT_X = 1e5  # where step 0 sits on the log token axis


def checkpoints():
    return {step: load_result(f"a-adamw-{step}") for step in STEPS}


def token_axis(ax, results):
    ax.set_xscale("log")
    ax.set_xlim(INIT_X * 0.7, max(r["tokens"] for r in results.values()) * 1.4)
    ticks = [INIT_X, 1e6, 1e7, 1e8, 6.144e8]
    ax.set_xticks(ticks, ["init", "1M", "10M", "100M", "614M"])
    ax.minorticks_off()
    ax.set_xlabel("training tokens")


def x_of(r):
    return max(r["tokens"], INIT_X)


def energy(result, part):
    """(block names, mean [B, B] estimate of ||H_cb||_F^2, block sizes)."""
    e = result["energy"][part]
    blocks = hs.partition(result["names"], result["sizes"], hs.family_of if part == "family" else hs.layer_of)
    sizes = [hs.block_size(blocks[b]) for b in e["blocks"]]
    return e["blocks"], np.mean(np.array(e["per_probe"]), axis=0), np.array(sizes, dtype=float)


# ---------------------------------------------------------------- (a)
def part_a(results):
    xs = [x_of(r) for r in results.values()]
    fig, (fx, tx, nx) = plt.subplots(1, 3, figsize=(18, 4.8))
    fx.plot(xs, [r["lambda_max"] for r in results.values()], "k-o", ms=4, lw=2, label="λ_max (full H)")
    for fam in FAMILIES:
        fx.plot(xs, [r["families"][fam]["lambda_max"] for r in results.values()], "o-", ms=3,
                color=FAMILY_COLORS[fam], label=fam)
    fx.set(yscale="log", ylabel="top eigenvalue of the block", title="(a) sharpest family over training")
    fx.legend(fontsize=8, ncol=2)

    trace_sq = [energy(r, "family")[1].sum() / r["size"] for r in results.values()]
    tx.plot(xs, trace_sq, "o-", color="C0", label="tr(H²) / d")
    tx.plot(xs, [r["lambda_max"] ** 2 / r["size"] for r in results.values()], "o--", color="0.5", ms=3,
            label="λ_max² / d (top eigenvalue alone)")
    tx.set(yscale="log", ylabel="mean squared eigenvalue", title="(a) tr(H²)/d vs. λ_max²/d")
    tx.legend(fontsize=8)

    nx.plot(xs, [-r["lambda_min"] for r in results.values()], "o-", color="C3", label="−λ_min (most negative)")
    nx.plot(xs, [r["lambda_max"] for r in results.values()], "o-", color="k", label="λ_max")
    nx2 = nx.twinx()
    nx2.plot(xs, [-r["lambda_min"] / r["lambda_max"] for r in results.values()], "s:", color="C3", ms=3)
    nx2.set_ylabel("|λ_min| / λ_max (dotted)", color="C3")
    nx2.set_ylim(0, 1.05)
    nx.set(yscale="log", ylabel="eigenvalue magnitude", title="(a) negative curvature")
    nx.legend(fontsize=8, loc="lower left")
    for a in (fx, tx, nx):
        token_axis(a, results)
        a.grid(alpha=0.3, which="both")
        plain_log_ticks(a)
    fig.suptitle("P4(a): Hessian of the default AdamW d8 run over training (64-row held-out probe, FP32)")
    fig.tight_layout()
    savefig(fig, "p4a_curvature.png")

    # Densities: Ritz values through a symmetric log so outliers and the bulk fit on one axis.
    tau = 1e-3
    t = lambda x: np.sign(x) * np.log10(1 + np.abs(x) / tau)
    dens_steps = [s for s in p.DENSITY_AT if results[s].get("density_runs")]
    fig, axes = plt.subplots(1, len(dens_steps), figsize=(4.8 * len(dens_steps), 4.2), sharey=True)
    for ax, step in zip(axes, dens_steps):
        r = results[step]
        runs = [dict(ritz=t(np.array(run["ritz"])), weights=run["weights"]) for run in r["density_runs"]]
        grid = np.linspace(t(r["lambda_min"]) - 0.5, t(r["lambda_max"]) + 0.5, 2000)
        d = hs.density(runs, grid, width=0.04)
        ax.plot(grid, np.maximum(d, 1e-8), color="C0")
        ax.fill_between(grid, 1e-8, np.maximum(d, 1e-8), alpha=0.2)
        for run in runs:
            ax.plot(run["ritz"], np.full(len(run["ritz"]), 2e-8), "|", color="k", ms=6, alpha=0.4)
        ticks = [v for v in (-1000, -10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10, 1000)
                 if r["lambda_min"] * 1.5 <= v <= r["lambda_max"] * 1.5]
        ax.set_xticks([t(v) for v in ticks], [plain_number(v) for v in ticks])
        ax.axvline(t(r["lambda_max"]), color="k", ls="--", lw=0.8)
        ax.axvline(t(r["lambda_min"]), color="C3", ls="--", lw=0.8)
        ax.set(yscale="log", ylim=(1e-8, 50), xlabel=f"eigenvalue (symlog, linear within ±{tau:g})",
               title=f"{r['tokens'] / 1e6:.0f}M tokens (step {step}): "
                     f"λ ∈ [{plain_number(round(r['lambda_min'], 3))}, {plain_number(round(r['lambda_max'], 3))}]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density (per unit of transformed axis)")
    fig.suptitle("P4(a): Hessian eigenvalue density (4 Lanczos runs × 96 steps; ticks = Ritz values; "
                 "dashed = λ_min, λ_max)")
    fig.tight_layout()
    savefig(fig, "p4a_density.png")

    # Block structure.
    fig, (bx, dx) = plt.subplots(1, 2, figsize=(14, 4.8))
    for part, color in (("family", "C0"), ("layer", "C2")):
        inside, baseline = [], []
        for r in results.values():
            keys, E, sizes = energy(r, part)
            inside.append(np.trace(E) / E.sum())
            baseline.append(np.sum((sizes / sizes.sum()) ** 2))
        bx.plot(xs, inside, "o-", color=color, label=f"inside {part} blocks ({len(keys)} blocks)")
        bx.plot(xs, baseline, "--", color=color, lw=1, label=f"random partition, same sizes ({part})")
    bx.set(ylim=(0, 1.02), ylabel="share of ||H||_F² inside diagonal blocks",
           title="(a) how block-diagonal is H?")
    token_axis(bx, results)
    bx.legend(fontsize=8)
    show = [s for s in (15, 153, 1526, 9375) if s in results]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
    for step, color in zip(show, cmap):
        keys, E, sizes = energy(results[step], "layer")
        layers = [k for k in keys if k.startswith("L")]
        idx = [keys.index(k) for k in layers]
        dx.plot([int(k[1:]) for k in layers], [E[i, i] / sizes[i] for i in idx], "o-", color=color,
                label=f"{results[step]['tokens'] / 1e6:.0f}M tokens")
    dx.set(yscale="log", xlabel="layer", ylabel="tr(H_LL²) / n_L  (diagonal block, per parameter)",
           title="(a) does curvature grow with depth?")
    dx.legend(fontsize=8)
    plain_log_ticks(dx)
    for a in (bx, dx):
        a.grid(alpha=0.3, which="both")
    fig.suptitle("P4(a): block structure of the Hessian (Hutchinson block energies, 4 probes per block)")
    fig.tight_layout()
    savefig(fig, "p4a_blocks.png")

    print("(a) step  tokens   lambda_max  sharpest family      lambda_min  inside(family) inside(layer)")
    for step, r in results.items():
        fam = max(FAMILIES, key=lambda f: r["families"][f]["lambda_max"])
        kf, Ef, _ = energy(r, "family")
        kl, El, _ = energy(r, "layer")
        print(f"{step:7d} {r['tokens'] / 1e6:7.1f}M {r['lambda_max']:10.3f}  {fam:10s} "
              f"{r['families'][fam]['lambda_max']:9.3f} {r['lambda_min']:10.3f} {np.trace(Ef) / Ef.sum():12.3f} "
              f"{np.trace(El) / El.sum():12.3f}")


# ---------------------------------------------------------------- (b)
def training_val_losses():
    import wandb

    from train import checked_train_config, training_run_name
    from utils import WANDB_ENTITY, WANDB_PROJECT

    names = {training_run_name(checked_train_config(c)): c for c in p.RUNS}
    out = {}
    for run in wandb.Api(timeout=60).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [p.EXPERIMENT_KEY]}}):
        if run.name in names and run.state == "finished":
            out[run.name] = run.summary["val_loss"]
    return out


def part_b():
    rescales = {label: load_result(f"b-rescale-{label}") for label in p.RESCALES}
    fig, (rx, tx, vx) = plt.subplots(1, 3, figsize=(19, 5))
    labels = list(rescales)
    width = 0.27
    for i, (key, name, color) in enumerate((("all", "λ_max (full H)", "k"), ("head", "head block", "C3"),
                                            ("attention", "attention block", "C0"))):
        vals = [r["lambda_max"] if key == "all" else r["families"][key]["lambda_max"] for r in rescales.values()]
        rx.bar(np.arange(len(labels)) + (i - 1) * width, vals, width, color=color, label=name)
    rx.set_xticks(range(len(labels)),
                  [f"{lab}\nprobe loss {r['probe_loss']:.5f}" for lab, r in rescales.items()], fontsize=8)
    rx.set(yscale="log", ylim=(0.5, 40), ylabel="top eigenvalue", title="(b) rescaling the final checkpoint by hand")
    rx.legend(fontsize=8)

    tensors = p.AFFECTED
    short = {p.HEAD: "lm_head", p.FINAL_NORM: "final norm gain", p.QK[0]: "L0 q_proj", p.QK[-2]: "L7 q_proj"}
    width = 0.8 / len(rescales)
    for i, (lab, r) in enumerate(rescales.items()):
        tx.bar(np.arange(len(tensors)) + (i - (len(rescales) - 1) / 2) * width,
               [r["tensors"][t]["lambda_max"] for t in tensors], width, label=lab)
    tx.set_xticks(range(len(tensors)), [short[t] for t in tensors])
    tx.set(yscale="log", ylabel="top eigenvalue of the tensor's own block",
           title="(b) affected blocks before (identity) and after")
    tx.legend(fontsize=8)

    reference = load_result("b-rescale-reference")
    default_val = 2.9271  # p2_river_valley linear LR 3e-3: the default recipe, same seed as the shipped run
    vals = training_val_losses()
    vx.axhspan(2.925, 2.931, color="C2", alpha=0.15, label="target val loss 2.925–2.931")
    vx.axvspan(reference["lambda_max"] / 2, reference["lambda_max"] * 2, color="0.85", alpha=0.6,
               label="within 2× of default λ_max")
    vx.plot(reference["lambda_max"], default_val, "k*", ms=14, label="default recipe")
    for job, config in zip(p.FINAL_JOBS, p.RUNS):
        r = load_result(job["name"])
        name = job["name"].removeprefix("b-final-")
        label = name.split("-p4b")[0].split("tok614M-")[-1] or ""
        label = ("head WD " + config.run_name_suffix.removeprefix("p4b-headwd")) if "headwd" in name else "global WD 0"
        v = vals.get(name)
        if v is not None:
            vx.plot(r["lambda_max"], v, "o", ms=8)
            offset = (-12, -26) if r["lambda_max"] < reference["lambda_max"] else (6, 4)
            vx.annotate(f"{label}\nλ={r['lambda_max']:.1f}, val={v:.4f}", (r["lambda_max"], v), xytext=offset,
                        textcoords="offset points", fontsize=8, ha="right" if offset[0] < 0 else "left")
    vx.set(xscale="log", xlabel="final λ_max", ylabel="final validation loss",
           title="(b) moving λ_max by training: weight decay on lm_head")
    vx.legend(fontsize=8, loc="center")
    vx.set_xlim(1.2, 80)
    vx.set_ylim(2.922, 2.956)
    for a in (rx, tx, vx):
        a.grid(alpha=0.3, which="both")
    plain_log_ticks(rx)
    plain_log_ticks(tx)
    plain_log_ticks(vx, "x", subs=(1.0, 2.0, 5.0))
    fig.suptitle("P4(b): λ_max moves with function-preserving reparametrizations (W_head/c with gain·c; "
                 "q,k projections before QK-norm)")
    fig.tight_layout()
    savefig(fig, "p4b_sharpness.png")


# ---------------------------------------------------------------- (c)
def arm(name):
    records = load_result(name)
    steps = [x for x in records if x.get("kind") == "step"]
    refresh = [x for x in records if x.get("kind") == "refresh"]
    start = next(x for x in records if x.get("kind") == "start")
    return start, steps, refresh


def part_c():
    late = {"c-full": ("full", "C0", "-"), "c-removed": ("top-10 removed", "C2", "-"),
            "c-top": ("top-10 only", "C3", "-"), "c-top-k1": ("top-1 only", "C1", "-"),
            "c-full-lr0.25": ("full, ¼ LR", "C0", "--"), "c-top-lr0.25": ("top-10 only, ¼ LR", "C3", "--")}
    early = {"c-full-step305": ("full", "C0", "-"), "c-removed-step305": ("top-10 removed", "C2", "-"),
             "c-top-step305": ("top-10 only", "C3", "-")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    def val_panel(ax, arms, title):
        for name, (label, color, ls) in arms.items():
            start, steps, _ = arm(name)
            pts = [(start["step"], start["val_loss"])] + [(x["step"], x["val_loss"]) for x in steps if "val_loss" in x]
            final = pts[-1][1]
            ax.plot(*zip(*pts), color=color, ls=ls, marker=".", label=f"{label} (final {final:.3f})")
        ax.set(yscale="log", xlabel="SGDM step", ylabel="validation loss", title=title)
        ax.legend(fontsize=8)
        plain_log_ticks(ax, subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))

    val_panel(axes[0, 0], late, "(c) from 100M tokens (step 1526), 300 steps")
    val_panel(axes[0, 1], early, "(c) from 20M tokens (step 305), 300 steps")

    ax = axes[1, 0]
    for name, (label, color, ls) in {**late}.items():
        _, steps, _ = arm(name)
        s = np.array([x["step"] for x in steps])
        f = np.array([x["top_fraction"] for x in steps])
        if len(f) >= 10:
            ax.plot(s[9:], np.convolve(f, np.ones(10) / 10, mode="valid"), color=color, ls=ls, label=label)
    ax.set(ylim=(0, 1), xlabel="SGDM step", ylabel="‖P_top g‖² / ‖g‖²  (10-step mean)",
           title="(c) fraction of the minibatch gradient in the top-k subspace")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, (label, color, ls) in {k: late[k] for k in ("c-full", "c-removed", "c-top", "c-top-lr0.25")}.items():
        _, _, refresh = arm(name)
        s = [x["step"] for x in refresh]
        ax.plot(s, [x["lambda_max"] for x in refresh], color=color, ls=ls, marker="o", ms=3, label=f"λ_max, {label}")
        ax.plot(s, [x["heavy_ball_threshold"] for x in refresh], color=color, ls=":", lw=1)
    ax.plot([], [], "k:", label="heavy-ball threshold 2(1+β)/η_t")
    ax.set(yscale="log", xlabel="SGDM step", ylabel="λ_max at each basis refresh",
           title="(c) sharpness vs. the stability threshold (from step 1526)")
    ax.legend(fontsize=8)
    plain_log_ticks(ax)
    for a in axes.flat:
        a.grid(alpha=0.3, which="both")
    fig.suptitle("P4(c): SGD with momentum continued with the full update, only its top-k Hessian component, "
                 "or with that component removed (basis refreshed every 25 steps)")
    fig.tight_layout()
    savefig(fig, "p4c_subspace.png")


def part_c_refresh():
    """Refresh interval: does a fresher basis rescue the top-10 arm, and does 'removed' still match 'full'?"""
    arms = {"c-full": ("full (no basis)", "C0", "-", None),
            "c-top-refresh5": ("top-10 only, refresh 5", "C3", "-", 5),
            "c-top": ("top-10 only, refresh 25", "C3", "--", 25),
            "c-top-refresh100": ("top-10 only, refresh 100", "C3", ":", 100),
            "c-removed-refresh5": ("top-10 removed, refresh 5", "C2", "-", 5),
            "c-removed": ("top-10 removed, refresh 25", "C2", "--", 25)}
    loaded = {}
    for name in arms:
        try:
            loaded[name] = arm(name)
        except (FileNotFoundError, StopIteration):
            print(f"(c) refresh: {name} not fetched yet")
    if "c-full" not in loaded or len(loaded) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    print(f"\n(c) {'arm':>28} {'final val':>9} {'mean top fraction':>17} {'mean basis overlap':>18}")
    for name, (start, steps, refresh) in loaded.items():
        label, color, ls, every = arms[name]
        pts = [(start["step"], start["val_loss"])] + [(x["step"], x["val_loss"]) for x in steps if "val_loss" in x]
        axes[0].plot(*zip(*pts), color=color, ls=ls, marker=".", label=f"{label} (final {pts[-1][1]:.3f})")
        s = np.array([x["step"] for x in steps])
        f = np.array([x["top_fraction"] for x in steps])
        if len(f) >= 10:
            axes[1].plot(s[9:], np.convolve(f, np.ones(10) / 10, mode="valid"), color=color, ls=ls, label=label)
        overlaps = [x["basis_overlap"] for x in refresh if x.get("basis_overlap") is not None]
        if every is not None and overlaps:
            axes[2].plot([x["step"] for x in refresh if x.get("basis_overlap") is not None], overlaps,
                         color=color, ls=ls, marker="o", ms=3, label=label)
        print(f"{label:>28} {pts[-1][1]:9.4f} {f.mean():17.3f} "
              f"{(np.mean(overlaps) if overlaps else float('nan')):18.3f}")
    axes[0].set(yscale="log", xlabel="SGDM step", ylabel="validation loss",
                title="validation loss (from 100M tokens, step 1526)")
    plain_log_ticks(axes[0], subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
    axes[1].set(ylim=(0, 1), xlabel="SGDM step", ylabel="‖P_top g‖² / ‖g‖²  (10-step mean)",
                title="fraction of the gradient in the current top-10 basis")
    axes[2].set(ylim=(0, 1), xlabel="SGDM step", ylabel="overlap with the previous basis",
                title="how much the top-10 subspace moved since the last refresh")
    for a in axes:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("P4(c): refresh interval of the top-10 basis (eigenvectors recomputed every 5 / 25 / 100 steps)")
    fig.tight_layout()
    savefig(fig, "p4c_refresh.png")


def main():
    part_a(checkpoints())
    part_b()
    part_c()
    part_c_refresh()


if __name__ == "__main__":
    main()
