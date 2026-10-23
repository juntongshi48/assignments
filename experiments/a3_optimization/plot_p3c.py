"""Problem 3(c): what decides whether AdamW trains at the edge of stability.

Reads PART_C_RUNS (and the part (b) base run) from W&B through plot_p3ab.load (cached under cache/p3/).
plots/p3c_knobs.png
  eta_t * lambda_pre / threshold vs. training tokens, one panel per knob (batch size, peak LR, beta1,
  beta2, weight decay, warmup), with the base recipe in black; threshold = 2 (1 + beta1) / (1 - beta1).
plots/p3c_summary.png
  Median ratio over 10-50M, 50-100M and 100-200M tokens for every run, and its final training loss.
plots/p3c_full_batch_adamw.png
  Full-batch AdamW on 32 sequences at four LRs: eta * lambda_pre against 38, and the training loss
  against the memorization floor ln(32) / 1024.
plots/p3c_beta2_raw_pre.png
  beta2 runs: raw lambda_max and lambda_pre, to separate the Hessian from the preconditioner.
"""

from __future__ import annotations

import math

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p3_edge_of_stability import MINIBATCH_RUNS, PART_C_LOW_LR_RUNS, PART_C_RUNS
from experiments.a3_optimization.plot_p1a import savefig
from experiments.a3_optimization.plot_p3ab import column, load, plain_log_ticks, plain_number
from experiments.a3_optimization.sharpness import window_median
from train import TrainConfig, checked_train_config

WINDOWS = ((10e6, 50e6), (50e6, 100e6), (100e6, 200e6))
BASE_LABEL = "base (bs 64, LR 3e-3, β1 0.9, β2 0.95, WD 0.1, warmup 3.1%)"


def threshold(config):
    return 2 * (1 + config.beta1) / (1 - config.beta1)


def warmup_updates(config):
    if config.warmup_steps is not None:
        return config.warmup_steps
    return int(config.num_train_sequences * config.num_epochs / config.batch_size * config.warmup_percent)


def tokens_per_update(config):
    return config.batch_size * config.train_dataset.context_length


def is_full_batch(config):
    return config.batch_size == config.num_train_sequences


def knob(config):
    """(knob name, value) for a minibatch run: the one setting that differs from the base recipe."""
    base = checked_train_config(MINIBATCH_RUNS[0])
    for name, field, fmt in (("batch size", "batch_size", "{:d}"), ("peak LR", "learning_rate", "{:g}"),
                             ("β1", "beta1", "{:g}"), ("β2", "beta2", "{:g}"),
                             ("weight decay", "weight_decay", "{:g}"), ("warmup", "warmup_percent", "{:.1%}")):
        if getattr(config, field) != getattr(base, field):
            return name, fmt.format(getattr(config, field))
    return "base", "rawpre replicate" if config.run_name_suffix else "base"


def ratio_curve(config, rows):
    """(tokens in M, eta * lambda_pre / threshold, True where the measurement is after warmup)."""
    s, prod = column(rows, "eta_lambda_pre")
    return s * tokens_per_update(config) / 1e6, prod / threshold(config), s > warmup_updates(config)


def draw_ratio(ax, config, rows, color, label, lw=1.0):
    """Post-warmup points solid; warmup points dashed and hollow, joined to the first point after warmup."""
    x, y, post = ratio_curve(config, rows)
    ax.plot(x[post], y[post], "o-", ms=3, lw=lw, color=color, label=label)
    if (~post).any():
        warm = ~post
        warm[np.argmax(post)] = True
        ax.plot(x[warm], y[warm], "o--", ms=3, lw=1, color=color, alpha=0.6, mfc="white", zorder=3)


def load_all():
    base = checked_train_config(MINIBATCH_RUNS[0])
    runs = [(base, load(MINIBATCH_RUNS[0]))]
    runs += [(checked_train_config(c), load(c)) for c in PART_C_RUNS + PART_C_LOW_LR_RUNS]
    return runs


def knobs_figure(runs):
    base_config, base_rows = runs[0]
    groups = {}
    for config, rows in runs[1:]:
        if not is_full_batch(config):
            name, value = knob(config)
            if name != "base":
                groups.setdefault(name, []).append((value, config, rows))
    order = ("batch size", "peak LR", "β1", "β2", "weight decay", "warmup")
    base_value = {"batch size": "64", "peak LR": "0.003", "β1": "0.9", "β2": "0.95", "weight decay": "0.1",
                  "warmup": "3.1%"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharey=True)
    for ax, name in zip(axes.flat, order):
        draw_ratio(ax, base_config, base_rows, "k", f"{name} {base_value[name]} (base)", lw=1.2)
        members = sorted(groups[name], key=lambda m: float(m[0].rstrip("%")))
        colors = ("C0", "C1", "C3") if len(members) == 3 else ("C0", "C3")
        for (value, config, rows), color in zip(members, colors):
            extra = f" (threshold {threshold(config):g})" if name == "β1" else ""
            draw_ratio(ax, config, rows, color, f"{name} {value}{extra}")
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set(xscale="log", yscale="log", title=name, xlabel="training tokens (M)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")
    for ax in axes[:, 0]:
        ax.set_ylabel("η_t λ_pre / [2(1+β1)/(1−β1)]")
    axes[0, 0].set_ylim(0.15, 30)
    for ax in axes.flat:
        plain_log_ticks(ax, "xy")
    fig.suptitle("P3(c): preconditioned sharpness relative to the AdamW stability threshold, one knob at a time "
                 "(1 = at the edge)\ndashed line with hollow points = during each run's warmup; the first measurement (update 16, ratio 17-48) is above the axis")
    fig.tight_layout()
    savefig(fig, "p3c_knobs.png")


def summary_figure(runs):
    rows_out = []
    for config, rows in runs:
        if is_full_batch(config):
            continue
        name, value = knob(config)
        meds = [window_median(rows, config, lo, hi) for lo, hi in WINDOWS]
        _, loss = zip(*rows.losses)
        final = float(np.mean(loss[-max(1, len(loss) // 20):]))
        rows_out.append((name, value, config, [m / threshold(config) for m in meds], final))
    order = ("base", "batch size", "peak LR", "β1", "β2", "weight decay", "warmup")
    rows_out.sort(key=lambda r: (order.index(r[0]), r[0] != "base" and float(r[1].rstrip("%"))))
    labels = [f"{n} = {v}" if n != "base" else ("base" if v == "base" else "base (replicate)")
              for n, v, *_ in rows_out]
    labels = [label + (f" ({int(r[2].num_train_sequences * r[2].num_epochs / r[2].batch_size):,} updates)"
                       if r[0] == "batch size" else "") for label, r in zip(labels, rows_out)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 8), sharey=True, gridspec_kw=dict(width_ratios=(3, 1.3)))
    y = np.arange(len(rows_out))[::-1]
    for k, ((lo, hi), marker, color) in enumerate(zip(WINDOWS, "^so", ("0.7", "C0", "C3"))):
        ax.scatter([r[3][k] for r in rows_out], y, marker=marker, color=color, s=40, zorder=3,
                   label=f"{lo / 1e6:.0f}–{hi / 1e6:.0f}M tokens" + (" (value shown)" if k == 2 else ""))
    for yi, r in zip(y, rows_out):
        ax.plot(r[3], [yi] * 3, color="0.6", lw=1, zorder=2)
        ax.annotate(f"{r[3][2]:.2f}", (max(r[3]), yi), xytext=(9, -3), textcoords="offset points", fontsize=8,
                    color="C3")
    ax.axvline(1, color="k", ls="--", lw=1, label="at the edge")
    ax.set(xscale="log", xlabel="median η_t λ_pre / [2(1+β1)/(1−β1)]", yticks=y, yticklabels=labels,
           title="Where each run sits relative to its threshold")
    ax.set_xlim(0.15, 15)
    plain_log_ticks(ax, "x")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both", axis="x")
    for yi, r in zip(y, rows_out):
        bx.barh(yi, r[4], color="C0" if r[0] != "base" else "k", height=0.6)
        bx.annotate(f"{r[4]:.3f}", (r[4], yi), xytext=(3, -3), textcoords="offset points", fontsize=8)
    bx.set(xlim=(3.0, 4.25), xlabel="final training loss (mean of last 5%)", title="Training loss at 196M tokens")
    bx.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain_number))
    bx.grid(alpha=0.3, axis="x")
    for boundary in np.cumsum([sum(r[0] == g for r in rows_out) for g in order])[:-1]:
        for a in (ax, bx):
            a.axhline(len(rows_out) - boundary - 0.5, color="0.8", lw=0.8)
    fig.suptitle("P3(c): median preconditioned sharpness / threshold by training window (one knob changed per run)")
    fig.tight_layout()
    savefig(fig, "p3c_summary.png")
    return rows_out


def full_batch_figure(runs):
    """eta * lambda_pre and loss for full-batch AdamW. The loss floor is ln(32) / 1024 (3e-4 and 1e-3 both end at
    0.00339): the sequences evidently share their first token, so only the next one stays unpredictable once
    the 32 are memorized. After a run gets within 1.5x of the floor its eta * lambda_pre points are drawn
    faded: that is AdamW at the minimum, not while training."""
    fb = sorted([(c, r) for c, r in runs if is_full_batch(c)], key=lambda m: m[0].learning_rate)
    floor = math.log(fb[0][0].num_train_sequences) / fb[0][0].train_dataset.context_length
    colors = ("C2", "C0", "C1", "C3")[-len(fb):]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw=dict(height_ratios=(3, 2)))
    for (config, rows), color in zip(fb, colors):
        u, loss = map(np.array, zip(*rows.losses))
        below = np.nonzero(loss < 1.5 * floor)[0]
        memorized = int(u[below[0]]) if len(below) else None
        s_, prod = column(rows, "eta_lambda_pre")
        live = s_ < memorized if memorized is not None else np.ones_like(s_, bool)
        label = f"AdamW η = {config.learning_rate:g}" + (f" (at floor from update {memorized})"
                                                         if memorized is not None else " (never reaches floor)")
        axes[0].plot(s_[live], prod[live], "o-", ms=3.5, lw=1.2, color=color, label=label)
        if (~live).any():
            axes[0].plot(s_[~live], prod[~live], "o-", ms=2.5, lw=0.8, color=color, alpha=0.3)
        axes[1].plot(u, loss, lw=0.9, color=color)
        if memorized is not None:
            for a in axes:
                a.axvline(memorized, color=color, ls=":", lw=1)
    axes[0].axhline(threshold(fb[0][0]), color="k", ls="--", lw=1, label="2(1+β1)/(1−β1) = 38")
    axes[0].set(yscale="log", ylabel="η λ_pre (full-batch Hessian)",
                title="Full-batch AdamW on the first 32 sequences (constant LR, no warmup, WD 0); "
                      "faded = after memorization")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].axhline(floor, color="k", ls="--", lw=1, label=f"memorization floor ln(32)/1024 = {floor:.5f}")
    axes[1].set(yscale="log", xlabel="update", ylabel="training loss (full batch)")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3, which="both")
        plain_log_ticks(a)
    fig.tight_layout()
    savefig(fig, "p3c_full_batch_adamw.png")


def beta2_figure(runs):
    members = [(c, r) for c, r in runs if not is_full_batch(c) and "lambda_max" in {k for row in r for k in row}]
    members.sort(key=lambda m: m[0].beta2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, (config, rows) in enumerate(members):
        color = cmap(i / (len(members) - 1))
        label = f"β2 = {config.beta2:g} (timescale ≈ {1 / (1 - config.beta2):.0f} updates)" + \
            (" base" if config.beta2 == TrainConfig.beta2 else "")
        for ax, key in zip(axes[:2], ("lambda_max", "lambda_max_pre")):
            s, v = column(rows, key)
            keep = s > warmup_updates(config)
            ax.plot(s[keep] * tokens_per_update(config) / 1e6, v[keep], "o-", ms=3, lw=1, color=color, label=label)
        s, raw = column(rows, "lambda_max")
        s2, pre = column(rows, "lambda_max_pre")
        common = sorted(set(s) & set(s2))
        raw_at, pre_at = dict(zip(s, raw)), dict(zip(s2, pre))
        keep = [t for t in common if t > warmup_updates(config)]
        axes[2].plot(np.array(keep) * tokens_per_update(config) / 1e6, [pre_at[t] / raw_at[t] for t in keep],
                     "o-", ms=3, lw=1, color=color, label=label)
    axes[0].set(title="raw λ_max (held-out probe Hessian)", ylabel="λ_max")
    axes[1].set(title="preconditioned λ_max^pre", ylabel="λ_max^pre")
    axes[2].set(title="λ_pre / λ_raw: amplification by the preconditioner", ylabel="λ_pre / λ_max")
    for ax in axes:
        ax.set(xscale="log", yscale="log", xlabel="training tokens (M)")
        ax.grid(alpha=0.3, which="both")
        plain_log_ticks(ax, "xy")
    axes[1].legend(fontsize=8)
    fig.suptitle("P3(c): β2 changes the raw curvature AdamW settles into, not the preconditioned one (warmup left out)")
    fig.tight_layout()
    savefig(fig, "p3c_beta2_raw_pre.png")


def main():
    runs = load_all()
    for name, value, config, ratios, final in summary_figure(runs):
        print(f"{name:12s} {value:18s} threshold {threshold(config):5.1f} ratio "
              f"{' / '.join(f'{r:.2f}' for r in ratios)}  final train {final:.3f}")
    knobs_figure(runs)
    full_batch_figure(runs)
    beta2_figure(runs)


if __name__ == "__main__":
    main()
