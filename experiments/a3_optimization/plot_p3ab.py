"""Problem 3(a) and (b): edge of stability, full-batch gradient descent and minibatch AdamW.

Reads the runs in p3_edge_of_stability (FULL_BATCH_RUNS, MINIBATCH_RUNS) from W&B through
sharpness.history (cached under cache/p3/).
plots/p3a_full_batch.png
  (a) i   lambda_max vs. update, and vs. training loss, with the 2 / eta threshold.
  (a) ii  eta_t * lambda_max and lambda_max for constant / halved / doubled LR at update 750,
          with the training loss.
plots/p3b_minibatch.png
  (b)     eta_t * lambda_pre on the held-out probe vs. training tokens (warmup left out), against
          38 = 2 (1 + beta1) / (1 - beta1); raw-LR-free lambda_pre; training loss.
Prints sharpness.summarize / window_median for each run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p3_edge_of_stability import FULL_BATCH_RUNS, MINIBATCH_RUNS
from experiments.a3_optimization.plot_p1a import savefig
from experiments.a3_optimization.sharpness import THRESHOLDS, History, history, summarize, window_median
from train import checked_train_config, training_run_name

CACHE = Path(__file__).resolve().parent / "cache" / "p3"
LABELS = {"constant": "constant η = 0.01", "step750x0.5": "η halved at 750", "step750x2": "η doubled at 750"}
COLORS = {"constant": "C0", "step750x0.5": "C2", "step750x2": "C3"}
ZOOM_FIRST, ZOOM_DOUBLED, ZOOM_HALVED = (300, 500), (700, 1000), (850, 1100)


def plain_number(v, _=None):
    """0.00005, 0.002, 20, 200, 10,000 -- never 5e-05 or 2x10^2."""
    if v == 0:
        return "0"
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    if abs(v) >= 1e-4:
        return f"{v:.10g}"
    decimals = -math.floor(math.log10(abs(v))) + 1
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def plain_log_ticks(ax, axes="y", subs=None):
    """Log-axis ticks at subs x 10^k (default 1, 2, 5; denser under 1.3 decades, only 1 over 4), as plain numbers."""
    for name in axes:
        axis = ax.yaxis if name == "y" else ax.xaxis
        lo, hi = (ax.get_ylim() if name == "y" else ax.get_xlim())
        decades = math.log10(max(hi, 1e-300) / max(lo, 1e-300)) if lo > 0 and hi > 0 else 0
        default = (1.0,) if decades > 4 else (1.0, 2.0, 5.0) if decades > 1.3 else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)
        axis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=subs or default))
        axis.set_major_formatter(mpl.ticker.FuncFormatter(plain_number))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


def load(config):
    """sharpness.history(config), cached by run name once the run has finished."""
    config = checked_train_config(config)
    path = CACHE / f"{training_run_name(config)}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        rows = history(config)
        data = {"rows": list(rows), "losses": rows.losses}
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    out = History(data["rows"])
    out.losses = [tuple(x) for x in data["losses"]]
    return out


def column(rows, key):
    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return np.array([p[0] for p in pts], dtype=float), np.array([p[1] for p in pts], dtype=float)


def full_batch_figure(runs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    clean = runs["constant"]
    eta = 0.01

    ax = axes[0, 0]
    s, lam = column(clean, "lambda_max")
    ax.plot(s, lam, "o-", ms=3, color=COLORS["constant"], label="λ_max")
    ax.axhline(2 / eta, color="k", ls="--", lw=1, label="2 / η = 200")
    ax.set(yscale="log", xlabel="update", ylabel="λ_max (full-batch Hessian)",
           title="(a) i  sharpness vs. update, N = 32, η = 0.01")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    loss_at = dict(clean.losses)
    pts = [(loss_at.get(r["step"]), r["lambda_max"]) for r in clean if r.get("lambda_max") is not None]
    pts = [p for p in pts if p[0] is not None]
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[r["step"] for r in clean
                    if r.get("lambda_max") is not None and loss_at.get(r["step"]) is not None],
                    cmap="viridis", s=14)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="0.7", lw=0.8, zorder=0)
    ax.axhline(2 / eta, color="k", ls="--", lw=1)
    ax.set(yscale="log", xscale="log", xlabel="training loss", ylabel="λ_max",
           title="(a) i  sharpness vs. training loss (training goes right → left)")
    ax.invert_xaxis()
    fig.colorbar(sc, ax=ax, label="update")

    ax = axes[1, 0]
    for name, rows in runs.items():
        s, prod = column(rows, "eta_lambda")
        ax.plot(s, prod, "o-", ms=2.5, lw=1, color=COLORS[name], label=LABELS[name])
    ax.axhline(THRESHOLDS["raw"], color="k", ls="--", lw=1, label="η λ_max = 2")
    ax.axvline(750, color="0.5", ls=":", lw=1)
    ax.set(yscale="log", xlabel="update", ylabel="η_t · λ_max",
           title="(a) ii  η_t λ_max when η changes at update 750")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, rows in runs.items():
        s, lam = column(rows, "lambda_max")
        ax.plot(s, lam, "o-", ms=2.5, lw=1, color=COLORS[name], label=f"λ_max, {LABELS[name]}")
    for name, factor in (("constant", 1), ("step750x0.5", 0.5), ("step750x2", 2)):
        ax.hlines(2 / (eta * factor), 750, 1500, color=COLORS[name], ls="--", lw=1)
    ax.hlines(2 / eta, 0, 750, color="k", ls="--", lw=1, label="2 / η_t")
    ax.axvline(750, color="0.5", ls=":", lw=1)
    ax.set(yscale="log", xlabel="update", ylabel="λ_max", title="(a) ii  λ_max against its threshold 2 / η_t")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.3, which="both")
        plain_log_ticks(a)
    plain_log_ticks(axes[0, 1], "x", subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
    fig.suptitle("P3(a): full-batch gradient descent on the first 32 training sequences (16× embedding init)")
    fig.tight_layout()
    savefig(fig, "p3a_full_batch.png")

    # Training loss (top) and eta_t * lambda_max (bottom). Columns: the whole run; the first crossing
    # of eta * lambda_max = 2 (same config before 750: the runs agree until the crossing, then GPU
    # nondeterminism is amplified); the LR change at 750 (all runs); the halved run climbing back
    # across 2 (that run only).
    first = [int(x) for x, y in zip(*column(runs["constant"], "eta_lambda")) if y > 2][0]
    reentry = [int(x) for x, y in zip(*column(runs["step750x0.5"], "eta_lambda")) if x > 750 and y > 2][0]
    windows = [
        ("whole run", (0, 1500), tuple(runs), None),
        ("first crossing of 2/η\n(same config before 750; split only after 2/η)", ZOOM_FIRST, tuple(runs),
         (first, "k", f"first above 2 at {first}")),
        ("LR change at 750: doubled run spikes above 2/η", ZOOM_DOUBLED, tuple(runs), None),
        ("η halved at 750: back across 2/η", ZOOM_HALVED, ("step750x0.5",),
         (reentry, COLORS["step750x0.5"], f"back above 2 at {reentry}")),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(21, 8), sharex="col", gridspec_kw=dict(height_ratios=(3, 2)))
    for col, (title, (lo, hi), names, mark) in enumerate(windows):
        top, bottom = axes[0, col], axes[1, col]
        for name in names:
            u, loss = map(np.array, zip(*runs[name].losses))
            keep = (u >= lo) & (u <= hi)
            top.plot(u[keep], loss[keep], lw=0.9, color=COLORS[name], label=LABELS[name])
            s, prod = column(runs[name], "eta_lambda")
            keep = (s >= lo) & (s <= hi)
            bottom.plot(s[keep], prod[keep], "o-", ms=3.5, lw=1, color=COLORS[name])
        bottom.axhline(THRESHOLDS["raw"], color="k", ls="--", lw=1, label="η λ_max = 2")
        for a in (top, bottom):
            if lo < 750 < hi:
                a.axvline(750, color="0.5", ls=":", lw=1)
            a.grid(alpha=0.3, which="both")
        if col == 0:
            for _, (zlo, zhi), _, _ in windows[1:]:
                for a in (top, bottom):
                    a.axvspan(zlo, zhi, color="0.5", alpha=0.15, zorder=0)
            top.set_yscale("log")
            bottom.set_yscale("log")
            plain_log_ticks(top)
            plain_log_ticks(bottom)
        else:
            for a in (top, bottom):
                a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
                a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain_number))
            top.legend(fontsize=8, loc="upper right")
        if mark is not None:
            x, color, text = mark
            for a in (top, bottom):
                a.axvline(x, color=color, ls="-.", lw=1)
            bottom.annotate(text, (x, 2), xytext=(15, -35), textcoords="offset points", fontsize=8,
                            arrowprops=dict(arrowstyle="->", color="0.4"))
        top.set(title=title, ylabel="training loss (full batch)" if col == 0 else None)
        bottom.set(xlim=(lo, hi), xlabel="update", ylabel="η_t · λ_max" if col == 0 else None)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("P3(a): training loss and η_t λ_max (η_t λ_max measured every 25 updates; dotted line = LR change at 750)")
    fig.tight_layout()
    savefig(fig, "p3a_losses.png")


def minibatch_figure(config, rows):
    config = checked_train_config(config)
    per_update = config.batch_size * config.train_dataset.context_length
    warmup = int(config.num_train_sequences * config.num_epochs / config.batch_size * config.warmup_percent)  # as train.py
    threshold = 2 * (1 + config.beta1) / (1 - config.beta1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    s, prod = column(rows, "eta_lambda_pre")
    keep = s > warmup
    axes[0].plot(s[keep] * per_update / 1e6, prod[keep], "o-", ms=3)
    axes[0].axhline(threshold, color="k", ls="--", lw=1, label=f"2(1+β1)/(1−β1) = {threshold:g}")
    axes[0].set(yscale="log", xlabel="training tokens (M)", ylabel="η_t · λ_max^pre",
                title="(b) preconditioned sharpness × LR (after warmup)")
    axes[0].legend(fontsize=8)
    # The first measurements (v-hat built from a few tiny gradients) reach millions; zoom on the
    # post-warmup range and note the off-scale points instead.
    s, lam = column(rows, "lambda_max_pre")
    post = lam[s > warmup]
    lo, hi = 0.85 * post.min(), 1.25 * post.max()
    axes[1].plot(s * per_update / 1e6, lam, "o-", ms=3, color="C1")
    axes[1].axvline(warmup * per_update / 1e6, color="0.5", ls=":", lw=1, label="end of warmup")
    off = lam > hi
    if off.any():
        axes[1].annotate(f"{off.sum()} warmup points off scale (up to {plain_number(lam.max())})",
                         (s[off][-1] * per_update / 1e6, hi), xytext=(12, -14), textcoords="offset points",
                         fontsize=8, arrowprops=dict(arrowstyle="->", color="0.4"))
    axes[1].set(ylim=(lo, hi), xlabel="training tokens (M)", ylabel="λ_max^pre (held-out probe)",
                title="(b) preconditioned sharpness (zoomed past warmup)")
    axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain_number))
    axes[1].legend(fontsize=8, loc="lower left")
    u, loss = zip(*rows.losses)
    u = np.array(u, dtype=float) * per_update / 1e6
    loss = np.array(loss)
    axes[2].plot(u, loss, lw=0.4, color="0.6", label="per update")
    if len(loss) >= 50:
        axes[2].plot(u[49:], np.convolve(loss, np.ones(50) / 50, mode="valid"), color="C0", label="50-update mean")
    axes[2].set(xlabel="training tokens (M)", ylabel="training loss", ylim=(2.9, 6), title="(b) training loss")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3, which="both")
    plain_log_ticks(axes[0])
    fig.suptitle("P3(b): minibatch AdamW (d8, batch 64, LR 3e-3 constant after warmup, 0.32 epoch)")
    fig.tight_layout()
    savefig(fig, "p3b_minibatch.png")


def main():
    runs = {checked_train_config(c).lr_schedule: load(c) for c in FULL_BATCH_RUNS}
    for name, rows in runs.items():
        print(f"(a) {name:12s}", {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in summarize(rows).items()})
    full_batch_figure(runs)

    config = MINIBATCH_RUNS[0]
    rows = load(config)
    s, prod = column(rows, "eta_lambda_pre")
    print("(b) eta*lambda_pre: min %.2f, max %.2f" % (prod.min(), prod.max()))
    for lo, hi in ((10e6, 50e6), (50e6, 100e6), (100e6, 180e6)):
        print(f"(b) median eta*lambda_pre over {lo/1e6:.0f}-{hi/1e6:.0f}M tokens:",
              round(window_median(rows, checked_train_config(config), lo, hi), 2))
    minibatch_figure(config, rows)


if __name__ == "__main__":
    main()
