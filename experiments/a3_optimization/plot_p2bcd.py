"""Problem 2(b)-(d): EMA averaging, LR dips with rewarm, and horizon-free schedules.

Curves come from W&B through plot_p2a.fetch (finished runs cached under cache/p2/).
(b) plots/p2b_ema_curves.png   raw and EMA validation loss, and regret vs. the reference run
                               (best final loss in part (a): linear, LR 3e-3), each schedule at its
                               best peak LR.
    plots/p2b_ema_summary.png  final regret vs. averaging timescale, each schedule at its best peak LR.
(c) plots/p2c_rewarm.png       LR multiplier and regret vs. each dip's own control.
(d) plots/p2d_horizon_free.png horizon-free schedules read off at each budget vs. budget-tuned
                               linear decay.
Each part also prints a table.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p2_river_valley import (
    BUDGET_FRACTIONS,
    BUDGET_TUNED_RUNS,
    DIP_WINDOWS,
    EMA_DECAYS,
    HORIZON_FREE_RUNS,
    LOW_LR_RUNS,
    LR_SWEEP_RUNS,
    REWARM_CONTROLS,
    REWARM_RUNS,
    SCHEDULE_RUNS,
)
from experiments.a3_optimization.plot_p1a import savefig
from experiments.a3_optimization.plot_p2a import COLORS, KEYS, SCHEDULES, TOKENS_PER_STEP, fetch
from train import training_run_name

BASIC = ("learning_rate", "train_loss", "val_loss")
FULL_RUNS = SCHEDULE_RUNS + LR_SWEEP_RUNS + LOW_LR_RUNS
EMA_KEYS = {d: f"val_ema{d:g}_loss" for d in EMA_DECAYS}
EMA_CMAP = plt.cm.viridis(np.linspace(0, 0.9, len(EMA_DECAYS)))


def series(data, config, key="val_loss"):
    """{step: value} for one run's logged key."""
    steps, values = data[training_run_name(config)]["history"][key]
    return dict(zip(steps, values))


def regret(curve, ref):
    steps = sorted(set(curve) & set(ref))
    return np.array(steps, dtype=float), np.array([curve[s] - ref[s] for s in steps])


def tokens(steps):
    return np.asarray(steps, dtype=float) * TOKENS_PER_STEP / 1e6


def final(data, config, key="val_loss"):
    curve = series(data, config, key)
    return curve[max(curve)]


def reference(data):
    """The run with the lowest final raw validation loss among the schedule runs (linear, LR 3e-3)."""
    return min(FULL_RUNS, key=lambda c: final(data, c))


def best_per_schedule(data):
    """For each schedule, the peak LR with the lowest final raw validation loss."""
    return [min((c for c in FULL_RUNS if c.lr_schedule == schedule), key=lambda c: final(data, c))
            for schedule in SCHEDULES]


# ---------------------------------------------------------------- (b) EMA
def part_b(data):
    ref_config = reference(data)
    ref = series(data, ref_config)
    configs = best_per_schedule(data)
    ref_label = f"{SCHEDULES[ref_config.lr_schedule]} LR {ref_config.learning_rate:g}"
    fig, axes = plt.subplots(2, len(configs), figsize=(4.6 * len(configs), 8), sharex=True, sharey="row")
    for j, c in enumerate(configs):
        raw = series(data, c)
        ax, rx = axes[0, j], axes[1, j]
        s = sorted(raw)
        ax.plot(tokens(s), [raw[t] for t in s], color="k", lw=1.8, label="raw")
        st, r = regret(raw, ref)
        rx.plot(tokens(st), r, color="k", lw=1.8, label="raw")
        for (d, key), color in zip(EMA_KEYS.items(), EMA_CMAP):
            ema = series(data, c, key)
            s = sorted(ema)
            ax.plot(tokens(s), [ema[t] for t in s], color=color, lw=1, label=f"EMA {d:g} (~{1 / (1 - d):.0f} upd.)")
            st, r = regret(ema, ref)
            rx.plot(tokens(st), r, color=color, lw=1)
        ax.set(title=f"{SCHEDULES[c.lr_schedule]}, best peak LR {c.learning_rate:g}", ylim=(2.9, 3.35))
        rx.axhline(0, color="0.4", lw=0.8)
        rx.set(ylim=(-0.08, 0.25), xlabel="training tokens (M)")
        for a in (ax, rx):
            a.grid(alpha=0.3)
    axes[0, 0].set_ylabel("validation loss")
    axes[1, 0].set_ylabel(f"regret vs. {ref_label} (raw)")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("P2(b): evaluation-only parameter EMAs, each schedule at its best peak LR; "
                 f"regret R(t) = L_M(t) − L_ref(t) at equal tokens, ref = {ref_label}")
    fig.tight_layout()
    savefig(fig, "p2b_ema_curves.png")

    # Final regret vs. averaging timescale, each schedule at its best peak LR.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["raw"] + [f"{1 / (1 - d):.0f}" for d in EMA_DECAYS]
    final_ref = final(data, ref_config)
    print(f"\n(b) reference: {training_run_name(ref_config)}  final val {final_ref:.4f}")
    print(f"{'schedule':>16} {'LR':>7} {'raw':>8} " + " ".join(f"{'ema' + format(d, 'g'):>10}" for d in EMA_DECAYS)
          + f" {'gap closed':>10}")
    for c in configs:
        values = [final(data, c) - final_ref] + [final(data, c, k) - final_ref for k in EMA_KEYS.values()]
        ax.plot(range(len(values)), values, "o-", color=COLORS[c.lr_schedule],
                label=f"{SCHEDULES[c.lr_schedule]} (LR {c.learning_rate:g})")
        best = min(values[1:])
        closed = (values[0] - best) / values[0] if values[0] > 0 else float("nan")
        print(f"{SCHEDULES[c.lr_schedule]:>16} {c.learning_rate:7g} {values[0]:+8.4f} "
              + " ".join(f"{v:+10.4f}" for v in values[1:]) + f" {closed:10.0%}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(xlabel="EMA timescale 1/(1−β) in updates", ylabel=f"final regret vs. {ref_label} (symlog)",
           title="P2(b): final regret after averaging, each schedule at its best peak LR")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([-0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p2b_ema_summary.png")


# ---------------------------------------------------------------- (c) dips
def dip_multiplier(progress, dip):
    from lr_schedules import dip_multiplier as m

    return np.array([m(p, dip) for p in progress])


def part_c(data):
    total = SCHEDULE_RUNS[0].num_train_sequences // SCHEDULE_RUNS[0].batch_size
    depth_colors = {0.1: "C3", 0.5: "C0"}
    window_styles = dict(zip(DIP_WINDOWS.values(), ("-", "--", ":")))
    window_names = {v: k for k, v in DIP_WINDOWS.items()}
    schedules = list(REWARM_CONTROLS)
    fig, axes = plt.subplots(3, len(schedules), figsize=(7.5 * len(schedules), 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.6, 1.6]))
    print(f"\n(c) {'schedule':>8} {'depth':>5} {'window':>7} {'min regret':>10} {'at (M tok)':>10} "
          f"{'max regret':>10} {'at (M tok)':>10} {'final regret':>12} {'recovered (<0.005) at':>22}")
    for j, schedule in enumerate(schedules):
        control = series(data, REWARM_CONTROLS[schedule])
        lx, rx, zx = axes[0, j], axes[1, j], axes[2, j]
        for run in (r for r in REWARM_RUNS if r.lr_schedule == schedule):
            start, bottom, end, scale = run.lr_dip
            window = round(end - start, 4)
            style = dict(color=depth_colors[scale], ls=window_styles[window])
            label = f"×{scale:g}, {window_names[window]} ({window:.1%})"
            progress = np.linspace(0.37, 1.0, 2000)
            lx.plot(tokens(progress * total), dip_multiplier(progress, run.lr_dip), label=label, **style)
            st, r = regret(series(data, run), control)
            rx.plot(tokens(st), r, label=label, **style)
            zx.plot(tokens(st), r, **style)
            after_end = st > end * total
            recovered = next((t for t, v in zip(st[after_end], r[after_end])
                              if abs(v) < 0.005 and np.all(np.abs(r[(st >= t) & (st < 0.8 * total)]) < 0.005)), None)
            k, m = int(np.argmax(r)), int(np.argmin(r))
            print(f"{schedule:>12} {scale:5g} {window_names[window]:>7} {r[m]:+10.4f} {tokens(st[m]):10.0f} "
                  f"{r[k]:+10.4f} {tokens(st[k]):10.0f} "
                  f"{r[-1]:+12.4f} {'%.0fM tokens' % tokens(recovered) if recovered else 'not before cooldown':>22}")
        cooldown = 0.8 if schedule == "wsd0.2" else None
        for a in (lx, rx, zx):
            if cooldown:
                a.axvline(tokens(cooldown * total), color="0.5", ls=":", lw=1)
            a.grid(alpha=0.3)
        lx.set(title=f"{SCHEDULES[schedule]} control (dips start at 40% of training)",
               ylabel="LR multiplier from the dip")
        rx.axhline(0, color="0.4", lw=0.8)
        rx.set(ylabel="regret vs. control\n(full range)", ylim=(-0.15, 0.04),
               xlim=(tokens(0.37 * total), tokens(total) + 5))
        rx.legend(fontsize=8, loc="lower right")
        zx.axhline(0, color="0.4", lw=0.8)
        zx.set(xlabel="training tokens (M)", ylabel="regret vs. control\n(zoom: after the rewarm)", ylim=(-0.006, 0.034))
        if cooldown:
            zx.text(tokens(cooldown * total), 0.032, " final cooldown", fontsize=8, color="0.4", va="top")
    fig.suptitle("P2(c): decay the LR partway and rewarm, without resetting model or optimizer; "
                 "same data order and final cooldown as the control")
    fig.tight_layout()
    savefig(fig, "p2c_rewarm.png")


# ---------------------------------------------------------------- (d) horizon-free
def value_at(curve, step):
    """Validation loss at `step`, linearly interpolated between evaluations."""
    s = np.array(sorted(curve), dtype=float)
    return float(np.interp(step, s, [curve[t] for t in sorted(curve)]))


def fit_power_law(steps, values):
    """Least-squares L(t) = a + b t^-c: grid over c, linear solve for (a, b)."""
    t, y = np.asarray(steps, dtype=float), np.asarray(values)
    best = None
    for c in np.linspace(0.05, 3, 600):
        design = np.stack([np.ones_like(t), t ** -c], axis=1)
        (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        err = float(np.sum((design @ [a, b] - y) ** 2))
        if best is None or err < best[0]:
            best = (err, (float(a), float(b), float(c)))
    return best[1]


def part_d(data):
    total = SCHEDULE_RUNS[0].num_train_sequences // SCHEDULE_RUNS[0].batch_size
    budgets = [round(f * total) for f in BUDGET_FRACTIONS]
    tuned = {}
    for b, f in zip(budgets, BUDGET_FRACTIONS):
        pool = [c for c in BUDGET_TUNED_RUNS if abs(c.num_train_sequences - f * SCHEDULE_RUNS[0].num_train_sequences) < 1]
        if f == 1.0:
            pool = [c for c in FULL_RUNS if c.lr_schedule == "linear"]
        finals = [(series(data, c)[max(series(data, c))], c.learning_rate) for c in pool]
        tuned[b] = min(finals)
    horizon_free = {"invsqrt": HORIZON_FREE_RUNS[0], "SGDR (T = 2000)": HORIZON_FREE_RUNS[1],
                    "constant": next(c for c in SCHEDULE_RUNS if c.lr_schedule == "wsd0.0")}
    colors = {"invsqrt": "C4", "SGDR (T = 2000)": "C5", "constant": "C3"}

    fig, (sx, lx, rx) = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3, 1]))
    for name, c in horizon_free.items():
        lr = series(data, c, "learning_rate")
        s = sorted(lr)
        sx.plot(tokens(s), [lr[t] for t in s], color=colors[name], label=name)
        val = series(data, c)
        s = sorted(val)
        lx.plot(tokens(s), [val[t] for t in s], color=colors[name], label=name)
    ref = next(c for c in FULL_RUNS if c.lr_schedule == "linear" and c.learning_rate == tuned[total][1])
    val = series(data, ref)
    s = sorted(val)
    lx.plot(tokens(s), [val[t] for t in s], color="C0", lw=1, alpha=0.6, label=f"linear to full budget (LR {ref.learning_rate:g})")
    lx.plot(tokens(budgets), [tuned[b][0] for b in budgets], "k*", ms=12, zorder=5, label="linear tuned to each budget (best of 2–3 LRs)")
    # SGDR's minimum in each completed cycle (just before each restart), and a power law through them.
    sgdr = HORIZON_FREE_RUNS[1]
    cycle = int(sgdr.lr_schedule.removeprefix("sgdr"))
    val = series(data, sgdr)
    minima = []
    for k in range(1, (total - sgdr.warmup_steps) // cycle + 1):
        lo, hi = sgdr.warmup_steps + (k - 1) * cycle, sgdr.warmup_steps + k * cycle
        step = min((t for t in val if lo < t <= hi), key=val.get)
        minima.append((step, val[step]))
    (a, b, c), t = fit_power_law(*zip(*minima)), np.linspace(minima[0][0] * 0.5, total, 300)
    lx.plot(tokens([m[0] for m in minima]), [m[1] for m in minima], "o", color=colors["SGDR (T = 2000)"],
            mfc="white", ms=7, zorder=6)
    lx.plot(tokens(t), a + b * t ** -c, ":", color=colors["SGDR (T = 2000)"], lw=2,
            label=f"SGDR cycle minima, fit {a:.3f} + {b:.3g}·t^−{c:.2f}")
    ta, tb, tc = fit_power_law(budgets, [tuned[b][0] for b in budgets])
    lx.plot(tokens(t), ta + tb * t ** -tc, ":", color="k", lw=1.5,
            label=f"budget-tuned, fit {ta:.3f} + {tb:.3g}·t^−{tc:.2f}")
    print(f"\n(d) budget-tuned fit L = {ta:.4f} + {tb:.4g} t^-{tc:.3f}; residuals "
          + ", ".join(f"{tuned[b][0] - (ta + tb * b ** -tc):+.4f}" for b in budgets))
    print("(d) SGDR cycle minima: " + ", ".join(f"{tokens(s):.0f}M {v:.4f}" for s, v in minima)
          + f"; fit L = {a:.4f} + {b:.4g} t^-{c:.3f} (t in updates), at {tokens(total):.0f}M: {a + b * total ** -c:.4f}")
    for b in budgets:
        for a in (sx, lx):
            a.axvline(tokens(b), color="0.8", lw=0.8, zorder=0)
    sx.set(xlabel="training tokens (M)", ylabel="learning rate", title="Horizon-free schedules (warmup 94 updates)")
    sx.legend(fontsize=8)
    lx.set(xlabel="training tokens (M)", ylabel="validation loss", ylim=(2.9, 3.6),
           title="Validation loss; ★ = budget-tuned final loss")
    lx.legend(fontsize=8)

    print(f"\n(d) {'budget':>10} {'tuned (LR)':>16} " + " ".join(f"{n:>18}" for n in horizon_free))
    width = 0.25
    for i, name in enumerate(horizon_free):
        values = [value_at(series(data, horizon_free[name]), b) - tuned[b][0] for b in budgets]
        rx.bar(np.arange(len(budgets)) + (i - 1) * width, values, width, color=colors[name], label=name)
    for b in budgets:
        print(f"{tokens(b):9.0f}M {tuned[b][0]:9.4f} ({tuned[b][1]:g}) " + " ".join(
            f"{value_at(series(data, c), b):9.4f} ({value_at(series(data, c), b) - tuned[b][0]:+.3f})"
            for c in horizon_free.values()))
    rx.axhline(0, color="0.4", lw=0.8)
    rx.set_xticks(range(len(budgets)), [f"{tokens(b):.0f}M\n({f:g} of run)" for b, f in zip(budgets, BUDGET_FRACTIONS)])
    rx.set(ylabel="regret vs. budget-tuned linear", title="Regret at each stopping budget")
    rx.legend(fontsize=8)
    for a in (sx, lx, rx):
        a.grid(alpha=0.3)
    fig.suptitle("P2(d): schedules chosen without knowing the budget, read off at 77M / 154M / 307M / 614M tokens")
    fig.tight_layout()
    savefig(fig, "p2d_horizon_free.png")


def main():
    data = fetch(FULL_RUNS, KEYS)
    data.update(fetch(list(REWARM_CONTROLS.values()) + REWARM_RUNS + HORIZON_FREE_RUNS + BUDGET_TUNED_RUNS, BASIC))
    part_b(data)
    part_c(data)
    part_d(data)


if __name__ == "__main__":
    main()
