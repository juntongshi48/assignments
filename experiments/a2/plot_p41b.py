"""Problem 4.1(b): width-scaling probes at each (parameterization, width)'s best sampled LR (reuses 4.1(a) runs).

Through the five updates on the fixed probes: logit RMS, RMS and movement M_t of the final RMSNorm output (before
the readout multiplier), readout--feature-change alignment omega_move, and update alignment alpha_upd for every
block's q, k, v, o, gate, up, down matrices and the readout (log base fan-in of R(dA_t, u_{t-1})).
Also fits the cross-width exponent: if R = C n^alpha, the per-width alpha = log_k R mixes alpha with
log C / log k, so alpha is estimated as the slope of log R against log n across the three widths, at each step.
Writes plots/p41b_features.png, plots/p41b_alignment.png, and plots/p41b_alignment_cross_width.png.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p41a import POLICIES, download, load

MATRICES = ("q", "k", "v", "o", "gate", "up", "down", "head")
STYLE = {"kaiming": "--", "mup": "-"}


def best_records():
    """{(policy, width): (best sampled LR, record)}."""
    download()
    curves, records = load()
    out = {}
    for (policy, width), curve in curves.items():
        finite = {lr: loss for lr, loss in curve.items() if np.isfinite(loss)}
        lr = min(finite, key=finite.get)
        out[(policy, width)] = (lr, records[(policy, width, lr)])
    return out


def series(record):
    """Per-step diagnostics (steps 0..5) from one run's history."""
    h = record["history"]
    fn = [row["features"]["final_norm"] for row in h]
    omega = [row["readout_alignment"].get("movement", {}).get("omega") for row in h]
    return dict(step=np.array([row["step"] for row in h]),
                logit_rms=np.array([row["logit_rms"] for row in h]),
                norm_rms=np.array([f["rms"] for f in fn]),
                movement=np.array([f["movement"] for f in fn]),
                omega=np.array([np.nan if w is None else w for w in omega], dtype=float))


def alphas(record):
    """{(block or 'readout', matrix): array of alpha over steps 1..5}."""
    out = defaultdict(dict)
    for row in record["alignment"]:
        parts = row["parameter"].split(".")
        key = ("readout", "head") if parts[0] == "head" else (int(parts[1]), parts[2])
        out[key][row["step"]] = np.nan if row["alpha"] is None else row["alpha"]
    return {k: np.array([v[s] for s in sorted(v)]) for k, v in out.items()}


def cross_width(best, widths):
    """{policy: {(block, matrix) or 'omega_move': (fitted slope, pairwise slope 1, pairwise slope 2) per step}}.

    The logged per-width alpha is log_k R, so log R = alpha * log k. Fit
    log R = log C + alpha log n across widths (n = width; for `down` the fan-in is 3.5 n, absorbed into C)."""
    out = {}
    for policy in POLICIES:
        ws = [w for w in widths if (policy, w) in best]
        logn = np.log(ws)
        logs = defaultdict(list)
        for w in ws:
            rec = best[(policy, w)][1]
            for key, v in alphas(rec).items():
                fan_in = {r["parameter"]: r["fan_in"] for r in rec["alignment"]}
                name = "head.weight" if key[0] == "readout" else f"blocks.{key[0]}.{key[1]}.weight"
                logs[key].append(v * np.log(fan_in[name]))
            logs["omega_move"].append(series(rec)["omega"][1:] * np.log(w))
        out[policy] = {}
        for key, rows in logs.items():
            y = np.array(rows)  # [width, step]
            fit = np.polyfit(logn, y, 1)[0]
            pair1 = (y[1] - y[0]) / (logn[1] - logn[0])
            pair2 = (y[2] - y[1]) / (logn[2] - logn[1])
            out[policy][key] = (fit, pair1, pair2)
    return out


def plot_cross_width(cross):
    keys = [(b, m) for m in MATRICES[:-1] for b in (0, 1)]
    fig, axes = plt.subplots(len(POLICIES), len(MATRICES) + 1, figsize=(2.6 * (len(MATRICES) + 1), 3.6 * len(POLICIES)),
                             sharey=True)
    steps = np.arange(1, 6)
    for row, policy in enumerate(POLICIES):
        panels = [(m, [(b, m) for b in (0, 1)]) for m in MATRICES[:-1]] + [("head", [("readout", "head")]),
                                                                           ("omega_move", ["omega_move"])]
        for ax, (title, ks) in zip(axes[row], panels):
            for k, style in zip(ks, ("-", ":")):
                fit, p1, p2 = cross[policy][k]
                ax.plot(steps, fit, style, marker="o", ms=4, color="C3" if policy == "kaiming" else "C0",
                        label="fit over 3 widths" if style == "-" else "block 1 fit")
                ax.plot(steps, p1, "v", ms=4, color="0.55", label="640->2560" if style == "-" else None)
                ax.plot(steps, p2, "^", ms=4, color="0.3", label="2560->5120" if style == "-" else None)
            for y in (.5, 1):
                ax.axhline(y, color="0.5", ls=":", lw=1)
            ax.set_title(f"{POLICIES[policy][0]}: {title}", fontsize=10)
            ax.set_xticks(steps)
            if row == len(POLICIES) - 1:
                ax.set_xlabel("update t")
        axes[row][0].set_ylabel("cross-width exponent")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(r"Cross-width exponent: slope of $\log R=\log C+\alpha\log n$ over widths 640, 2560, 5120 "
                 r"(solid = block 0, dotted = block 1; triangles = slopes between adjacent widths; last panel: $\omega_{\rm move}$)",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "p41b_alignment_cross_width.png")


def main():
    best = best_records()
    widths = sorted({w for _, w in best})
    colors = dict(zip(widths, plt.cm.viridis(np.linspace(0.1, 0.85, len(widths)))))

    print("Best sampled LR, logit RMS (t=0..5), final-norm movement M_t (t=1..5), omega_move (t=1..5)")
    for (policy, width), (lr, rec) in sorted(best.items()):
        s = series(rec)
        print(f"  {POLICIES[policy][0]:>12} w={width:<5} lr {lr:<10.3g} logit RMS "
              + " ".join(f"{x:.2f}" for x in s["logit_rms"])
              + "  |  M_t " + " ".join(f"{x:.3f}" for x in s["movement"][1:])
              + "  |  omega " + " ".join(f"{x:.2f}" for x in s["omega"][1:]))
    print("\nalpha_upd averaged over steps 1-5 (both blocks for hidden matrices)")
    print(f"  {'':>12} {'width':>6} " + " ".join(f"{m:>6}" for m in MATRICES))
    for (policy, width), (_, rec) in sorted(best.items()):
        a = alphas(rec)
        means = [np.nanmean(np.concatenate([v for (b, m), v in a.items() if m == name])) for name in MATRICES]
        print(f"  {POLICIES[policy][0]:>12} {width:>6} " + " ".join(f"{x:6.2f}" for x in means))

    cross = cross_width(best, widths)
    print("\nCross-width exponents: slope of log R = log C + alpha log n over the three widths, per step")
    print("  (pairwise slopes 640->2560 / 2560->5120 in brackets: similar values = constant C, power law holds)")
    for policy in POLICIES:
        for key in sorted(cross[policy], key=str):
            fit, p1, p2 = cross[policy][key]
            print(f"  {POLICIES[policy][0]:>12} {str(key):<16} " + "  ".join(
                f"{f:.2f} [{a:.2f}/{b:.2f}]" for f, a, b in zip(fit, p1, p2)))

    # Figure 1: logits, final-norm features, omega_move
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    panels = (("logit_rms", "logit RMS"), ("norm_rms", "final RMSNorm output RMS"),
              ("movement", r"movement $M_t$ of final RMSNorm output"), ("omega", r"$\omega_{\rm move}^{(t)}$"))
    for ax, (key, title) in zip(axes, panels):
        for (policy, width), (lr, rec) in sorted(best.items()):
            s = series(rec)
            ax.plot(s["step"], s[key], STYLE[policy], marker="o", ms=4, color=colors[width],
                    label=f"{POLICIES[policy][0]}, w={width} (LR {lr:.2g})")
        ax.set(xlabel="update t", title=title)
        ax.set_xticks(range(6))
    for y, name in ((.5, "no alignment (1/2)"), (1, "full alignment (1)")):
        axes[3].axhline(y, color="0.5", ls=":", lw=1)
        axes[3].annotate(name, (0.05, y), fontsize=8, color="0.4", va="bottom")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed-probe diagnostics at each configuration's best sampled LR "
                 "(solid = standard muP, dashed = Kaiming; colour = width)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p41b_features.png")

    # Figure 2: alpha_upd per matrix through the five updates
    fig, axes = plt.subplots(len(POLICIES), len(MATRICES), figsize=(2.9 * len(MATRICES), 3.6 * len(POLICIES)),
                             sharey=True)
    for row, policy in enumerate(POLICIES):
        for col, name in enumerate(MATRICES):
            ax = axes[row][col]
            for width in widths:
                if (policy, width) not in best:
                    continue
                a = alphas(best[(policy, width)][1])
                for (block, m), v in sorted(a.items(), key=lambda kv: str(kv[0])):
                    if m != name:
                        continue
                    ax.plot(range(1, len(v) + 1), v, ":" if block == 1 else "-", marker="o", ms=3,
                            color=colors[width], label=f"w={width}" if block in (0, "readout") else None)
            for y in (.5, 1):
                ax.axhline(y, color="0.5", ls=":", lw=1)
            ax.set_title(f"{POLICIES[policy][0]}: {name}", fontsize=10)
            ax.set_xticks(range(1, 6))
            if row == len(POLICIES) - 1:
                ax.set_xlabel("update t")
        axes[row][0].set_ylabel(r"$\alpha_{\rm upd}^{(t)}$")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(r"Update alignment $\alpha_{\rm upd}=\log_k R(\Delta A_t, u_{t-1})$ at the best sampled LR "
                 "(solid = block 0, dotted = block 1; grey lines at 1/2 and 1)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p41b_alignment.png")
    plot_cross_width(cross)


if __name__ == "__main__":
    main()
