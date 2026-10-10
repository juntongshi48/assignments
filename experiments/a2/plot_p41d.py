"""Problem 4.1(d): depth-scaling probes at each (prescription, depth)'s best sampled LR (reuses 4.1(c) runs).

Through the five updates on the fixed probes: residual-stream RMS before normalization (input to the midpoint
block's first RMSNorm and to the final RMSNorm), branch-output RMS before multiplication by c_L (attention o and
feedforward down at the midpoint and last block), movement of the normalized residual-stream features at matched
relative positions (midpoint block's first RMSNorm output; final RMSNorm output), and omega_move.
Depth 2 is the shared source (the prescriptions coincide at r = 1). Writes plots/p41d_depth_probes.png.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import savefig
from experiments.a2.plot_p41c import PRESCRIPTIONS, load

DEPTH_STYLE = {2: ":", 100: "--", 1000: "-"}


def best_records():
    """{(prescription, depth): (best sampled LR, record)}; depth 2 is shared."""
    curves, records = load()
    out = {}
    for (policy, depth), curve in curves.items():
        finite = {lr: v for lr, v in curve.items() if np.isfinite(v)}
        lr = min(finite, key=finite.get)
        src = "mup" if depth == 2 else policy
        out[(policy, depth)] = (lr, records[(src, depth, lr)])
    return out


def series(record):
    """Per-step probes keyed by quantity; positions are the run's probed block indices."""
    h = record["history"]
    blocks = record["probe_block_indices"]
    mid, last = blocks[len(blocks) // 2], blocks[-1]

    def get(fn):
        return np.array([fn(row) for row in h])

    omega = [row["readout_alignment"].get("movement", {}).get("omega") for row in h]
    return dict(
        step=get(lambda r: r["step"]),
        resid_mid=get(lambda r: r["residual_rms"][f"blocks.{mid}.norm1"]),
        resid_final=get(lambda r: r["residual_rms"]["final_norm"]),
        branch_o=get(lambda r: r["unscaled_branch_rms"][f"blocks.{last}.o"]),
        branch_down=get(lambda r: r["unscaled_branch_rms"][f"blocks.{last}.down"]),
        move_mid=get(lambda r: r["features"][f"blocks.{mid}.norm1"]["movement"]),
        move_final=get(lambda r: r["features"]["final_norm"]["movement"]),
        norm_final=get(lambda r: r["features"]["final_norm"]["rms"]),
        omega=np.array([np.nan if w is None else w for w in omega], dtype=float),
        logit_rms=get(lambda r: r["logit_rms"]),
        mid=mid, last=last)


def main():
    best = best_records()
    depths = sorted({d for _, d in best})
    print("At each best sampled LR; values at t = 0 -> t = 5")
    print(f"{'':>13} {'L':>5} {'LR':>7}  {'resid RMS mid':>15} {'resid RMS final':>16} {'branch down (last)':>19} "
          f"{'norm RMS final':>15} {'M_t mid':>8} {'M_t final':>9} {'omega t=1..5':>22}")
    for policy, (label, _) in PRESCRIPTIONS.items():
        for d in depths:
            lr, rec = best[(policy, d)]
            s = series(rec)
            print(f"{label:>13} {d:>5} {lr:>7.3g}  {s['resid_mid'][0]:6.2f}->{s['resid_mid'][-1]:6.2f}  "
                  f"{s['resid_final'][0]:7.2f}->{s['resid_final'][-1]:6.2f}  {s['branch_down'][0]:8.3f}->{s['branch_down'][-1]:7.3f}  "
                  f"{s['norm_final'][0]:6.3f}->{s['norm_final'][-1]:5.3f}  {s['move_mid'][-1]:8.3f} {s['move_final'][-1]:9.3f}  "
                  + " ".join(f"{w:.2f}" for w in s["omega"][1:]))

    panels = (("resid_final", "residual-stream RMS before final norm", True),
              ("resid_mid", "residual-stream RMS before midpoint block's norm", True),
              ("branch_down", "last block's FFN branch RMS before $c_L$", True),
              ("move_mid", r"movement $M_t$ of midpoint normalized features", False),
              ("move_final", r"movement $M_t$ of final normalized features", False),
              ("omega", r"$\omega_{\rm move}^{(t)}$", False))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4))
    for ax, (key, title, log) in zip(axes, panels):
        for policy, (label, color) in PRESCRIPTIONS.items():
            for d in depths:
                if d == 2 and policy != "mup":
                    continue
                s = series(best[(policy, d)][1])
                ax.plot(s["step"], s[key], DEPTH_STYLE[d], marker="o", ms=3, color="0.35" if d == 2 else color,
                        label=f"depth 2 (shared)" if d == 2 else f"{label}, L={d}")
        if log:
            ax.set_yscale("log")
        if key == "omega":
            for y in (.5, 1):
                ax.axhline(y, color="0.5", ls=":", lw=1)
        ax.set(xlabel="update t", title=title)
        ax.title.set_fontsize(9)
        ax.set_xticks(range(6))
    axes[0].legend(fontsize=7)
    fig.suptitle("Depth probes at each prescription's best sampled LR (dotted = depth 2, dashed = 100, solid = 1000)",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "p41d_depth_probes.png")


if __name__ == "__main__":
    main()
