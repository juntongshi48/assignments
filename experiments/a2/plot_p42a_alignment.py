"""Problem 4.2(a): update alignment and readout--feature-change alignment through training at the source LR.

At peak LR .003 (Problem 1's best sampled LR at 153.6M tokens), for the course baseline (SP) and muP at widths
128 and 256 (our runs' alignment.jsonl / features.jsonl, downloaded from W&B into cache/p42/) and the supplied
width-512 reference (reference_diagnostics(), shared: m = 1). Per-width alpha_upd = log_k R(dA_t, u_{t-1}),
averaged over the 8 layers (band: min-max over layers), and omega_move = log_n S(V_0, h_t - h_0).
Also the cross-width exponent: slope of log R = log C + alpha log n over widths 128, 256, 512 at each step.
Writes plots/p42a_alignment.png and plots/p42a_alignment_cross_width.png.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import PROJECT_PATH, load_runs
from experiments.a2.mup import EXPERIMENT_KEY, SOURCE_LR
from experiments.a2.plot_p1a import savefig
from experiments.a2.provided_sweeps import reference_diagnostics

CACHE = Path(__file__).resolve().parent / "cache" / "p42"
MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
PRESCRIPTIONS = {"sp": ("course baseline (SP)", "--"), "mup": ("muP", "-")}
WIDTH_COLORS = {128: "C1", 256: "C2", 512: "k"}


def _download(run, name):
    path = CACHE / run.id / name
    if not path.exists():
        import wandb
        wandb.Api().run(f"{PROJECT_PATH}/{run.id}").file(name).download(root=str(path.parent), replace=True)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def logs():
    """{(prescription, width): (alignment rows, [(step, omega_move)])}; width 512 is the shared reference."""
    out = {}
    for r in load_runs([EXPERIMENT_KEY]):
        if r.state != "finished" or r.config["learning_rate"] != SOURCE_LR:
            continue
        prescription = r.name.rsplit("-", 1)[-1]
        width = int(r.name.split("-w", 1)[1].split("-", 1)[0])
        features = _download(r, "features.jsonl")
        omega = [(f["step"], f["readout_alignment"]["movement"]["omega"]) for f in features]
        out[(prescription, width)] = (_download(r, "alignment.jsonl"), omega)
    ref = reference_diagnostics()
    omega = [(f["step"], f["movement"]["omega"]) for f in ref["readout_alignment"] if f["kind"] == "fixed_batch"]
    for p in PRESCRIPTIONS:
        out[(p, 512)] = (ref["alignment"], omega)
    return out


def by_matrix(rows):
    """{matrix: {step: [(alpha, log R, fan_in) per layer]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["alpha"] is None:
            continue
        name = "lm_head" if r["parameter"].startswith("lm_head") else r["parameter"].split(".")[-2]
        out[name][r["step"]].append((r["alpha"], r["alpha"] * np.log(r["fan_in"])))
    return out


def main():
    data = logs()
    widths = sorted({w for _, w in data})
    fig, axes = plt.subplots(2, 5, figsize=(24, 8.5), sharex=True)
    panels = list(MATRICES) + ["omega_move"]
    for ax, name in zip(axes.flat, panels):
        for (p, w), (rows, omega) in sorted(data.items()):
            if w == 512 and p == "mup":
                continue  # shared reference, drawn once
            label = "width 512 reference (shared)" if w == 512 else f"{PRESCRIPTIONS[p][0]}, width {w}"
            style = "-" if w == 512 else PRESCRIPTIONS[p][1]
            if name == "omega_move":
                s = [(t, o) for t, o in omega if o is not None]
                ax.plot([t for t, _ in s], [o for _, o in s], style, marker="o", ms=3, color=WIDTH_COLORS[w], label=label)
                continue
            m = by_matrix(rows)[name]
            steps = sorted(m)
            mean = np.array([np.mean([a for a, _ in m[t]]) for t in steps])
            lo = np.array([np.min([a for a, _ in m[t]]) for t in steps])
            hi = np.array([np.max([a for a, _ in m[t]]) for t in steps])
            ax.plot(steps, mean, style, color=WIDTH_COLORS[w], lw=1.3, label=label)
            if name != "lm_head":
                ax.fill_between(steps, lo, hi, color=WIDTH_COLORS[w], alpha=.08)
        for y in (.5, 1):
            ax.axhline(y, color="0.5", ls=":", lw=1)
        ax.set(xscale="log", title=name if name != "omega_move" else r"$\omega_{\rm move}$ (readout vs. feature change)")
        ax.set_ylim(.3, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("optimizer update t")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"per-width $\alpha_{\rm upd}^{(t)}=\log_k R$")
    axes[0][0].legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    fig.suptitle(f"Alignment through training at the source LR {SOURCE_LR:g} (153.6M tokens): mean over 8 layers, "
                 "band = min-max over layers; dashed = SP, solid = muP; grey lines at 1/2 and 1", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p42a_alignment.png")

    # Cross-width exponent per step: slope of log R vs log n over widths 128, 256, 512 (layer-averaged log R).
    fig, axes = plt.subplots(2, 5, figsize=(24, 8.5), sharex=True)
    print("Cross-width exponent (slope of log R vs log n over widths 128/256/512), layer-averaged, at steps "
          "2, 5, 23, 234, 2344")
    for ax, name in zip(axes.flat, panels):
        for p, (label, style) in PRESCRIPTIONS.items():
            per_width = {}
            for w in widths:
                rows, omega = data[(p, w)]
                if name == "omega_move":
                    per_width[w] = {t: o * np.log(w) for t, o in omega if o is not None}
                else:
                    m = by_matrix(rows)[name]
                    per_width[w] = {t: np.mean([lr for _, lr in v]) for t, v in m.items()}
            steps = sorted(set.intersection(*(set(v) for v in per_width.values())))
            slope = [np.polyfit(np.log(widths), [per_width[w][t] for w in widths], 1)[0] for t in steps]
            ax.plot(steps, slope, style, marker="o", ms=3, color="C3" if p == "sp" else "C0", label=label)
            pick = {t: s for t, s in zip(steps, slope) if t in (2, 5, 23, 234, 2344)}
            print(f"  {name:>11} {label:>22}: " + "  ".join(f"t={t}: {s:.2f}" for t, s in sorted(pick.items())))
        for y in (.5, 1):
            ax.axhline(y, color="0.5", ls=":", lw=1)
        ax.set(xscale="log", title=name if name != "omega_move" else r"$\omega_{\rm move}$")
    for ax in axes[-1]:
        ax.set_xlabel("optimizer update t")
    for ax in axes[:, 0]:
        ax.set_ylabel("cross-width exponent")
    axes[0][0].legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    fig.suptitle(r"Cross-width exponent at the source LR: slope of $\log R=\log C+\alpha\log n$ over widths "
                 "128, 256, 512 (layer-averaged log R)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p42a_alignment_cross_width.png")


if __name__ == "__main__":
    main()
