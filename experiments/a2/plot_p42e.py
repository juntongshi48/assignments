"""Problem 4.2(e): training-regime diagnostics for the 4.2 runs.

For every configuration (SP / muP at widths 128, 256, 1024; standard muP / Depth-muP / CompleteP at depths 4, 16;
width-512 depth-8 reference from reference_diagnostics()) at the directly transferred LR (.003) and at the best
sampled LR: logit RMS, movement of normalized hidden features (midpoint layer's input RMSNorm output and final
RMSNorm output), residual RMS before the final norm, omega_move, and the gradient diagnostics (pre-clip norm,
embedding fraction of the squared gradient norm, clip coefficient). Checkpoints: updates 0-5, 23, 234, 1172, 2344.
Logs are read from the Modal volume (ckpts/<run>/{features,gradients}.jsonl) into cache/p42e/.
Writes plots/p42e_width_regime.png and plots/p42e_depth_regime.png.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.results import load_runs
from experiments.a2.mup import DEPTH_KEY, EXPERIMENT_KEY, SOURCE_LR
from experiments.a2.plot_p1a import savefig
from experiments.a2.provided_sweeps import reference_diagnostics
from modal_utils import MODAL_ENVIRONMENT, VOLUME_NAME

CACHE = Path(__file__).resolve().parent / "cache" / "p42e"
WIDTH_COLORS = {128: "C1", 256: "C2", 512: "k", 1024: "C4"}
DEPTH_COLORS = {"mup": "C0", "depth-mup": "C2", "completep": "C1"}


def _volume_log(run_name, name):
    path = CACHE / run_name / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, f"/ckpts/{run_name}/{name}", str(path),
                        "--env", MODAL_ENVIRONMENT], check=True, capture_output=True)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(features, gradients, depth):
    """Per-checkpoint diagnostics: {quantity: (steps, values)}."""
    mid = f"model.layers.{depth // 2}.input_layernorm"
    steps = [f["step"] for f in features]
    omega = [f["readout_alignment"]["movement"]["omega"] for f in features]
    out = dict(
        logit_rms=(steps, [f["logit_rms"] for f in features]),
        move_final=(steps, [f["features"]["model.norm"]["movement"] for f in features]),
        move_mid=(steps, [f["features"][mid]["movement"] for f in features]),
        resid_final=(steps, [f["residual_rms"]["model.norm"] for f in features]),
        omega=(steps, [np.nan if w is None else w for w in omega]),
    )
    gsteps = [g["step"] for g in gradients]
    out.update(pre_clip_norm=(gsteps, [g["pre_clip_norm"] for g in gradients]),
               embedding_fraction=(gsteps, [g["embedding_fraction_squared_norm"] for g in gradients]),
               clip_coefficient=(gsteps, [g["clip_coefficient"] for g in gradients]))
    return out


def configurations():
    """{(family, key, which): (lr, final loss, diagnostics)}; which in {'transferred', 'best'}."""
    out = {}
    ref = reference_diagnostics()
    ref_features = [dict(f, readout_alignment=r) for f, r in zip(
        [f for f in ref["diagnostics"] if f["kind"] == "fixed_batch"],
        [r for r in ref["readout_alignment"] if r["kind"] == "fixed_batch"])]
    ref_diag = summarize(ref_features, ref["gradients"], 8)
    for family, tag in (("width", EXPERIMENT_KEY), ("depth", DEPTH_KEY)):
        groups = {}
        for r in load_runs([tag]):
            if r.state != "finished":
                continue
            label = r.name.rsplit(f"{tag}-", 1)[1]
            size = int(r.name.split("-w", 1)[1].split("-", 1)[0]) if family == "width" else \
                int(r.name.split("model-a2-d", 1)[1].split("-", 1)[0])
            if family == "width" and size == 512:
                continue  # the single extra width-512 point; the reference run covers width 512
            groups.setdefault((label, size), []).append(r)
        for (label, size), runs in groups.items():
            best = min(runs, key=lambda r: r.val_loss)
            transferred = next(r for r in runs if r.config["learning_rate"] == SOURCE_LR)
            for which, r in (("transferred", transferred), ("best", best)):
                depth = 8 if family == "width" else size
                diag = summarize(_volume_log(r.name, "features.jsonl"), _volume_log(r.name, "gradients.jsonl"), depth)
                out[(family, (label, size), which)] = (r.config["learning_rate"], r.val_loss, diag)
        out[(family, ("reference", 512 if family == "width" else 8), "transferred")] = (SOURCE_LR, 3.2291, ref_diag)
    return out


def at(diag, key, step):
    steps, values = diag[key]
    return values[steps.index(step)] if step in steps else np.nan


def print_table(cfgs, family):
    print(f"\n{family}: LR, final loss | logit RMS t=0,5,end | M_t final t=1,5,end | M_t mid t=5,end | "
          f"resid RMS end | omega t=5,end | clip coef t=1,5,23,end | emb frac t=1,23,end")
    for (fam, (label, size), which), (lr, loss, d) in sorted(cfgs.items(), key=lambda kv: str(kv[0])):
        if fam != family:
            continue
        print(f"  {label:>10} {size:>5} {which:>11}  lr {lr:<8g} {loss:.4f} | "
              f"{at(d, 'logit_rms', 0):.2f} {at(d, 'logit_rms', 5):.2f} {at(d, 'logit_rms', 2344):.2f} | "
              f"{at(d, 'move_final', 1):.2f} {at(d, 'move_final', 5):.2f} {at(d, 'move_final', 2344):.2f} | "
              f"{at(d, 'move_mid', 5):.2f} {at(d, 'move_mid', 2344):.2f} | {at(d, 'resid_final', 2344):7.2f} | "
              f"{at(d, 'omega', 5):.2f} {at(d, 'omega', 2344):.2f} | "
              f"{at(d, 'clip_coefficient', 1):.3f} {at(d, 'clip_coefficient', 5):.3f} "
              f"{at(d, 'clip_coefficient', 23):.3f} {at(d, 'clip_coefficient', 2344):.2f} | "
              f"{at(d, 'embedding_fraction', 1):.2f} {at(d, 'embedding_fraction', 23):.2f} "
              f"{at(d, 'embedding_fraction', 2344):.3f}")


def plot(cfgs, family, fname):
    panels = (("logit_rms", "logit RMS", True), ("move_final", r"movement $M_t$, final normalized features", False),
              ("move_mid", r"movement $M_t$, midpoint normalized features", False),
              ("resid_final", "residual RMS before final norm", True),
              ("clip_coefficient", "clip coefficient", True), ("embedding_fraction", "embedding share of |grad|$^2$", False))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4))
    for ax, (key, title, log) in zip(axes, panels):
        for (fam, (label, size), which), (lr, loss, d) in sorted(cfgs.items(), key=lambda kv: str(kv[0])):
            if fam != family:
                continue
            steps, values = d[key]
            x = np.array(steps, dtype=float) + 1  # log axis: plot update t at t + 1
            if family == "width":
                color = WIDTH_COLORS[size]
                style = "-" if label in ("mup", "reference") else "--"
                name = "width 512 reference" if label == "reference" else f"{'muP' if label == 'mup' else 'SP'} w{size}"
            else:
                color = "k" if label == "reference" else DEPTH_COLORS[label]
                style = "-" if size in (16, 8) else "--"
                name = "depth 8 reference" if label == "reference" else f"{label} L{size}"
            ax.plot(x, values, style, marker="o" if which == "best" else "x", ms=3, color=color,
                    alpha=1 if which == "transferred" else .55,
                    label=f"{name}, {'LR .003' if which == 'transferred' else f'best LR {lr:g}'}")
        ax.set(xscale="log", yscale="log" if log else "linear", xlabel="update t + 1", title=title)
        ax.title.set_fontsize(9)
    axes[0].legend(fontsize=6)
    fig.suptitle(("Width" if family == "width" else "Depth") + " runs: diagnostics on fixed inputs "
                 "(x = transferred LR .003, o = best sampled LR)", fontsize=11)
    fig.tight_layout()
    savefig(fig, fname)


def main():
    cfgs = configurations()
    for family, fname in (("width", "p42e_width_regime.png"), ("depth", "p42e_depth_regime.png")):
        print_table(cfgs, family)
        plot(cfgs, family, fname)


if __name__ == "__main__":
    main()
