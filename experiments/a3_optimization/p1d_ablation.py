"""Problem 1(d): does peak LR or batch size move the averaging boundary? Reruns the P1 (x, y) grid.

Each variant changes one knob of the P1 base (constant LR after warmup, LR 3e-3, batch 64):

    lr1.5e-3, lr6e-3   peak LR x0.5 / x2
    bs128              batch x2 at LR 3e-3 (half the gradient noise per update)

The grid is in updates, as in parts (a)-(c). x = 1000 is dropped (it always averaged at the base
setting), so each prefix stops at 500. bs128 fixes warmup_steps=94 (the base's 1% of 9375) so the LR
schedule is the same function of the update count as the base; its default would be 47.

    uv run python -m experiments.a3_optimization.p1d_ablation preview
    uv run python -m experiments.a3_optimization.p1d_ablation prefix [VARIANT ...]
    uv run python -m experiments.a3_optimization.p1d_ablation branches [VARIANT ...]
    uv run python -m experiments.a3_optimization.p1d_ablation measure [VARIANT ...]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.a3_optimization.p1_mode_connectivity import BASE, BRANCH_ORDER_SEEDS, BRANCH_STEPS
from train import training_run_name

EXPERIMENT_KEY = "a3-mode-connectivity-ablation"
PREFIX_STEPS = (0, 50, 100, 250, 500)
base = replace(BASE, wandb_tags=(*BASE.wandb_tags, EXPERIMENT_KEY))
VARIANTS = {
    "lr1.5e-3": replace(base, learning_rate=1.5e-3),
    "lr6e-3": replace(base, learning_rate=6e-3),
    "bs128": replace(base, batch_size=128, warmup_steps=94),
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def results_path(variant):
    return RESULTS_DIR / f"{EXPERIMENT_KEY}-{variant}.json"


def prefix(variant):
    return replace(VARIANTS[variant], keep_checkpoint_steps=tuple(x for x in PREFIX_STEPS if x > 0),
                   stop_at_step=max(PREFIX_STEPS))


def branch(variant, x, seed):
    fork = {}
    if x > 0:
        fork = dict(fork_from_run=training_run_name(prefix(variant)), fork_from_step=x)
    return replace(VARIANTS[variant], batch_order_seed=seed, stop_at_step=x + max(BRANCH_STEPS),
                   keep_checkpoint_steps=tuple(x + y for y in BRANCH_STEPS), **fork)


def branches(variant):
    return {x: [branch(variant, x, seed) for seed in BRANCH_ORDER_SEEDS] for x in PREFIX_STEPS}


def pairs(variant):
    """[(dict(x, y), ((run_a, step), (run_b, step)))] for every cell."""
    out = []
    for x, (a, b) in branches(variant).items():
        for y in BRANCH_STEPS:
            out.append((dict(x=x, y=y), ((training_run_name(a), x + y), (training_run_name(b), x + y))))
    return out


def preview(variants):
    for v in variants:
        p = prefix(v)
        print(f"{v}: prefix {training_run_name(p)} (stop {p.stop_at_step}, keep {p.keep_checkpoint_steps})")
        for x, runs in branches(v).items():
            print(f"  x={x:<4} " + "  ".join(training_run_name(r) for r in runs))
        steps = p.stop_at_step + sum(r.stop_at_step - (r.fork_from_step or 0)
                                     for runs in branches(v).values() for r in runs)
        print(f"  {1 + 2 * len(PREFIX_STEPS)} runs, {steps} updates at batch {p.batch_size}; "
              f"measure {len(pairs(v))} cells")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "prefix", "branches", "measure"))
    parser.add_argument("variants", nargs="*", help=f"subset of {list(VARIANTS)} (default: all)")
    args = parser.parse_args()
    variants = args.variants or list(VARIANTS)
    if unknown := set(variants) - set(VARIANTS):
        parser.error(f"unknown variants {sorted(unknown)}; choose from {list(VARIANTS)}")
    if args.stage == "preview":
        preview(variants)
    elif args.stage == "prefix":
        from modal_train import launch_training_jobs

        launch_training_jobs([prefix(v) for v in variants])
    elif args.stage == "branches":
        from experiments.a3_optimization.p1_mode_connectivity import check_prefix_checkpoints
        from modal_train import launch_training_jobs

        for v in variants:
            check_prefix_checkpoints(prefix(v))
        launch_training_jobs([r for v in variants for runs in branches(v).values() for r in runs])
    else:
        from experiments.a3_optimization.connectivity import measure_pairs_on_modal

        labeled = [(v, label, pair) for v in variants for label, pair in pairs(v)]
        rows = measure_pairs_on_modal([pair for _, _, pair in labeled])
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for v in variants:
            out = [dict(row, variant=v, **label) for (rv, label, _), row in zip(labeled, rows) if rv == v]
            results_path(v).write_text(json.dumps(out, indent=2) + "\n")
            print(f"Wrote {len(out)} cells to {results_path(v)}")


if __name__ == "__main__":
    main()
