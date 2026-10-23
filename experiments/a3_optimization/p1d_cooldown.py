"""Problem 1(d): does an LR cooldown change whether branches can be averaged? Fixed total budget T.

Every cell ends at T = 1000 updates; only the branch point x varies (y = T - x). Two variants:

    constant  constant LR (after warmup) through T, as in parts (a)-(c)
    cooldown  identical through COOLDOWN_START = 750, then linear decay to 0 at T

Both variants use the same data orders, so for x <= 750 a pair's two variants are identical up to
step 750. For x = 950 the shared trunk itself is already cooling when the branches split.
Runs fork from existing P1 checkpoints where possible. Stages must run in order, each after the
previous one has finished:

    uv run python -m experiments.a3_optimization.p1d_cooldown stage1
    uv run python -m experiments.a3_optimization.p1d_cooldown stage2
    uv run python -m experiments.a3_optimization.p1d_cooldown stage3
    uv run python -m experiments.a3_optimization.p1d_cooldown measure
    uv run python -m experiments.a3_optimization.p1d_cooldown preview   # no Modal jobs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.a3_optimization.p1_mode_connectivity import (
    BASE,
    BRANCH_ORDER_SEEDS,
    BRANCHES,
    PREFIX,
)
from train import training_run_name

EXPERIMENT_KEY = "a3-mode-connectivity-cooldown"
T = 1000
COOLDOWN_START = 750
BRANCH_POINTS = (0, 50, 250, 500, 750, 950)
COOLDOWN_CHECKPOINTS = (800, 900, T)
HORIZON = BASE.num_train_sequences // BASE.batch_size  # 9375 updates in the full schedule
# Linear decay from 1 at COOLDOWN_START to 0 at T. end=1.0 keeps the multiplier from returning
# to 1 before the end of the full schedule; every run stops at T anyway.
COOLDOWN = (COOLDOWN_START / HORIZON, T / HORIZON, 1.0, 0.0)
RESULTS_PATH = Path(__file__).resolve().parent / "results" / f"{EXPERIMENT_KEY}.json"

base = replace(BASE, wandb_tags=(*BASE.wandb_tags, EXPERIMENT_KEY))


def fork(source, step, *, seed=None, stop, keep, cooldown=False):
    """A run continuing `source` (a TrainConfig) from `step` to `stop`."""
    return replace(
        base,
        batch_order_seed=seed,
        lr_dip=COOLDOWN if cooldown else None,
        fork_from_run=training_run_name(source),
        fork_from_step=step,
        stop_at_step=stop,
        keep_checkpoint_steps=keep,
    )


# Shared trunks past step 500 (the P1 prefix kept only 500 and 1000 in this range).
TRUNK = fork(PREFIX, 500, stop=950, keep=(COOLDOWN_START, 950))
TRUNK_COOLED = fork(TRUNK, COOLDOWN_START, stop=950, keep=(950,), cooldown=True)


def constant_run(x, i):
    """Constant-LR branch for (x, seed index i) that reaches T, keeping COOLDOWN_START if it passes it."""
    seed = BRANCH_ORDER_SEEDS[i]
    if x in BRANCHES:
        source = BRANCHES[x][i]
        if source.stop_at_step >= T:
            return source  # (500, 500): the existing P1 branch already ends at T
        keep = tuple(s for s in (COOLDOWN_START, T) if s > source.stop_at_step)
        return fork(source, source.stop_at_step, seed=seed, stop=T, keep=keep)
    trunk_step = x
    return fork(TRUNK, trunk_step, seed=seed, stop=T, keep=(T,))


def cooldown_run(x, i):
    seed = BRANCH_ORDER_SEEDS[i]
    if x > COOLDOWN_START:
        return fork(TRUNK_COOLED, x, seed=seed, stop=T, keep=(T,), cooldown=True)
    if x == COOLDOWN_START:
        return fork(TRUNK, x, seed=seed, stop=T, keep=COOLDOWN_CHECKPOINTS, cooldown=True)
    # Fork the constant-LR path of the same pair at COOLDOWN_START.
    source = BRANCHES[x][i] if BRANCHES[x][i].stop_at_step >= COOLDOWN_START else constant_run(x, i)
    return fork(source, COOLDOWN_START, seed=seed, stop=T, keep=COOLDOWN_CHECKPOINTS, cooldown=True)


CONSTANT = {x: [constant_run(x, i) for i in range(len(BRANCH_ORDER_SEEDS))] for x in BRANCH_POINTS}
COOLED = {x: [cooldown_run(x, i) for i in range(len(BRANCH_ORDER_SEEDS))] for x in BRANCH_POINTS}

existing = {training_run_name(run) for runs in BRANCHES.values() for run in runs}
all_runs = [TRUNK, TRUNK_COOLED] + [r for runs in (*CONSTANT.values(), *COOLED.values()) for r in runs]
new_runs = {training_run_name(r): r for r in all_runs if training_run_name(r) not in existing}


def stages():
    """Split new runs into stages: a run goes one stage after the run it forks from."""
    stage_of = {name: 0 for name in existing}
    stage_of[training_run_name(PREFIX)] = 0
    pending = dict(new_runs)
    while pending:
        for name, run in list(pending.items()):
            if run.fork_from_run in stage_of:
                stage_of[name] = stage_of[run.fork_from_run] + 1
                del pending[name]
    out = {}
    for name in new_runs:
        out.setdefault(stage_of[name], []).append(new_runs[name])
    return dict(sorted(out.items()))


def cells():
    """[(label, ((run_a, step), (run_b, step)))] for every measured cell."""
    out = []
    for x in BRANCH_POINTS:
        a, b = CONSTANT[x]
        out.append((dict(variant="constant", x=x, step=T),
                    ((training_run_name(a), T), (training_run_name(b), T))))
        a, b = COOLED[x]
        for step in a.keep_checkpoint_steps:
            out.append((dict(variant="cooldown", x=x, step=step),
                        ((training_run_name(a), step), (training_run_name(b), step))))
    return out


def check_sources(runs):
    """Fail before launching if any fork source checkpoint is not on the volume yet."""
    from pathlib import PurePosixPath

    from modal.exception import NotFoundError

    from modal_train import _model_volume_path
    from modal_utils import user_volume

    missing = []
    for run in runs:
        try:
            entries = user_volume.listdir(_model_volume_path(run.fork_from_run))
        except NotFoundError:
            entries = []
        if f"step_{run.fork_from_step}.pt" not in {PurePosixPath(e.path).name for e in entries}:
            missing.append(f"{run.fork_from_run} step {run.fork_from_step}")
    if missing:
        raise SystemExit("Fork sources not ready (finish the previous stage first):\n  " + "\n  ".join(missing))


def preview():
    for stage, runs in stages().items():
        steps = sum(r.stop_at_step - r.fork_from_step for r in runs)
        print(f"stage{stage}: {len(runs)} runs, {steps} steps")
        for r in runs:
            print(f"  {training_run_name(r)}\n      fork {r.fork_from_run} @ {r.fork_from_step} -> "
                  f"{r.stop_at_step}, keep {r.keep_checkpoint_steps}, dip {r.lr_dip}")
    print(f"measure: {len(cells())} cells")
    for label, _ in cells():
        print(f"  {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "stage1", "stage2", "stage3", "measure"))
    args = parser.parse_args()
    if args.stage == "preview":
        preview()
    elif args.stage == "measure":
        from experiments.a3_optimization.connectivity import measure_pairs_on_modal

        labeled = cells()
        rows = measure_pairs_on_modal([pair for _, pair in labeled])
        for (label, _), row in zip(labeled, rows):
            row.update(label)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {len(rows)} cells to {RESULTS_PATH}")
    else:
        from modal_train import launch_training_jobs

        runs = stages()[int(args.stage.removeprefix("stage"))]
        check_sources(runs)
        launch_training_jobs(runs)


if __name__ == "__main__":
    main()
