"""Problem 4.1(a): five-step width-transfer sweeps for Kaiming and standard muP (widths 640, 2560, 5120).

Runs are grouped into a few GPU containers (one Modal call per group) so the image and data are loaded once
per group instead of once per run; each run still goes through p31_student.main, so outputs, W&B logging, and
filenames match a single modal_stress launch. A run whose loss or gradient goes nonfinite is recorded as
diverged in <name>.diverged.json instead of stopping the group. Existing outputs are skipped.

Stage 1 (--stage pilot): both policies at width 640 over a factor-2 grid, to locate the optimum.
Stage 1b (--stage pilot-extend): extra width-640 LRs once the pilot showed where the optimum is.
Stage 2 (--stage wide): widths 2560 and 5120 on the grids in WIDE_GRIDS.
Stage 2c (--stage kaiming-refine): fill-in points around the bracketed Kaiming optima.
Stage 2b (--stage kaiming-bracket): lower Kaiming LRs at 2560 and 5120, whose optima fell below the wide grids.
"""
import argparse
import json
from pathlib import PurePosixPath

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS, MODAL_ENVIRONMENT,
                         MODAL_SHARED_DATASETS_DIR, timestamped_modal_app_name, secrets)

OUTPUT_DIR = '/root/data/a2-stress'
PILOT_WIDTH = 640
PILOT_LRS = (.001, .002, .004, .008, .016, .032, .064)
PILOT_EXTRA_LRS = (.00025, .0005, .0014, .0028)  # added after the pilot put the optimum near .0015


def output_name(policy, width, lr):
    return f'p41w-{policy}-w{width}-lr{lr:g}.json'


def pilot_runs(lrs=PILOT_LRS):
    return [(policy, PILOT_WIDTH, lr) for policy in ('kaiming', 'mup') for lr in lrs]


# Wide-stage grids on the sqrt(2) LR ladder used at width 640. muP is predicted to keep its width-640 optimum
# (~.0016-.0019), so it reuses the bracketing width-640 points plus .0007 and an .00025 edge check. Kaiming's
# optimum is predicted to fall by m^-1/2 to m^-1 relative to width 640 (~.0014), so its grid moves down.
WIDE_GRIDS = {
    ('mup', 2560): (.00025, .0005, .0007, .001, .0014, .002, .0028, .004),
    ('mup', 5120): (.00025, .0005, .0007, .001, .0014, .002, .0028, .004),
    ('kaiming', 2560): (.00018, .00025, .00035, .0005, .0007, .001),
    ('kaiming', 5120): (.00009, .000125, .00018, .00025, .00035, .0005, .0007),
}


# Kaiming's optimum fell below both wide grids; extend each down a factor-2 ladder from its lowest point.
KAIMING_BRACKET = {2560: (.00009, .000045), 5120: (.000045, .0000225)}


# Fill-in points around the bracketed Kaiming optima (sqrt(2) neighbours, and one more factor-2 step at 5120).
KAIMING_REFINE = {2560: (round(.00018 / 2 ** .5, 9),),
                  5120: (.0000225 / 2, round(.000045 / 2 ** .5, 9), round(.000045 * 2 ** .5, 9))}


def kaiming_refine_runs():
    return [('kaiming', width, lr) for width, lrs in KAIMING_REFINE.items() for lr in lrs]


def kaiming_bracket_runs():
    return [('kaiming', width, lr) for width, lrs in KAIMING_BRACKET.items() for lr in lrs]


def wide_runs():
    return [(policy, width, lr) for (policy, width), lrs in WIDE_GRIDS.items() for lr in lrs]


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', retries=0, timeout=6 * 3600)
def _run_group(jobs):
    import os
    from experiments.a2.p31_student import main
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    for policy, width, lr in jobs:
        name = output_name(policy, width, lr)
        path = PurePosixPath(OUTPUT_DIR) / name
        if os.path.exists(path) or os.path.exists(str(path).replace('.json', '.diverged.json')):
            print(f'skip existing {name}')
            continue
        args = ['--train-path', str(data / 'train'), '--val-path', str(data / 'val'), '--output', str(path),
                '--lr', str(lr), '--width', str(width), '--depth', '2', '--head-dim', '64',
                '--precision', 'fp32', '--policy', policy]
        try:
            main(args)
        except FloatingPointError as e:  # nonfinite loss or gradient: a diverged LR, not a bug
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(str(path).replace('.json', '.diverged.json'), 'w') as f:
                json.dump(dict(policy=policy, width=width, base_lr=lr, error=str(e)), f)
            print(f'{name}: diverged ({e})')
        finally:
            user_volume.commit()


def groups(runs, n_groups):
    """Split runs into n_groups with roughly equal cost (cost ~ width^2)."""
    out, load = [[] for _ in range(n_groups)], [0.] * n_groups
    for run in sorted(runs, key=lambda r: -r[1]):
        i = load.index(min(load))
        out[i].append(run)
        load[i] += (run[1] / PILOT_WIDTH) ** 2
    return [g for g in out if g]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('pilot', 'pilot-extend', 'wide', 'kaiming-bracket', 'kaiming-refine'), required=True)
    p.add_argument('--groups', type=int, default=2, help='parallel GPU containers')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    runs = {'pilot': pilot_runs, 'pilot-extend': lambda: pilot_runs(PILOT_EXTRA_LRS),
            'wide': wide_runs, 'kaiming-bracket': kaiming_bracket_runs,
            'kaiming-refine': kaiming_refine_runs}[a.stage]()
    for policy, width, lr in runs:
        print(f'{policy:<8} width {width:<5} base LR {lr:g}')
    print(f'{len(runs)} runs in {len(groups(runs, a.groups))} containers')
    if not a.execute:
        print('Preview only. Add --execute to launch.')
        return
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-p41w'), detach=True, environment_name=MODAL_ENVIRONMENT):
            remote = _run_group.with_options(secrets=secrets(include_wandb=True))
            for g in groups(runs, a.groups):
                call = remote.spawn(g)
                print(f'spawned {call.object_id}: {len(g)} runs')


if __name__ == '__main__':
    main()
