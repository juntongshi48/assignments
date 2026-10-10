"""Problem 4.1(c): five-step depth-transfer sweeps for standard muP, Depth-muP, and CompleteP.

Width and reference width 64 (one head), reference depth 2 (r = L / 2), BF16 autocast with FP32 residuals.
At depth 2 the three prescriptions coincide (r = 1), so one sweep (policy 'mup') is the shared source.
Runs skip the per-matrix alignment snapshots (--no-alignment): part (d) needs only the feature diagnostics and
omega_move, and the snapshots would hold ~20 GB of inputs on CPU at depth 1000.
Runs are grouped into GPU containers; diverged runs are recorded as <name>.diverged.json; existing outputs skipped.
"""
import argparse
import json
from pathlib import PurePosixPath

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS, MODAL_ENVIRONMENT,
                         MODAL_SHARED_DATASETS_DIR, timestamped_modal_app_name, secrets)

OUTPUT_DIR = '/root/data/a2-stress'
WIDTH = 64
PRESCRIPTIONS = ('mup', 'depth-mup', 'completep')
PROBE_LRS = (.0003, .001, .003, .01, .03)  # 1-3 ladder anchored at .003: two points below, two above


def output_name(policy, depth, lr):
    return f'p41d-{policy}-L{depth}-lr{lr:g}.json'


HIGH_LRS = (.1, .3)  # extend the 1-3 ladder upward: most probes were best at the .03 edge


def high_runs():
    """.1 and .3 for the shared depth-2 source and each prescription at depth 100."""
    return [('mup', 2, lr) for lr in HIGH_LRS] + [(p, 100, lr) for p in PRESCRIPTIONS for lr in HIGH_LRS]


FILL_LRS = (.00548, .0173, .0548)  # geometric midpoints of .003-.01-.03-.1
DEEP_LRS = (.003, .01, .03, .1)


def fill_runs():
    """Midpoints for the shared depth-2 source and each prescription at depth 100 (7 points over .003-.1)."""
    return [('mup', 2, lr) for lr in FILL_LRS] + [(p, 100, lr) for p in PRESCRIPTIONS for lr in FILL_LRS]


def deep_runs():
    """Depth 1000 (r = 500) probes for each prescription."""
    return [(p, 1000, lr) for p in PRESCRIPTIONS for lr in DEEP_LRS]


def deep_fill_runs():
    """Midpoints at depth 1000, giving the same 7-point grid over .003-.1 as depths 2 and 100."""
    return [(p, 1000, lr) for p in PRESCRIPTIONS for lr in FILL_LRS]


def probe_runs():
    """Five probing LRs for the shared depth-2 source and for each prescription at depth 100."""
    runs = [('mup', 2, lr) for lr in PROBE_LRS]
    runs += [(policy, 100, lr) for policy in PRESCRIPTIONS for lr in PROBE_LRS]
    return runs


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', retries=0, timeout=6 * 3600)
def _run_group(jobs):
    import os
    from experiments.a2.p31_student import main
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    for policy, depth, lr in jobs:
        path = str(PurePosixPath(OUTPUT_DIR) / output_name(policy, depth, lr))
        diverged = path.replace('.json', '.diverged.json')
        if os.path.exists(path) or os.path.exists(diverged):
            print(f'skip existing {path}')
            continue
        args = ['--train-path', str(data / 'train'), '--val-path', str(data / 'val'), '--output', path,
                '--lr', str(lr), '--width', str(WIDTH), '--depth', str(depth), '--head-dim', '64',
                '--precision', 'mp', '--policy', policy, '--base-width', str(WIDTH), '--base-depth', '2',
                '--no-alignment']
        try:
            main(args)
        except FloatingPointError as e:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(diverged, 'w') as f:
                json.dump(dict(policy=policy, depth=depth, base_lr=lr, error=str(e)), f)
            print(f'{path}: diverged ({e})')
        finally:
            user_volume.commit()


def groups(runs, n_groups):
    """Split runs into n_groups with roughly equal cost (cost ~ depth)."""
    out, load = [[] for _ in range(n_groups)], [0.] * n_groups
    for run in sorted(runs, key=lambda r: -r[1]):
        i = load.index(min(load))
        out[i].append(run)
        load[i] += run[1]
    return [g for g in out if g]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('probe', 'high', 'fill', 'deep', 'deep-fill'), required=True)
    p.add_argument('--groups', type=int, default=2, help='parallel GPU containers')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    runs = {'probe': probe_runs, 'high': high_runs, 'fill': fill_runs, 'deep': deep_runs,
            'deep-fill': deep_fill_runs}[a.stage]()
    for policy, depth, lr in runs:
        print(f'{policy:<10} depth {depth:<5} base LR {lr:g}')
    print(f'{len(runs)} runs in {len(groups(runs, a.groups))} containers')
    if not a.execute:
        print('Preview only. Add --execute to launch.')
        return
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-p41d'), detach=True, environment_name=MODAL_ENVIRONMENT):
            remote = _run_group.with_options(secrets=secrets(include_wandb=True))
            for g in groups(runs, a.groups):
                call = remote.spawn(g)
                print(f'spawned {call.object_id}: {len(g)} runs')


if __name__ == '__main__':
    main()
