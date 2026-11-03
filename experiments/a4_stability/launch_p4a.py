"""P4 (a): base GRU and its component ablations at (1/3, 1, 3) x center.

    uv run python -m experiments.a4_stability.launch_p4a          # dry run
    uv run python -m experiments.a4_stability.launch_p4a --execute
"""
import sys

from .sweeps import VARIANTS, config

FACTORS = (1/3, 1., 3.)
NAMES = tuple(name for name, variant in VARIANTS.items() if variant.problem == 4)
MAX_PARALLEL = 2  # Course environments allow 2 concurrent GPUs.


def rates(name):
    return tuple(float(f'{VARIANTS[name].center*factor:.3g}') for factor in FACTORS)


def configs(names=NAMES):
    return [config(name, lr) for name in names for lr in rates(name)]


if __name__ == '__main__':
    cfgs = configs()
    for cfg in cfgs:
        print(cfg.run_name, cfg.model_builder, cfg.learning_rate)
    print(f'{len(cfgs)} runs')
    if '--execute' in sys.argv:
        from modal_train import launch_training_jobs
        launch_training_jobs(cfgs, max_parallel_runs=MAX_PARALLEL)
