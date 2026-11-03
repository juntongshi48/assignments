"""P4 (b): pre-norm-across-time GRU proposals at 1x and 2x the GRU's best LR.

    uv run python -m experiments.a4_stability.launch_p4b          # dry run
    uv run python -m experiments.a4_stability.launch_p4b --execute
"""
import sys

from .sweeps import config

NAMES = ('gru-prenorm', 'gru-prenorm-linear')
RATES = (.003, .006)  # Base GRU: 3.273 / 3.190 / 3.438 at .001 / .003 / .009.
MAX_PARALLEL = 2  # Course environments allow 2 concurrent GPUs.


def configs(names=NAMES, rates=RATES):
    return [config(name, lr) for lr in rates for name in names]


if __name__ == '__main__':
    cfgs = configs()
    for cfg in cfgs:
        print(cfg.run_name, cfg.model_builder, cfg.learning_rate)
    print(f'{len(cfgs)} runs')
    if '--execute' in sys.argv:
        from modal_train import launch_training_jobs
        launch_training_jobs(cfgs, max_parallel_runs=MAX_PARALLEL)
