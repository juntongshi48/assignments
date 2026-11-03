"""P3 (a)-(c): fixed, full, and blocked residual variants at (1/3, 1, 3) x center.

    uv run python -m experiments.a4_stability.launch_p3abc          # dry run
    uv run python -m experiments.a4_stability.launch_p3abc --execute
"""
import sys

from .sweeps import VARIANTS, config

FACTORS = (1/3, 1., 3.)
NAMES = tuple(name for name, variant in VARIANTS.items() if variant.problem == 3)
MAX_PARALLEL = 2  # Course environments allow 2 concurrent GPUs.


def rates(name):
    return tuple(float(f'{VARIANTS[name].center*factor:.3g}') for factor in FACTORS)


def configs(names=NAMES):
    return [config(name, lr) for name in names for lr in rates(name)]


def followup():
    """Full AttnRes d8 OOMs at batch 64 on an H100 (17 routers each keep an FP32 stack of
    up to 17 sources); two micro-batches of 32 keep the batch but skip torch.compile.
    d8-block2's best rate was the low edge of its grid."""
    return ([config('d8-attention', lr, num_micro_batches=2) for lr in rates('d8-attention')]
            + [config('d8-block2', .001)])


if __name__ == '__main__':
    cfgs = followup() if '--followup' in sys.argv else configs()
    for cfg in cfgs:
        print(cfg.run_name, cfg.model_builder, cfg.learning_rate)
    print(f'{len(cfgs)} runs')
    if '--execute' in sys.argv:
        from modal_train import launch_training_jobs
        launch_training_jobs(cfgs, max_parallel_runs=MAX_PARALLEL)
