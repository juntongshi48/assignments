import sys

from modal_train import DEFAULT_MODAL_GPU, _completed_model_exists, launch_training_jobs
from train import TrainConfig, checked_train_config, training_run_name

# Course accounts get 2 concurrent GPUs; extra calls queue inside the Modal app.
MAX_PARALLEL_RUNS = 2
DEFAULT_METRIC_LOGGERS = TrainConfig.metric_loggers


def main_for_parts(parts):
    """CLI for experiment files: `python -m <module> <part> [--dry-run]`.

    `parts` maps a part name to a builder returning list[TrainConfig], or to
    (builder, gpu) for parts that must run on a non-default GPU.
    """
    args = [arg for arg in sys.argv[1:] if arg != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if len(args) != 1 or args[0] not in parts:
        raise SystemExit(f"usage: python -m <module> {{{','.join(parts)}}} [--dry-run]")
    spec = parts[args[0]]
    builder, gpu = spec if isinstance(spec, tuple) else (spec, DEFAULT_MODAL_GPU)
    runs = builder()
    names = [training_run_name(checked_train_config(run)) for run in runs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise SystemExit(f"duplicate run names: {sorted(duplicates)}")
    if dry_run:
        for run, name in zip(runs, names):
            done, _ = _completed_model_exists(run)
            print(f"{'skip  ' if done else 'LAUNCH'} [{gpu}] {name}  tags={list(run.wandb_tags)}")
        print(f"{len(runs)} configs")
        return
    launch_training_jobs(runs, gpu=gpu, max_parallel_runs=MAX_PARALLEL_RUNS)
