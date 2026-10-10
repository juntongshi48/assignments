"""P2(a) follow-up: weight-norm trajectories at a better WD per budget.

Repeats the A1 d6 budget series (LR .003, batch 64, A1 defaults including per-module RMS logging)
with WD set to the P2a paraboloid's best WD at LR .003 for each budget, so each run pairs with an
existing WD = .1 run in plot_p2a_norms.py. The WDs come from the d8 P2a fits; d6's optimum may differ.
"""
import argparse

from model_config import depth_model_config
from train import TrainConfig, checked_train_config, training_run_name

EXPERIMENT_KEY = "a2-p2a-normwd"
# num_train_sequences -> best WD at LR .003 from the P2a paraboloid fits (0.595, 0.303, 0.210, 0.146).
BETTER_WD = {150_000: .6, 300_000: .3, 600_000: .21, 1_200_000: .15}


def runs():
    return [TrainConfig(model_config=depth_model_config(6), learning_rate=.003, weight_decay=wd,
                        num_train_sequences=sequences, run_name_suffix=EXPERIMENT_KEY,
                        wandb_tags=(EXPERIMENT_KEY,))
            for sequences, wd in BETTER_WD.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Submit the runs to Modal.")
    args = parser.parse_args(argv)
    configs = runs()
    for c in configs:
        print(f"{training_run_name(checked_train_config(c))}  wd={c.weight_decay:g}  "
              f"loggers={[l.fn if isinstance(l.fn, str) else type(l.fn).__name__ for l in c.metric_loggers]}")
    if args.execute:
        from modal_train import launch_training_jobs
        launch_training_jobs(configs, max_parallel_runs=2)
    else:
        print("Preview only. Add --execute to train these configurations.")


if __name__ == "__main__":
    main()
