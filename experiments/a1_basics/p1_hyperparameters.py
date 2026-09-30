import sys

from modal_train import launch_training_jobs
from train import TrainConfig


# Course accounts get 2 concurrent GPUs; extra calls queue inside the Modal app.
MAX_PARALLEL_RUNS = 2

# Part (a): one-at-a-time sweeps around the d8 default recipe. Every run keeps
# model=d8 and num_train_sequences=600k, so params x tokens is fixed.
PART_A_KEY = "p1a-v1"
PART_A_SWEEPS = {
    "learning_rate": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    # bs256 dropped: it OOMed, and bs128 is already well past the optimum.
    "batch_size": (16, 32, 64, 128),
    # 0.3 added because 0.1 was the best point, at the edge of the original grid.
    "warmup_percent": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
    "weight_decay": (0.01, 0.03, 0.1, 0.3, 1.0),
}


def build_part_a_runs():
    defaults = TrainConfig()
    runs = [TrainConfig(run_name_suffix=PART_A_KEY, wandb_tags=(PART_A_KEY, "baseline"))]
    for name, values in PART_A_SWEEPS.items():
        for value in values:
            if value == getattr(defaults, name):
                continue  # shared baseline run above
            runs.append(
                TrainConfig(
                    **{name: value},
                    run_name_suffix=PART_A_KEY,
                    wandb_tags=(PART_A_KEY, f"sweep-{name}"),
                )
            )
    return runs


# Part (b): grids of LR against each other hyperparameter (3x4 for batch size, which
# gained bs16 once bs32 turned out best, else 3x3). Every grid contains
# the defaults, so its default row/column are part (a) runs; only the 4 corners
# (both values off-default) are launched here.
PART_B_KEY = "p1b-v1"
PART_B_GRIDS = {
    ("learning_rate", "batch_size"): ((1e-3, 3e-3, 1e-2), (16, 32, 64, 128)),
    ("learning_rate", "warmup_percent"): ((1e-3, 3e-3, 1e-2), (0.01, 0.1, 0.3)),
    # Anti-diagonals share the same lr*wd product.
    ("learning_rate", "weight_decay"): ((1e-3, 3e-3, 1e-2), (0.1, 0.3, 1.0)),
}


def build_part_b_runs():
    defaults = TrainConfig()
    runs = []
    for (x_name, y_name), (x_values, y_values) in PART_B_GRIDS.items():
        assert getattr(defaults, x_name) in x_values and getattr(defaults, y_name) in y_values
        for x in x_values:
            for y in y_values:
                if x == getattr(defaults, x_name) or y == getattr(defaults, y_name):
                    continue  # covered by part (a)
                runs.append(
                    TrainConfig(
                        **{x_name: x, y_name: y},
                        run_name_suffix=PART_B_KEY,
                        wandb_tags=(PART_B_KEY, f"grid-{x_name}-{y_name}"),
                    )
                )
    return runs


# Part (c): LR schedule x warmup. The built-in "constant" schedule ignores
# warmup_percent, so constant-with-warmup is "wsd0.0" (warmup, then stable with no
# decay); at warmup 0 it is identical to "constant".
PART_C_KEY = "p1c-v1"
PART_C_SCHEDULES = ("linear", "cos", "wsd0.0", "wsd0.2")
PART_C_WARMUPS = (0.0, 0.1, 0.3)


def build_part_c_runs():
    defaults = TrainConfig()
    runs = []
    for lr_schedule in PART_C_SCHEDULES:
        for warmup_percent in PART_C_WARMUPS:
            if lr_schedule == defaults.lr_schedule and warmup_percent in PART_A_SWEEPS["warmup_percent"]:
                continue  # covered by part (a)
            runs.append(
                TrainConfig(
                    lr_schedule=lr_schedule,
                    warmup_percent=warmup_percent,
                    run_name_suffix=PART_C_KEY,
                    wandb_tags=(PART_C_KEY, "grid-lr_schedule-warmup_percent"),
                )
            )
    return runs


# Part (c) follow-ups, all tagged PART_C_KEY. Missing grid cells come from earlier
# parts: linear x LR at warmup 0.1 from part (b), schedules at LR 3e-3 from part (c),
# clipped runs from part (a).
PART_C2_WARMUP = 0.1


def build_part_c2_runs():
    runs = []
    # Does the best peak LR depend on the schedule?
    for lr_schedule in ("cos", "wsd0.2"):
        for learning_rate in (1e-3, 1e-2):
            runs.append(
                TrainConfig(
                    lr_schedule=lr_schedule,
                    learning_rate=learning_rate,
                    warmup_percent=PART_C2_WARMUP,
                    run_name_suffix=PART_C_KEY,
                    wandb_tags=(PART_C_KEY, "grid-lr_schedule-learning_rate"),
                )
            )
    # WSD decay length, alongside wsd0.0 and wsd0.2 from part (c).
    for lr_schedule in ("wsd0.1", "wsd0.5"):
        runs.append(
            TrainConfig(
                lr_schedule=lr_schedule,
                warmup_percent=PART_C2_WARMUP,
                run_name_suffix=PART_C_KEY,
                wandb_tags=(PART_C_KEY, "sweep-wsd_decay_fraction"),
            )
        )
    # Is gradient clipping what keeps high-LR runs stable?
    for learning_rate in (3e-3, 1e-2):
        runs.append(
            TrainConfig(
                learning_rate=learning_rate,
                grad_norm=None,
                run_name_suffix=PART_C_KEY,
                wandb_tags=(PART_C_KEY, "grid-grad_norm-learning_rate"),
            )
        )
    return runs


PARTS = {
    "a": build_part_a_runs,
    "b": build_part_b_runs,
    "c": build_part_c_runs,
    "c2": build_part_c2_runs,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PARTS:
        raise SystemExit(f"usage: python -m experiments.a1_basics.p1_hyperparameters {{{','.join(PARTS)}}}")
    launch_training_jobs(PARTS[sys.argv[1]](), max_parallel_runs=MAX_PARALLEL_RUNS)


if __name__ == "__main__":
    main()
