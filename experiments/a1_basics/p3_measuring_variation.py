from experiments.a1_basics.common import DEFAULT_METRIC_LOGGERS, main_for_parts
from experiments.a1_basics.perturbation import FINGERPRINT_LOGGER
from model_config import depth_model_config
from train import TrainConfig

SEEDS = (1, 2, 3)  # seed 42 comes from existing runs in every group
DETERMINISTIC_LOGGERS = DEFAULT_METRIC_LOGGERS + (FINGERPRINT_LOGGER,)

# Part (a): all three sources at once (paired seeds, nondeterministic kernels).
P3A_KEY = "p3a-v1"


def build_part_a_runs():
    return [
        TrainConfig(
            model_seed=seed,
            data_seed=seed,
            run_name_suffix=P3A_KEY,
            wandb_tags=(P3A_KEY, "all-sources"),
        )
        for seed in SEEDS
    ]


# Part (b): one source at a time. Seed-only runs are deterministic so hardware noise is
# off. The two deterministic references are the handout's negative control and the
# Problem 4 baseline, so they also log the parameter fingerprint.
P3B_KEY = "p3b-v1"
A100_GPU = "A100-80GB"


def build_part_b_runs():
    runs = [
        TrainConfig(
            model_seed=seed,
            deterministic=True,
            run_name_suffix=P3B_KEY,
            wandb_tags=(P3B_KEY, "init-only"),
        )
        for seed in SEEDS
    ]
    runs += [
        TrainConfig(
            data_seed=seed,
            deterministic=True,
            run_name_suffix=P3B_KEY,
            wandb_tags=(P3B_KEY, "data-order-only"),
        )
        for seed in SEEDS
    ]
    runs += [
        TrainConfig(
            deterministic=True,
            metric_loggers=DETERMINISTIC_LOGGERS,
            run_name_suffix=f"deterministic-reference-{index}",
            wandb_tags=(P3B_KEY, "deterministic-reference"),
        )
        for index in (1, 2)
    ]
    return runs


# Hardware nondeterminism runs launch as separate parts (separate Modal apps):
# configure_deterministic_training sets process-wide torch flags and never resets them,
# and Modal reuses containers within an app, so mixing deterministic and nondeterministic
# runs in one app could leak determinism into the replicates (or vice versa).
def build_part_b_h100_runs():
    # The Problem 1 baseline and the original smoke run are already two identical-config
    # H100 replicates; this adds a third.
    return [
        TrainConfig(
            run_name_suffix=f"{P3B_KEY}-h100-replicate3",
            wandb_tags=(P3B_KEY, "hardware-nondeterminism", "H100"),
        )
    ]


def build_part_b_a100_runs():
    return [
        TrainConfig(
            run_name_suffix=f"{P3B_KEY}-a100-replicate{index}",
            wandb_tags=(P3B_KEY, "hardware-nondeterminism", "A100"),
        )
        for index in (1, 2)
    ]


# Part (c): does variability change with hyperparameters or model capability?
# Seed 42 for each group: lr 0.01 and bs16 from p1a-v1, d9 from p2a-v1.
P3C_KEY = "p3c-v1"
P3C_GROUPS = {
    "lr0.01": dict(learning_rate=0.01),
    "bs16": dict(batch_size=16),
    "d9": dict(model_config=depth_model_config(9)),
}


def build_part_c_runs():
    return [
        TrainConfig(
            **overrides,
            model_seed=seed,
            data_seed=seed,
            run_name_suffix=P3C_KEY,
            wandb_tags=(P3C_KEY, f"group-{group}"),
        )
        for group, overrides in P3C_GROUPS.items()
        for seed in SEEDS
    ]


PARTS = {
    "a": build_part_a_runs,
    "b": build_part_b_runs,
    "b-h100": build_part_b_h100_runs,
    "b-a100": (build_part_b_a100_runs, A100_GPU),
    "c": build_part_c_runs,
}


if __name__ == "__main__":
    main_for_parts(PARTS)
