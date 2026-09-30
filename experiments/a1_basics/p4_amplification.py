from experiments.a1_basics.common import main_for_parts
from experiments.a1_basics.p3_measuring_variation import DETERMINISTIC_LOGGERS
from experiments.a1_basics.perturbation import NOISE_SETTINGS, noise_logger
from train import TrainConfig

# All runs are deterministic and compare against Problem 3's deterministic references
# (p3b-v1 "deterministic-reference-1/2"), which log the same parameter fingerprint.
P4_KEY = "p4-v1"


def build_part_a_runs():
    # Token 100 of the first shuffled sequence set to 17.
    return [
        TrainConfig(
            deterministic=True,
            perturb_one_token=True,
            metric_loggers=DETERMINISTIC_LOGGERS,
            run_name_suffix=P4_KEY,
            wandb_tags=(P4_KEY, "data-perturbation"),
        )
    ]


def build_part_b_runs():
    # Parameter noise of relative size `scale` after `after_steps` optimizer steps.
    return [
        TrainConfig(
            deterministic=True,
            metric_loggers=DETERMINISTIC_LOGGERS + (noise_logger(after_steps, scale),),
            run_name_suffix=f"{P4_KEY}-noise{scale:.0e}-after{after_steps}",
            wandb_tags=(P4_KEY, "param-noise"),
        )
        for after_steps, scale in NOISE_SETTINGS
    ]


PARTS = {
    "a": build_part_a_runs,
    "b": build_part_b_runs,
}


if __name__ == "__main__":
    main_for_parts(PARTS)
