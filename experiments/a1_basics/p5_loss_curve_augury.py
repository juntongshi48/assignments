from experiments.a1_basics.common import main_for_parts
from train import TrainConfig

# Loss-curve shapes mostly come from Problem 1's LR / batch size / schedule sweeps.
# New runs fill the momentum gap, add a beta2 smoothness probe, and add bs256 (the one
# config in the handout's example question that Problem 1 lacks; it OOMs with
# torch.compile at full size, so it is split into two micro-batches of 128).
P5_KEY = "p5-v1"


def build_part_a_runs():
    runs = [
        TrainConfig(beta1=beta1, run_name_suffix=P5_KEY, wandb_tags=(P5_KEY, "momentum"))
        for beta1 in (0.0, 0.5, 0.95)
    ]
    runs.append(TrainConfig(beta2=0.99, run_name_suffix=P5_KEY, wandb_tags=(P5_KEY, "beta2")))
    runs.append(
        TrainConfig(
            batch_size=256,
            num_micro_batches=2,
            run_name_suffix=P5_KEY,
            wandb_tags=(P5_KEY, "batch-size"),
        )
    )
    return runs


PARTS = {
    "a": build_part_a_runs,
}


if __name__ == "__main__":
    main_for_parts(PARTS)
