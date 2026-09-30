from experiments.a1_basics.common import main_for_parts
from train import TrainConfig

# Parameter / activation / gradient RMS are logged every 100 steps by default, so most
# of Problem 6 reuses Problem 1 (LR, weight decay, warmup, batch size, schedules,
# clipping x LR). New runs fill gaps in that intervention dictionary.
# Note: logged gradient RMS is captured before clipping (train.py AFTER_BACKWARD), so a
# tighter clip threshold shows up only through its downstream effects.
P6_KEY = "p6-v1"


def build_part_c_runs():
    return [
        TrainConfig(**overrides, run_name_suffix=P6_KEY, wandb_tags=(P6_KEY, f"intervention-{name}"))
        for name, overrides in {
            "no-qk-norm": dict(qk_norm=False),
            "no-weight-decay": dict(weight_decay=0.0),
            "clip0.1": dict(grad_norm=0.1),
        }.items()
    ]


PARTS = {
    "c": build_part_c_runs,
}


if __name__ == "__main__":
    main_for_parts(PARTS)
