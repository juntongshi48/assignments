from experiments.a1_basics.common import main_for_parts
from model_config import depth_model_config
from train import TrainConfig

DEPTHS = range(4, 10)

# Part (a): the handout's four variants across d4-d9 at the fixed 600k-sequence budget.
# Three d8 points are reused from Problem 1 instead of rerun: baseline (p1a-v1),
# lr 0.03 (p1a-v1), and constant, which ignores warmup and so matches p1c-v1's
# wsd0.0 run at warmup 0 step for step.
P2A_KEY = "p2a-v1"
P2A_VARIANTS = {
    "baseline": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.0),
    "constant": dict(learning_rate=0.003, lr_schedule="constant", dropout=0.0),
    "dropout0.2": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.2),
    "lr0.03": dict(learning_rate=0.03, lr_schedule="linear", dropout=0.0),
}
P2A_REUSED_D8 = {"baseline", "constant", "lr0.03"}


def build_part_a_runs():
    return [
        TrainConfig(
            model_config=depth_model_config(depth),
            **overrides,
            run_name_suffix=P2A_KEY,
            wandb_tags=(P2A_KEY, f"variant-{variant}"),
        )
        for depth in DEPTHS
        for variant, overrides in P2A_VARIANTS.items()
        if not (depth == 8 and variant in P2A_REUSED_D8)
    ]


# Part (b): interventions expected to change the slope but keep a power law.
# Model axis: LR scaled as 1/width (0.024 / depth, so d8 is the 0.003 baseline).
# Data axis at d6: bs64 vs bs128 over token budgets (bs128 has half the optimizer steps).
P2B_KEY = "p2b-v1"
P2B_WIDTH_SCALED_DEPTHS = (4, 5, 6, 7, 9)
P2B_DATA_DEPTH = 6
P2B_BS64_SEQUENCES = (150_000, 300_000, 1_200_000)  # 600k is the part (a) d6 baseline
P2B_BS128_SEQUENCES = (300_000, 600_000, 1_200_000)


def width_scaled_lr(depth):
    return float(f"{0.024 / depth:.4g}")


def build_part_b_runs():
    runs = [
        TrainConfig(
            model_config=depth_model_config(depth),
            learning_rate=width_scaled_lr(depth),
            run_name_suffix=P2B_KEY,
            wandb_tags=(P2B_KEY, "lr-width-scaled"),
        )
        for depth in P2B_WIDTH_SCALED_DEPTHS
    ]
    for batch_size, sequence_budgets in ((64, P2B_BS64_SEQUENCES), (128, P2B_BS128_SEQUENCES)):
        runs += [
            TrainConfig(
                model_config=depth_model_config(P2B_DATA_DEPTH),
                batch_size=batch_size,
                num_train_sequences=num_train_sequences,
                run_name_suffix=P2B_KEY,
                wandb_tags=(P2B_KEY, f"data-scaling-bs{batch_size}"),
            )
            for num_train_sequences in sequence_budgets
        ]
    return runs


# Part (c): screen single-change interventions for power-law breakage at d4/d6/d9, then
# fill in d5/d7/d8 for the ones that bend (add their names to P2C_EXPAND).
P2C_KEY = "p2c-v1"
P2C_SCREEN_DEPTHS = (4, 6, 9)
P2C_EXPAND_DEPTHS = (5, 7, 8)
P2C_INTERVENTIONS = {
    # ~60k unique sequences repeated 10x: same total tokens, 10x less unique data.
    # 60,032 = 938 full batches of 64, avoiding a partial batch every epoch.
    "repeat10x": dict(num_train_sequences=60_032, num_epochs=10.0),
    "no-qk-norm": dict(qk_norm=False),
    "no-warmup": dict(warmup_percent=0.0),
    # One change relative to part (a)'s lr0.03 curve.
    "lr0.03-no-clip": dict(learning_rate=0.03, grad_norm=None),
}
# Before expanding, drop d8 where it already exists: no-warmup d8 is p1c-v1's linear
# warmup-0 run, and no-qk-norm d8 is p6-v1's run.
P2C_EXPAND = ()


def build_intervention_runs(depths, names):
    return [
        TrainConfig(
            model_config=depth_model_config(depth),
            **P2C_INTERVENTIONS[name],
            run_name_suffix=P2C_KEY,
            wandb_tags=(P2C_KEY, f"intervention-{name}"),
        )
        for name in names
        for depth in depths
    ]


def build_part_c_runs():
    return build_intervention_runs(P2C_SCREEN_DEPTHS, P2C_INTERVENTIONS)


def build_part_c_expand_runs():
    if not P2C_EXPAND:
        raise SystemExit("Set P2C_EXPAND to the interventions to expand after the screen.")
    return build_intervention_runs(P2C_EXPAND_DEPTHS, P2C_EXPAND)


PARTS = {
    "a": build_part_a_runs,
    "b": build_part_b_runs,
    "c": build_part_c_runs,
    "c-expand": build_part_c_expand_runs,
}


if __name__ == "__main__":
    main_for_parts(PARTS)
