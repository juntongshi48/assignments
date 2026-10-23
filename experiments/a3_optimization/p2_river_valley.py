"""Problem 2: schedules, parameter averaging, rewarming, and horizon-free schedules.

    uv run python -m experiments.a3_optimization.p2_river_valley STAGE

Stages: schedules, lr_sweep, lr_low, rewarm_controls, rewarm (after rewarm_controls finish), horizon_free.
"""

import argparse
from dataclasses import replace

from modal_train import launch_training_jobs
from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-river-valley"
# Each run logs val_ema{decay}_loss next to val_loss at every evaluation.
# Timescales 1/(1 - decay) of ~10, 20, 100, 200, 1000, and 2000 updates (of 9375).
EMA_DECAYS = (0.9, 0.95, 0.99, 0.995, 0.999, 0.9995)

# Parts (a) and (b): one run per schedule, with evaluation-only EMAs.
SCHEDULE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, ema_decays=EMA_DECAYS, wandb_tags=(EXPERIMENT_KEY,))
    for lr_schedule in (
        "wsd0.0",  # constant after warmup
        "linear",
        "cos",
        "wsd0.2",  # plateau, then decay over the last 20% of training
    )
]

# Part (a) robustness: the same four schedules (and EMAs) at half and double the peak LR.
LR_ALTERNATIVES = (1.5e-3, 6e-3)
LR_SWEEP_RUNS = [replace(run, learning_rate=lr) for lr in LR_ALTERNATIVES for run in SCHEDULE_RUNS]
# Constant and WSD still improve as the LR drops to 1.5e-3, so their optimum may be lower.
LOW_LR_RUNS = [replace(run, learning_rate=7.5e-4) for run in SCHEDULE_RUNS if run.lr_schedule in ("wsd0.0", "wsd0.2")]

# Part (c): temporary LR dips. Each control keeps its state at DIP_START, and every dip run forks
# from it there, so a dip run and its control are identical before the dip and share the data
# order and the final cooldown afterwards. lr_dip = (start, bottom, end, scale) in fractions of
# training: linear decay to scale x LR by `bottom`, linear rewarm to full LR by `end`.
TOTAL_STEPS = TrainConfig.num_train_sequences // TrainConfig.batch_size  # 9375
DIP_START = 0.4
DIP_STEP = round(DIP_START * TOTAL_STEPS)  # 3750
DIP_DEPTHS = (0.1, 0.5)  # LR multiplier at the bottom of the dip
DIP_WINDOWS = {"short": 0.025, "medium": 0.1, "long": 0.2}  # start -> end; bottom at the midpoint


def dip(scale, window, start=DIP_START):
    return (start, start + window / 2, start + window, scale)


REWARM_CONTROLS = {
    schedule: TrainConfig(lr_schedule=schedule, keep_checkpoint_steps=(DIP_STEP,), run_name_suffix="p2c-control",
                          wandb_tags=(EXPERIMENT_KEY,))
    for schedule in ("wsd0.2", "linear")
}


def rewarm_run(schedule, scale, window):
    control = REWARM_CONTROLS[schedule]
    return replace(control, lr_dip=dip(scale, window), keep_checkpoint_steps=(), run_name_suffix=None,
                   fork_from_run=training_run_name(control), fork_from_step=DIP_STEP)


# Depth x duration on wsd0.2 (constant LR at the dip); two dips on linear, where the LR at the dip
# is already 0.6x peak, to ask whether the regret depends on the schedule.
REWARM_RUNS = [rewarm_run("wsd0.2", scale, window) for scale in DIP_DEPTHS for window in DIP_WINDOWS.values()]
REWARM_RUNS += [rewarm_run("linear", 0.1, DIP_WINDOWS[name]) for name in ("short", "medium")]

# Part (d): horizon-free schedules, run to the full budget and read off at BUDGET_FRACTIONS of it.
# warmup_steps fixes the warmup in updates, so it does not depend on the budget; sgdrT restarts
# every T updates (cycles end at 94 + 2000k, so the smaller budgets end mid-cycle).
BUDGET_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
HORIZON_FREE_RUNS = [
    TrainConfig(lr_schedule="invsqrt", warmup_steps=94, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(lr_schedule="sgdr2000", warmup_steps=94, min_lr_ratio=0.1,
                wandb_tags=(EXPERIMENT_KEY,)),
]
# Budget-tuned baselines: linear decay to zero over each smaller budget, at two peak LRs (the best
# per budget is the reference). The full-budget linear runs are in SCHEDULE_RUNS / LR_SWEEP_RUNS.
# warmup_steps=94 matches the full-budget runs (1% of 9,375); the default 1% of a smaller budget
# (12-47 updates) is too short at peak LR 3e-3 / 6e-3 and handicaps the baseline.
BUDGET_TUNED_LRS = (3e-3, 6e-3)
BUDGET_TUNED_RUNS = [
    TrainConfig(lr_schedule="linear", learning_rate=lr, warmup_steps=94,
                num_train_sequences=round(fraction * TrainConfig.num_train_sequences),
                wandb_tags=(EXPERIMENT_KEY,))
    for fraction in BUDGET_FRACTIONS if fraction < 1 for lr in BUDGET_TUNED_LRS
]

STAGES = dict(schedules=SCHEDULE_RUNS, lr_sweep=LR_SWEEP_RUNS, lr_low=LOW_LR_RUNS,
              rewarm_controls=list(REWARM_CONTROLS.values()), rewarm=REWARM_RUNS,
              horizon_free=HORIZON_FREE_RUNS + BUDGET_TUNED_RUNS, budget_tuned=BUDGET_TUNED_RUNS)
RUNS = list({training_run_name(run): run for runs in STAGES.values() for run in runs}.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    args = parser.parse_args()
    if args.stage == "rewarm":
        from experiments.a3_optimization.p1d_cooldown import check_sources

        check_sources(REWARM_RUNS)
    launch_training_jobs(STAGES[args.stage])


if __name__ == "__main__":
    main()
