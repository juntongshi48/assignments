"""Problem 3: sharpness and the edge of stability.

    uv run python -m experiments.a3_optimization.p3_edge_of_stability [a|b|c|all]

Both helpers attach sharpness_logging.SharpnessLogger, which logs to W&B under
logging/sharpness/*. Read the results back with experiments.a3_optimization.sharpness.
"""

import argparse

from metric_logging import AFTER_BACKWARD, MetricLogger
from modal_train import launch_training_jobs
from sharpness_logging import SharpnessLogger
from train import TrainConfig


EXPERIMENT_KEY = "a3-edge-of-stability"
FRACTION = 0.32  # of an epoch: 3,000 updates, 197M tokens at batch size 64


def full_batch(num_sequences, updates, every=25, **changes):
    """Full-batch gradient descent on the first `num_sequences` training sequences.

    Every update uses all of them, so one epoch is one update. lambda_max is
    measured on the same sequences every `every` updates. Keyword arguments
    override any setting, e.g. learning_rate=0.02 or optimizer_name="adamw".
    """
    optimizer_name = changes.get("optimizer_name", "sgd")
    settings = dict(
        num_train_sequences=num_sequences,
        batch_size=num_sequences,
        num_epochs=float(updates),
        optimizer_name=optimizer_name,
        # plain gradient descent for sgd (its default beta1 is momentum); AdamW keeps 0.9
        beta1=0.0 if optimizer_name == "sgd" else TrainConfig.beta1,
        lr_schedule="constant",
        weight_decay=0.0,
        grad_norm=None,
        embedding_init_scale=16.0,
        num_evals=10,
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=every)),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


def measurement_steps(batch_size):
    """Every 16 updates to update 208, then every 100 to 1,500, then every 300, at
    batch size 64; other batch sizes measure at the same token counts."""
    at_64 = [*range(0, 209, 16), *range(300, 1501, 100), *range(1800, 3001, 300)]
    total = int(FRACTION * (600_000 // batch_size))
    return sorted({round(u * 64 / batch_size) for u in at_64} & set(range(total)))


def minibatch(batch_size=64, kinds=("pre",), **changes):
    """The course recipe with a constant learning rate after warmup, cut at 0.32
    epochs, measuring the preconditioned sharpness on 64 held-out sequences.

    Warmup is 3.125% of the run: 93 updates at batch size 64, and the same
    6M tokens at every batch size.
    """
    settings = dict(
        num_epochs=FRACTION,
        warmup_percent=0.03125,
        lr_schedule="wsd0.0",
        batch_size=batch_size,
        num_micro_batches=max(1, batch_size // 64),
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(
            steps=measurement_steps(batch_size), probe="heldout", kinds=kinds)),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


# Part (a): full-batch training.
FULL_BATCH_RUNS = [
    full_batch(32, 1500, learning_rate=0.01),  # (a) i: the clean case
    # (a) ii: halve / double the learning rate at update 750 (step schedules have no warmup)
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x0.5"),
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x2"),
]

# Part (b): minibatch training.
MINIBATCH_RUNS = [
    minibatch(),  # (b): the base recipe
]

RUNS = FULL_BATCH_RUNS + MINIBATCH_RUNS


# Part (c): what decides whether training enters and stays at the edge? One knob per run, from
# the part (b) recipe (batch 64, LR 3e-3, beta1 0.9, beta2 0.95, WD 0.1, 3.125% warmup). The
# threshold for eta * lambda_pre is 2 (1 + beta1) / (1 - beta1), so the beta1 runs move it.
# kinds=("raw", "pre") also logs the unpreconditioned lambda_max, to tell a change in the Hessian
# from a change in AdamW's preconditioner (one more eigensolve per measurement).
RAW_PRE = ("raw", "pre")
PART_C_RUNS = [
    # Full-batch AdamW: is minibatch noise what keeps (b) below 38?
    full_batch(32, 1500, optimizer_name="adamw", learning_rate=3e-4),
    full_batch(32, 1500, optimizer_name="adamw", learning_rate=1e-3),
    # Base recipe again, logging raw and preconditioned sharpness (baseline for beta2).
    minibatch(kinds=RAW_PRE, run_name_suffix="rawpre"),
    # Gradient noise per update.
    *(minibatch(batch_size=b) for b in (16, 128, 512)),
    # Step size.
    *(minibatch(learning_rate=lr) for lr in (1e-3, 9e-3)),
    # Momentum: moves the threshold to 2 (beta1 = 0) and 78 (beta1 = 0.95).
    *(minibatch(beta1=b1) for b1 in (0.0, 0.95)),
    # Preconditioner timescale 1 / (1 - beta2): ~10, 100, 1000 updates (default 0.95: ~20).
    *(minibatch(beta2=b2, kinds=RAW_PRE) for b2 in (0.9, 0.99, 0.999)),
    # Parameter scale.
    *(minibatch(weight_decay=wd) for wd in (0.0, 0.3)),
    # Warmup: the only time (b) reached the edge was just after it.
    *(minibatch(warmup_percent=w) for w in (0.0, 0.1)),
]


# Full-batch AdamW at 3e-4 / 1e-3 memorizes the 32 sequences within ~150-300 updates (loss floor
# ln(32)/1024 = 0.00338), so most of those runs measure AdamW at the floor. Lower LRs keep it training.
PART_C_LOW_LR_RUNS = [full_batch(32, 1500, optimizer_name="adamw", learning_rate=lr) for lr in (3e-5, 1e-4)]

STAGES = dict(a=FULL_BATCH_RUNS, b=MINIBATCH_RUNS, c=PART_C_RUNS, c_low_lr=PART_C_LOW_LR_RUNS,
              all=RUNS + PART_C_RUNS + PART_C_LOW_LR_RUNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=tuple(STAGES))
    launch_training_jobs(STAGES[parser.parse_args().stage])


if __name__ == "__main__":
    main()
