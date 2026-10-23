"""Problem 4: Inside the Hessian.

Each stage launches its jobs on Modal and returns; re-running a stage launches only the
jobs whose results are not on your volume yet. `fetch` downloads every finished result.

    uv run python -m experiments.a3_optimization.p4_inside_the_hessian a          # part (a)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-rescale  # part (b)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-train
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-measure  # after b-train
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian c          # part (c)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian fetch

Add --local to run measurement stages on this machine's GPU instead. Read results with

    from experiments.a3_optimization.hessian_jobs import load_result
    r = load_result("a-adamw-9375")
    r["lambda_max"], r["top"]["residuals"][0], r["families"]["head"]["lambda_max"]
    r["energy"]["layer"]["blocks"], r["energy"]["layer"]["per_probe"]   # [probe, block, block]
"""

import argparse

from experiments.a3_optimization.hessian_checkpoints import FINAL_STEP, STEPS
from experiments.a3_optimization.hessian_jobs import EXPERIMENT_KEY, fetch, launch
from train import TrainConfig, training_run_name


# Part (a): the shipped AdamW checkpoints (no training). Each checkpoint costs about 13
# H100-minutes with these options; a density adds about 13 more.
CHECKPOINTS = STEPS  # 0, 3, 15, ..., 9375
DENSITY_AT = (0, 153, 1526, 9375)  # initialization, 10M, 100M, 614M tokens

LOOK_JOBS = [
    {"name": f"a-adamw-{step}",
     "measure": {"run": "adamw", "step": step,
                 "extremes": True,  # lambda_max, lambda_min, top-5 eigenvectors by family
                 "families": True,  # top eigenvalue of each family's block
                 "energy": ("family", "layer"),  # ||H_cb||_F^2 between blocks
                 "density_runs": 4 if step in DENSITY_AT else 0}}
    for step in CHECKPOINTS
]

# Part (b): hand rescaling of the shipped final checkpoint (no training). `scale`
# multiplies named parameters by constants before measuring; `tensors` reports the top
# eigenvalue of each named tensor's own block, with its RMS and gradient norm.
FINAL = {"extremes": True, "families": True}

RESCALE_JOBS = [
    {"name": "b-rescale-reference",
     "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL, "tensors": []}},
]

# Function-preserving rescalings of the final checkpoint (probe_loss should not change):
#   logits = W_head (g * x / rms(x)), so W_head -> W_head / c with g -> c g is exact, and the head
#   block's Hessian scales by c^2 (the head holds ~5.6 of lambda_max ~6.0);
#   QK-norm normalizes q and k per head, so q_proj, k_proj -> c * (...) is exact, and their blocks'
#   Hessians scale by 1 / c^2.
HEAD, FINAL_NORM = "lm_head.weight", "model.norm.weight"
QK = [f"model.layers.{i}.self_attn.{w}_proj.weight" for i in range(8) for w in ("q", "k")]
AFFECTED = [HEAD, FINAL_NORM, QK[0], QK[-2]]  # head, final gain, q_proj of the first and last layer
RESCALES = {
    "identity": {},  # "before": the same tensors measured on the unchanged checkpoint
    "head-x0.5": {HEAD: 0.5, FINAL_NORM: 2.0},
    "head-x2": {HEAD: 2.0, FINAL_NORM: 0.5},
    "qk-x0.25": {name: 0.25 for name in QK},
}
RESCALE_JOBS += [
    {"name": f"b-rescale-{label}",
     "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL, "scale": scale, "tensors": AFFECTED}}
    for label, scale in RESCALES.items()
]

# Part (b): training changes. A run takes about 11 minutes on an H200 (a little more on
# an H100); measuring its final checkpoint about 12.
# The same equilibrium moved by training: weight decay pulls lm_head's scale down while the
# undecayed final-norm gain compensates. More head decay -> smaller head, larger gain -> sharper head
# block; none -> flatter. p4_optimizers overrides one tensor's decay; global weight_decay=0 is the
# hyperparameter-only version.
P4_OPTIMIZER = "experiments.a3_optimization.p4_optimizers:build_optimizer"


def head_decay(wd):
    return TrainConfig(optimizer_builder=P4_OPTIMIZER, optimizer_kwargs={"overrides": {HEAD: wd}},
                       run_name_suffix=f"p4b-headwd{wd:g}", wandb_tags=(EXPERIMENT_KEY,))


RUNS = [
    head_decay(0.0),
    head_decay(0.4),
    head_decay(1.0),
    TrainConfig(weight_decay=0.0, run_name_suffix="p4b", wandb_tags=(EXPERIMENT_KEY,)),
]

FINAL_JOBS = [
    {"name": f"b-final-{training_run_name(config)}",
     "measure": {"run": training_run_name(config), "step": FINAL_STEP, **FINAL}}
    for config in RUNS
]

# Part (c): continue the shipped SGDM run from 100M tokens with the full update, the
# update projected onto the top-k eigenvectors, and the update with them removed. An arm
# costs 25-60 H100-minutes, mostly in its 12 basis refreshes.
START = {"run": "sgdm", "step": 1526, "k": 10, "steps": 300, "refresh": 25,
         "reset_momentum": True}

SUBSPACE_JOBS = [
    {"name": f"c-{arm}", "continue": {**START, "transform": arm}}
    for arm in ("full", "top", "removed")
]
# Variations. The top-10 arm diverged while "removed" matched "full": is the top subspace simply
# above the heavy-ball stability threshold 2(1 + beta) / eta? Quarter the LR (with its own full
# control); shrink the subspace to k = 1; and repeat all three arms early in training (20M tokens).
SUBSPACE_JOBS += [
    {"name": "c-top-lr0.25", "continue": {**START, "transform": "top", "lr_scale": 0.25}},
    {"name": "c-full-lr0.25", "continue": {**START, "transform": "full", "lr_scale": 0.25}},
    {"name": "c-top-k1", "continue": {**START, "transform": "top", "k": 1}},
] + [
    {"name": f"c-{arm}-step305", "continue": {**START, "transform": arm, "step": 305}}
    for arm in ("full", "top", "removed")
]


# Refresh interval: is the top-10 arm's divergence caused by a stale basis? Refresh 5x as often
# (60 eigensolves, ~2.3 H100-hours per arm) and 4x less often (3 eigensolves). The full arm does not
# use the basis, so c-full stays the control.
REFRESH_JOBS = [
    {"name": "c-top-refresh5", "continue": {**START, "transform": "top", "refresh": 5}},
    {"name": "c-removed-refresh5", "continue": {**START, "transform": "removed", "refresh": 5}},
    {"name": "c-top-refresh100", "continue": {**START, "transform": "top", "refresh": 100}},
]
SUBSPACE_JOBS += REFRESH_JOBS


# TODO: Part (a): anything else you need, e.g. the baseline for the block-structure
# question. H.hvp (hessian.ProbeHessian) applies H to any vector you construct.
# TODO: Part (b): one rescaling job per transformation you test, e.g.
# RESCALE_JOBS.append({"name": "b-rescale-<label>",
#     "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL,
#                 "scale": {"<parameter name>": 0.25, ...}, "tensors": ["<parameter name>", ...]}})
# and the training changes in RUNS.
# TODO: Part (c): vary what you think decides the outcome, e.g.
# SUBSPACE_JOBS.append({"name": "c-removed-<label>", "continue": {**START, "transform": "removed", ...}})
# Options: k, refresh, steps, lr_scale (multiplies the run's schedule), reset_momentum,
# and the starting step (any of STEPS; the run has optimizer state at each).

STAGES = {"a": LOOK_JOBS, "b-rescale": RESCALE_JOBS, "b-measure": FINAL_JOBS, "c": SUBSPACE_JOBS,
          "c-refresh": REFRESH_JOBS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=(*STAGES, "b-train", "fetch"))
    parser.add_argument("--local", action="store_true", help="run on this machine's GPU")
    args = parser.parse_args()
    if args.stage == "b-train":
        from modal_train import launch_training_jobs

        launch_training_jobs(RUNS)
    elif args.stage == "fetch":
        print(fetch([j["name"] for jobs in STAGES.values() for j in jobs]))
    else:
        launch(STAGES[args.stage], local=args.local)


if __name__ == "__main__":
    main()
