"""Optimizer builder for Problem 4(b): the course AdamW with per-parameter weight decay overrides.

Weight decay is the force that sets a parameter's scale, and for parameters followed by an
RMSNorm (or, for lm_head, preceded by the final norm's gain) the scale is free to move without
changing the function. Overriding the decay of one such tensor shifts that equilibrium, and with
it the curvature of its block, by mostly reparametrizing the same model.

    TrainConfig(optimizer_builder="experiments.a3_optimization.p4_optimizers:build_optimizer",
                optimizer_kwargs={"overrides": {"lm_head.weight": 1.0}}, run_name_suffix=...)
"""

import torch

from optimizers import ADAMW_EPSILON, build_masked_weight_decay_parameter_groups


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2, overrides=None):
    """Same groups as optimizers.build_optimizer, except each name in `overrides` gets its own
    weight decay (also allowed for parameters the default leaves undecayed, such as norm gains)."""
    if optimizer_name != "adamw":
        raise ValueError("p4_optimizers.build_optimizer supports adamw only.")
    overrides = dict(overrides or {})
    named = dict(model.named_parameters())
    missing = set(overrides) - set(named)
    if missing:
        raise KeyError(f"No parameters named {sorted(missing)}.")
    special = {id(named[name]) for name in overrides}

    groups = []
    for group in build_masked_weight_decay_parameter_groups(model, weight_decay):
        rest = [p for p in group["params"] if id(p) not in special]
        if rest:
            groups.append({**group, "params": rest})
    groups += [{"params": [named[name]], "weight_decay": wd} for name, wd in overrides.items()]

    fused = any(p.is_cuda for group in groups for p in group["params"])
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=0.0, betas=(beta1, beta2),
                             eps=ADAMW_EPSILON, **({"fused": True} if fused else {}))
