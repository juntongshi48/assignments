"""Hyperball (P1d): an Adam direction rescaled to each linear matrix's norm, then projected back
onto that norm. Embeddings and normalization gains use ordinary Adam at a fixed LR ratio.

Use through TrainConfig:
    optimizer_builder='experiments.a2.hyperball:build_hyperball', weight_decay=0.0
The scheduler scales each group's base LR by the same warmup/decay multiplier, so the
Hyperball:Adam LR ratio stays fixed throughout training.
"""

import torch
from torch import nn

from optimizers import ADAMW_EPSILON

# Adam-group LR relative to the listed Hyperball LR, matching the supplied P1d runs.
ADAM_LR_RATIO = 0.000656 / 0.00630


class Hyperball(torch.optim.Optimizer):
    """Adam moments for every parameter; groups with hyperball=True use the norm-preserving update

        W~ = W - lr * (||W||_F / ||U||_F) * U,    W <- (||W||_F / ||W~||_F) * W~,

    and the other groups take a plain Adam step W <- W - lr * U. No weight decay in either.
    """

    def __init__(self, params, lr, betas=(.9, .95), eps=ADAMW_EPSILON):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, hyperball=False))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["exp_avg"], state["exp_avg_sq"]
                m.lerp_(p.grad, 1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                direction = (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt_().add_(eps))
                if not group["hyperball"]:
                    p.sub_(direction, alpha=lr)
                    continue
                w_norm, u_norm = p.norm(), direction.norm()
                if w_norm == 0 or u_norm == 0:
                    continue
                proposal = p - (lr * w_norm / u_norm) * direction
                proposal_norm = proposal.norm()
                if proposal_norm == 0:
                    continue
                p.copy_(proposal.mul_(w_norm / proposal_norm))
        return loss


def build_hyperball(model, optimizer_name, learning_rate, weight_decay, beta1, beta2,
                    adam_lr_ratio=ADAM_LR_RATIO):
    """TrainConfig.optimizer_builder entry point; optimizer_name is ignored."""
    if weight_decay != 0:
        raise ValueError("Hyperball controls weight norms itself; set weight_decay=0.0")
    matrices, others, seen = [], [], set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (matrices if isinstance(module, nn.Linear) and name == "weight" else others).append(p)
    # Hyperball group first: train.py logs param_groups[0]["lr"] as the applied LR.
    return Hyperball([
        dict(params=matrices, lr=learning_rate, hyperball=True),
        dict(params=others, lr=learning_rate * adam_lr_ratio, hyperball=False),
    ], lr=learning_rate, betas=(beta1, beta2))
