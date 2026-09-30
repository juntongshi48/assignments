"""Metric-logger hooks for Problems 3-4: parameter noise injection and a parameter fingerprint.

Both use private torch.Generators, so they never consume the global RNG and leave the
training trajectory of an unperturbed run unchanged.

Modal resolves loggers by import path, so every noise setting is a module-level
object named by `noise_logger_name` and referenced as `noise_logger_path(...)`.
"""

import math

import torch

from metric_logging import AFTER_EVAL, AFTER_TRAIN_STEP, MetricLogger

NOISE_SEED = 0
FINGERPRINT_SEED = 1234
FINGERPRINT_DIM = 64

# (after_steps, relative scale) for Problem 4(b). after_steps=1 is the earliest point a
# post-step hook can reach. 1e-7 is roughly one fp32 ulp of a typical weight.
NOISE_SETTINGS = (
    *((after_steps, scale) for scale in (1e-6, 1e-3) for after_steps in (1, 2000, 5000, 8000)),
    (1, 1e-7),
    (5000, 1e-1),
)


def _sorted_parameters(model):
    return sorted(model.named_parameters(), key=lambda item: item[0])


class ParamNoise:
    """After `after_steps` optimizer steps, add scale * RMS(W) * Z to every tensor W.

    Z comes from a fixed seed, so runs that differ only in `scale` share one direction.
    Logs the realized relative change ||W' - W|| / ||W|| to confirm it was not rounded away.
    """

    def __init__(self, after_steps, scale, seed=NOISE_SEED):
        self.after_steps = after_steps
        self.scale = scale
        self.seed = seed

    def __call__(self, ctx):
        if ctx.step + 1 != self.after_steps:
            return {}
        params = _sorted_parameters(ctx.model)
        generator = torch.Generator(device=params[0][1].device).manual_seed(self.seed)
        delta_sq = norm_sq = 0.0
        with torch.no_grad():
            for _, param in params:
                noise = torch.randn(param.shape, generator=generator, device=param.device, dtype=torch.float32)
                before = param.detach().float().clone()
                rms = before.pow(2).mean().sqrt()
                param.add_((noise * (self.scale * rms)).to(param.dtype))
                delta_sq += (param.detach().float() - before).pow(2).sum().item()
                norm_sq += before.pow(2).sum().item()
        return {
            "perturbation/requested_relative_scale": self.scale,
            "perturbation/realized_relative_norm": math.sqrt(delta_sq / norm_sq),
        }


def noise_logger_name(after_steps, scale):
    return f"noise_after{after_steps}_scale{scale:.0e}".replace("-", "m").replace("+", "")


for _after_steps, _scale in NOISE_SETTINGS:
    globals()[noise_logger_name(_after_steps, _scale)] = ParamNoise(_after_steps, _scale)


def noise_logger(after_steps, scale):
    return MetricLogger(event=AFTER_TRAIN_STEP, fn=f"{__name__}:{noise_logger_name(after_steps, scale)}")


def log_param_fingerprint(ctx):
    """Project all parameters onto FINGERPRINT_DIM fixed Gaussian directions.

    For two runs a and b, ||theta_a - theta_b|| ~= sqrt(mean_i (p_a,i - p_b,i)^2), so
    comparing logged projections gives parameter-space divergence without checkpoints.
    """
    params = _sorted_parameters(ctx.model)
    device = params[0][1].device
    generator = torch.Generator(device=device).manual_seed(FINGERPRINT_SEED)
    projection = torch.zeros(FINGERPRINT_DIM, device=device)
    norm_sq = torch.zeros((), device=device)
    with torch.no_grad():
        for _, param in params:
            flat = param.detach().float().flatten()
            directions = torch.randn(
                (FINGERPRINT_DIM, flat.numel()), generator=generator, device=device, dtype=torch.float32
            )
            projection += directions @ flat
            norm_sq += flat.pow(2).sum()
    stats = {f"param_fingerprint/{i:02d}": value for i, value in enumerate(projection.tolist())}
    stats["param_fingerprint/norm"] = norm_sq.sqrt().item()
    return stats


FINGERPRINT_LOGGER = MetricLogger(event=AFTER_EVAL, fn=log_param_fingerprint)
