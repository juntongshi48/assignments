"""Problem 1 follow-up: prediction ensembles of weight-averaged midpoints, for every way to pair four runs.

Cells (500, 250) and (250, 500) both end at T = 750 updates, giving four branches that share the
first 250 updates:

    a1, a2 = fork250-order{1,2}   (branched at 250)
    b1, b2 = fork500-order{1,2}   (branched at 500)

batch_order_seed permutes batches by global step, so a1 and b1 (and a2 and b2) see identical
batches over steps 500-750. PAIRINGS lists the three ways to split the four runs into two pairs.
For each pair: its midpoint (0.5/0.5 weight average) loss, barrier, and endpoint ensemble; for each
pairing: the 2-way prediction ensemble of its two midpoints. Also: each endpoint, the 4-way weight
average, and the 4-way endpoint ensemble. Everything is validation loss at T = 750.

    uv run python -m experiments.a3_optimization.p1_midpoint_ensemble
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import modal

from experiments.a3_optimization.connectivity import _val_batches, checkpoint_path
from experiments.a3_optimization.p1_mode_connectivity import BRANCHES, EXPERIMENT_KEY
from modal_utils import MODAL_ENVIRONMENT, MODAL_MODEL_DIR, VOLUME_MOUNTS, build_image, timestamped_modal_app_name
from train import training_run_name

T = 750
MEMBERS = dict(a1=(250, 0), a2=(250, 1), b1=(500, 0), b2=(500, 1))  # name: (x, index into BRANCHES[x])
PAIRINGS = {
    "by_cell": (("a1", "a2"), ("b1", "b2")),  # same branch point, different data order
    "by_seed": (("a1", "b1"), ("a2", "b2")),  # same batches over steps 500-750
    "crossed": (("a1", "b2"), ("a2", "b1")),  # different branch point and data order
}
RESULTS_PATH = Path(__file__).resolve().parent / "results" / f"{EXPERIMENT_KEY}-midpoint-pairings.json"


def member_refs():
    """{member: (run_name, step)} at total step T."""
    return {name: (training_run_name(BRANCHES[x][i]), T) for name, (x, i) in MEMBERS.items()}


def mixture_loss(models, batches, device, precision):
    """NLL of the uniform mixture of the models' next-token probabilities."""
    import torch
    import torch.nn.functional as F
    from utils import autocast_context

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids in batches:
            input_ids = input_ids.to(device)
            targets = input_ids[..., 1:].unsqueeze(-1)
            log_probs = []
            for model in models:
                with autocast_context(precision, device=device):
                    logits = model(input_ids=input_ids)
                log_probs.append(F.log_softmax(logits[..., :-1, :].float(), dim=-1).gather(-1, targets))
            mixture = torch.logsumexp(torch.stack(log_probs), dim=0) - math.log(len(models))
            total_nll -= mixture.sum().item()
            total_tokens += targets.numel()
    return total_nll / total_tokens


def averaged_model(models):
    """A copy of models[0] whose parameters are the uniform average of the models' parameters."""
    import copy

    import torch

    average = copy.deepcopy(models[0])
    with torch.no_grad():
        for target, *sources in zip(average.parameters(), *(m.parameters() for m in models)):
            target.copy_(torch.stack([s.detach().float() for s in sources]).mean(0).to(target.dtype))
    return average.eval()


def measure(refs, *, load, batches, pairings=PAIRINGS, device="cuda", precision="mp"):
    """refs: {member: ref}; load(ref) returns an eval-mode model. Returns {key: loss}."""
    from train import evaluate

    config = SimpleNamespace(precision=precision)
    loss = lambda model: evaluate(model, batches, config, device).item()
    ensemble = lambda models: mixture_loss(models, batches, device, precision)

    models = {name: load(ref) for name, ref in refs.items()}
    out = {f"endpoint/{name}": loss(model) for name, model in models.items()}
    out["average4"] = loss(averaged_model(list(models.values())))
    out["ens/endpoints/all4"] = ensemble(list(models.values()))
    for pairing, pairs in pairings.items():
        midpoints = []
        for pair in pairs:
            key = "+".join(pair)
            midpoint = averaged_model([models[name] for name in pair])
            midpoints.append(midpoint)
            out[f"{pairing}/midpoint/{key}"] = loss(midpoint)
            out[f"{pairing}/barrier/{key}"] = out[f"{pairing}/midpoint/{key}"] - sum(
                out[f"endpoint/{name}"] for name in pair) / len(pair)
            out[f"{pairing}/ens_endpoints/{key}"] = ensemble([models[name] for name in pair])
        out[f"{pairing}/ens_midpoints"] = ensemble(midpoints)
    return out


app = modal.App("dl-alchemy-a3-midpoint-ensemble")


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=60 * 60)
def _measure_remote(refs):
    from model_io import load_model

    load = lambda ref: load_model(checkpoint_path(MODAL_MODEL_DIR, *ref), device="cuda").eval()
    return measure({name: tuple(ref) for name, ref in refs.items()}, load=load, batches=_val_batches())


def main():
    refs = member_refs()
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name("dl-alchemy-a3-midpoint-ensemble"),
                     environment_name=MODAL_ENVIRONMENT):
            losses = _measure_remote.remote(refs)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(dict(refs=refs, pairings=PAIRINGS, losses=losses), indent=2) + "\n")
    for key, value in losses.items():
        print(f"{key:36s} {value:+.4f}" if "/barrier/" in key else f"{key:36s} {value:.4f}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
