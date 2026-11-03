"""Problem 2: fixed-input depth and time probes for supplied RNN checkpoints, run locally.

    uv run python -m experiments.a4_stability.p2_probe

Checkpoints are local copies of the shared volume (`modal volume get -e cs312-shared-data hard-dl-dclm-v1
a4_stability/<run> ckpts/a4_stability/`); the held-out batch is P1's (first 8 rows of the ds42 val split).
For every run this writes results/a4_stability/p2/<run>-{early,final}.json, plus one <variant>-init.json
rebuilt with model seed 42. Each file holds
  depth:      diagnostics.profile (depth rows; per layer and time: h RMS, dL/dh and dL_T/dh RMS, Pr(|h| > .95))
  positions:  per-position depth-stream RMS and gradient RMS for L and L_T at every branch input and the output
  recurrence: per layer, mean tanh'(h_t) = 1 - h_t^2 by position, and the spectrum of the temporal matrix W_h
  output:     logit scale and, for gated variants, the sigmoid gate values
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import profile, save
from .p1_probe import CKPT_ROOT, held_out, initial_model, output_and_gates, trained_model
from .p2_rnn_depth import Recurrent

OUTPUT = Path('results/a4_stability/p2')
VARIANTS = ('rnn1', 'rnn4', 'rnn', 'rnn-no-residual', 'rnn-no-prenorm', 'rnn-rope', 'rnn-final-norm',
            'rnn-postnorm', 'rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm')


def per_position(x):
    return x.detach().double().square().mean((0, 2)).sqrt().tolist()


def position_profile(model, tokens):
    """Per-position RMS (over batch and width) of every depth-stream entry and its L / L_T gradients.

    Entries are the branch inputs followed by the output-norm input. For the plain RNNs these are the
    embedding and each layer's h; for the modern RNN they alternate recurrent and MLP inputs."""
    stream, handles = {}, []

    def keep(index):
        def hook(module, args):
            x = args[0]
            x.retain_grad()
            stream[index] = x
        return hook
    branches = model.branches()
    for j, branch in enumerate(branches):
        handles.append(branch.register_forward_pre_hook(keep(j)))
    handles.append(model.output_norm.register_forward_pre_hook(keep(len(branches))))
    result = {}
    try:
        for objective in ('full', 'last'):
            model.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            loss = (F.cross_entropy(logits[:, -1], tokens[:, -1]) if objective == 'last' else
                    F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()))
            loss.backward()
            entries = [stream[j] for j in sorted(stream)]
            if objective == 'full':
                result['activation'] = [per_position(x) for x in entries]
            result[f'gradient_{objective}'] = [per_position(x.grad) for x in entries]
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    return result


def recurrence_profile(model, tokens):
    """The one-step Jacobian is dh_t/dh_{t-1} = diag(1 - h_t^2) W_h: its two factors, per layer.

    Recording states retains their gradients, so the forward runs with autograd on."""
    sequences = [m for m in model.modules() if isinstance(m, Recurrent)]
    for m in sequences:
        m.record_states = True
    try:
        model(tokens[:, :-1])
        layers = []
        for m in sequences:
            h = torch.stack(m.states, 1).detach().double()
            derivative = 1 - h.square()
            W = m.cell.hidden.weight.detach().double()
            singular = torch.linalg.svdvals(W)
            eigen = torch.linalg.eigvals(W).abs()
            layers.append(dict(
                tanh_derivative=derivative.mean((0, 2)).tolist(),
                tanh_derivative_mean=derivative.mean().item(),
                W_h_singular_max=singular[0].item(), W_h_singular_median=singular.median().item(),
                W_h_spectral_radius=eigen.max().item(),
                W_h_singular_quantiles=torch.quantile(singular, torch.tensor([.1, .5, .9], dtype=torch.double)).tolist(),
                bias_rms=m.cell.hidden.bias.detach().double().square().mean().sqrt().item()
                         if m.cell.hidden.bias is not None else 0.))
        return layers
    finally:
        for m in sequences:
            m.record_states, m.states = False, []
        model.zero_grad(set_to_none=True)


def measure(model, tokens):
    model.eval()
    result = dict(depth=profile(model, tokens), positions=position_profile(model, tokens),
                  recurrence=recurrence_profile(model, tokens), output=output_and_gates(model, tokens))
    result['parameters'] = sum(p.numel() for p in model.parameters())
    return result


def measure_variant(name, tokens, *, stages=('early', 'final'), force=False):
    """Init once per variant, then every downloaded LR's checkpoints."""
    from .sweeps import learning_rates, run_name
    jobs = [(f'a4-{name}-init', lambda: initial_model(name))]
    for lr in learning_rates(name):
        run = run_name(name, lr)
        for stage in stages:
            if (CKPT_ROOT/run/('early' if stage == 'early' else '')/'model.pt').exists():
                jobs.append((f'{run}-{stage}', lambda run=run, stage=stage: trained_model(run, stage)))
    for label, load in jobs:
        path = OUTPUT/f'{label}.json'
        if path.exists() and not force:
            continue
        stats = measure(load(), tokens)
        stats['tokens'] = dict(rows=tokens.shape[0], first_token=tokens[:, 0].tolist())
        save(path, stats)
        sat = np.mean([np.mean(s) for s in stats['depth']['full']['fraction_abs_gt_095']])
        print(path, f"L={stats['depth']['full']['loss']:.4f} L_T={stats['depth']['last']['loss']:.4f} sat={sat:.3f}")


if __name__ == '__main__':
    torch.set_num_threads(8)
    tokens = held_out()
    for variant in VARIANTS:
        measure_variant(variant, tokens)
