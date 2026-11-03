"""Problem 4: temporal probes for the GRU, its ablations, our pre-norm updates and the tanh baseline, on CPU.

    uv run python -m experiments.a4_stability.p4_probe [run ...]

Checkpoints are local copies of our volume (`modal volume get -e cs312-juntong volume-dl_alchemy
/ckpts/<run>/<file> ckpts/a4_stability/<run>/`: config.json, metadata.json, model.pt, step_100.pt); the tanh
baseline is the supplied a4-rnn-lr0.002. Each run writes results/a4_stability/p4/<run>-{init,early,final}.json:
  probe:    p2_probe.measure on P1's held-out batch (per layer and time: h RMS, dL/dh_t and dL_T/dh_t RMS,
            Pr(|h| > .95); depth profile; one-step Jacobian factors)
  gates:    per layer, the update gate z (fraction of NEW information) and reset gate r: mean, quantiles,
            mean by position, and per-channel mean z summarized as a memory timescale 1/z
and, for the final checkpoint only,
  context:  last-token loss on CONTEXT_ROWS validation rows keeping only the most recent `keep` tokens
  context_bos: the same with BOS kept in front (`--bos` adds it to existing final JSONs)
  reach:    RMS of dL_T/d(embedding output) by position on REACH_ROWS rows
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .diagnostics import context_loss, save
from .p1_probe import held_out, initial_model
from .p2_context import gradient_reach, validation_rows
from .p2_probe import measure as temporal_probe
from .p3_probe import checkpoint
from .p4_recurrent_updates import Gate
from .p2_rnn_depth import Recurrent

OUTPUT = Path('results/a4_stability/p4')
RUNS = {'a4-gru-lr0.003': 'gru', 'a4-gru-no-reset-lr0.003': 'gru-no-reset',
        'a4-gru-no-update-lr0.003': 'gru-no-update', 'a4-gru-fixed-update-lr0.003': 'gru-fixed-update',
        'a4-gru-input-gates-lr0.003': 'gru-input-gates', 'a4-gru-prenorm-lr0.006': 'gru-prenorm',
        'a4-gru-prenorm-linear-lr0.003': 'gru-prenorm-linear', 'a4-rnn-lr0.002': 'rnn'}
KEEPS = (1, 4, 16, 64, 256)
CONTEXT_ROWS, REACH_ROWS, CHUNK = 64, 16, 16
QUANTILES = (.05, .25, .5, .75, .95)


def gate_profile(model, tokens):
    """Record every gate output during one eager forward; CPU runs cell.step token by token."""
    layers = [m for m in model.modules() if isinstance(m, Recurrent)]
    seen, handles = {}, []
    for i, layer in enumerate(layers):
        for kind in ('update', 'reset'):
            gate = getattr(layer.cell, kind, None)
            if isinstance(gate, Gate):
                handles.append(gate.register_forward_hook(
                    lambda module, args, out, key=(i, kind): seen.setdefault(key, []).append(out.detach())))
    try:
        with torch.no_grad():
            model(tokens[:, :-1])
    finally:
        for handle in handles:
            handle.remove()
    result = []
    for i in range(len(layers)):
        row = {}
        for kind, values in ((k, seen.get((i, k))) for k in ('update', 'reset')):
            if values is None:
                continue
            g = torch.stack(values, 1).double()  # (batch, time, width)
            channel = g.mean((0, 1))
            row[kind] = dict(mean=g.mean().item(),
                             quantiles=torch.quantile(g.flatten()[::7], torch.tensor(QUANTILES, dtype=g.dtype)).tolist(),
                             by_position=g.mean((0, 2)).tolist(),
                             channel_quantiles=torch.quantile(channel, torch.tensor(QUANTILES, dtype=g.dtype)).tolist())
            if kind == 'update':
                timescale = 1 / channel.clamp_min(1e-6)
                row[kind].update(timescale_median=timescale.median().item(),
                                 frac_channels_timescale_gt_20=(timescale > 20).double().mean().item(),
                                 frac_channels_timescale_gt_100=(timescale > 100).double().mean().item())
        result.append(row)
    return result


def context_use(model, rows, first=False):
    """first=True keeps BOS before the last `keep` tokens: without it the tanh RNN leaves its training
    distribution (P2 found +50 nats at every k), so the BOS-kept curve is the one that measures context use."""
    length = rows.shape[1] - 1
    out = dict(full=[], keep={})
    with torch.no_grad():
        for start in range(0, len(rows), CHUNK):
            batch = rows[start:start + CHUNK]
            out['full'] += context_loss(model, batch, keep=length).tolist()
            for keep in KEEPS:
                out['keep'].setdefault(str(keep), []).extend(context_loss(model, batch, keep, first).tolist())
    return out


def add_bos_context(run, rows):
    path = OUTPUT/f'{run}-final.json'
    stats = json.loads(path.read_text())
    if 'context_bos' not in stats:
        stats['context_bos'] = context_use(checkpoint(run, 'final'), rows, first=True)
        save(path, stats)
        print(path, 'context_bos', flush=True)


def build(run, stage):
    return initial_model(RUNS[run]) if stage == 'init' else checkpoint(run, stage)


def measure(run, tokens, rows, stages=('init', 'early', 'final')):
    for stage in stages:
        path = OUTPUT/f'{run}-{stage}.json'
        if path.exists():
            continue
        torch.manual_seed(0)
        model = build(run, stage)
        stats = dict(probe=temporal_probe(model, tokens), gates=gate_profile(model, tokens))
        if stage == 'final':
            stats['context'] = context_use(model, rows)
            stats['context_bos'] = context_use(model, rows, first=True)
            stats['reach'] = gradient_reach(model, rows[:REACH_ROWS])
        save(path, stats)
        print(path, f"L={stats['probe']['depth']['full']['loss']:.4f}", flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    runs = [a for a in sys.argv[1:] if a != '--bos'] or list(RUNS)
    tokens, rows = held_out(), validation_rows(CONTEXT_ROWS)
    if '--bos' in sys.argv:
        for run in runs:
            add_bos_context(run, rows)
        sys.exit()
    for run in runs:
        measure(run, tokens, rows, ('final',) if run == 'a4-rnn-lr0.002' else ('init', 'early', 'final'))
