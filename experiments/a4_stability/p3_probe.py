"""Problem 3: fixed-input depth probes for our residual-mixing checkpoints and the supplied baselines, run locally.

    uv run python -m experiments.a4_stability.p3_probe

Checkpoints are local copies of our volume (`modal volume get -e cs312-juntong volume-dl_alchemy
/ckpts/<run>/<file> ckpts/a4_stability/<run>/`: config.json, metadata.json, model.pt, step_100.pt), or of the
shared volume for the RNN baseline. The held-out batch is p1_probe's (first BATCH rows of the ds42 val split).
For every (variant, lr) this writes results/a4_stability/p3/<run>-{early,final}.json, plus one <variant>-init.json
rebuilt with model seed 42. Each file is diagnostics.profile: activation / gradient RMS for L and L_T,
raw branch/input ratios, adjacent cosine across depth, and per-router softmax weights over sources.
"""
from pathlib import Path

import torch

from .diagnostics import profile, save
from .p1_probe import CKPT_ROOT, held_out, initial_model
from .sweeps import run_name

OUTPUT = Path('results/a4_stability/p3')
# Best rate of each variant's (1/3, 1, 3) x center grid (d8-ema01 ties at every rate); the d8 baseline
# reuses P1's probes.
BEST = {'rnn': .002, 'rnn-half': .001, 'rnn-ema01': .001, 'rnn-ema09': .001, 'rnn-uniform': .001,
        'rnn-attention': .001, 'rnn-block2': .001, 'rnn-block4': .001,
        'd8-half': .003, 'd8-ema01': .003, 'd8-ema09': .003, 'd8-uniform': .003,
        'd8-attention': .009, 'd8-block2': .003, 'd8-block4': .003}
EXTRA = {'rnn': (.001,),  # The RNN baseline at the variants' common best rate.
         'd8-attention': (.003,)}  # Within .0023 of its edge-best .009.


def checkpoint(run, stage):
    from model_io import load_model
    directory = CKPT_ROOT/run
    if stage == 'early':
        path = directory/'early' if (directory/'early').is_dir() else directory/'step_100.pt'
    else:
        path = directory
    return load_model(path, device='cpu', precision='fp32').eval()


def measure(name, rates, tokens, force=False):
    jobs = [(f'{name}-init', lambda: initial_model(name))]
    for lr in rates:
        run = run_name(name, lr)
        if not (CKPT_ROOT/run/'model.pt').exists():
            print('missing', run)
            continue
        jobs += [(f'{run}-{stage}', lambda run=run, stage=stage: checkpoint(run, stage))
                 for stage in ('early', 'final')]
    for label, build in jobs:
        path = OUTPUT/f'{label}.json'
        if path.exists() and not force:
            continue
        torch.manual_seed(0)
        save(path, profile(build(), tokens))
        print(path)


if __name__ == '__main__':
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokens = held_out()
    for name, lr in BEST.items():
        measure(name, (lr,) + EXTRA.get(name, ()), tokens)
