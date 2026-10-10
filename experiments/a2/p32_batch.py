"""Problem 3.2 source runs at 614.4M tokens (B = 8, 16, 32; one B = 64 fill-in).

(a) / hypothesis (i): WD .1, LR sweep. Reused rather than rerun: B = 16 and 32 at LR .001/.003/.01 (A1 P1
    runs, `REUSED_A1`; same model, data prefix, and AdamW math as A2) and B = 64 (A2 P1a). Only B = 8 is new,
    centred on the B = 8 optimum extrapolated from those fits (~0.0009-0.0014).
(b) hypothesis (ii): LR .0015, WD sweep bracketing both "WD constant" (~0.43, the P2a optimum at LR .0015)
    and "WD proportional to B" (the WD-timescale rule). B = 64 reuses P2a's WD .1/.2/.4 and adds .8,
    since .4 was the best (edge) point. No existing runs at LR .0015 for B < 64.
Targets at B = 128/256 and part (c) depend on these fits and are not here.
"""
import argparse

EXPERIMENT_KEY = 'a2-p32-D614m'
TOKENS = 614_400_000
LR_SWEEP = {8: (.0005, .001, .002)}  # WD .1
REUSED_A1 = {  # (batch, lr) at WD .1 -> A1 run name
    (16, .001): 'model-d8-lr0.001-bs16-tok614M-p1b-v1', (16, .003): 'model-d8-lr0.003-bs16-tok614M-p1a-v1',
    (16, .01): 'model-d8-lr0.01-bs16-tok614M-p1b-v1', (32, .001): 'model-d8-lr0.001-bs32-tok614M-p1b-v1',
    (32, .003): 'model-d8-lr0.003-bs32-tok614M-p1a-v1', (32, .01): 'model-d8-lr0.01-bs32-tok614M-p1b-v1',
}
WD_SWEEP = {32: (.1, .2, .4, .8), 16: (.1, .2, .4, .8), 8: (.05, .1, .2, .4), 64: (.8,)}   # LR .0015


def source_runs():
    """(batch, lr, wd) for every source run, deduplicated, in launch order."""
    runs = [(b, lr, .1) for b, lrs in LR_SWEEP.items() for lr in lrs]
    runs += [(b, .0015, wd) for b, wds in WD_SWEEP.items() for wd in wds]
    return list(dict.fromkeys(runs))


def round2(x):
    return float(f'{x:.2g}')


# Predictions from plot_p32ab (fits over B = 8..64): (i) lr* = 3.455e-4 B^0.567 at WD .1;
# (ii) wd* = 8.367e-3 B^0.976 at LR .0015. Each target tests the prediction and x/2, x2 around it.
PREDICTED = {128: dict(lr=.00542, wd=.954), 256: dict(lr=.00803, wd=1.877)}


def target_runs():
    """(batch, lr, wd) for part (b)'s B = 128/256 tests of both hypotheses."""
    runs = []
    for b, p in PREDICTED.items():
        runs += [(b, round2(p['lr'] * f), .1) for f in (.5, 1, 2)]      # hypothesis (i)
        runs += [(b, .0015, round2(p['wd'] * f)) for f in (.5, 1, 2)]   # hypothesis (ii)
    return runs


def configs(runs):
    from experiments.a2.modal_launcher import config
    return [config(tokens=TOKENS, batch=b, learning_rate=lr, weight_decay=wd,
                   run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for b, lr, wd in runs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('source', 'targets'), required=True)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    runs = source_runs() if args.part == 'source' else target_runs()
    for b, lr, wd in runs:
        print(f'B={b:<3} lr={lr:<8g} wd={wd:g}')
    print(f'{len(runs)} runs')
    if args.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(configs(runs))
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
