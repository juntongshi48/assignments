"""Problem 3.2(c): momentum ablation at B = 8 and 256 (614.4M tokens), beta2 = .95 fixed.

Each batch holds its best measured LR-WD pair from 3.2(a, b) and sweeps beta1 (beta1 = 0 is the no-momentum
control, i.e. RMSProp with decoupled WD). The beta1 = .9 point at each batch is the existing source/target run
(tag a2-p32-D614m), so it is not rerun. beta1 = .98 and .99 exceed sqrt(beta2) ~ .975 and test whether the
NQM's large-beta1 blow-up (3.1d) shows up in the LM.
"""
import argparse

EXPERIMENT_KEY = 'a2-p32c-D614m'
TOKENS = 614_400_000
BEST_PAIR = {8: dict(lr=.001, wd=.1), 256: dict(lr=.0015, wd=1.9)}  # best measured (lr, wd) at beta1 = .9
BETA1S = (0.0, .5, .8, .95, .98, .99)  # .9 reused from a2-p32-D614m


def runs():
    """(batch, lr, wd, beta1) for every new run."""
    return [(b, p['lr'], p['wd'], beta1) for b, p in BEST_PAIR.items() for beta1 in BETA1S]


def configs(rs):
    from experiments.a2.modal_launcher import config
    return [config(tokens=TOKENS, batch=b, learning_rate=lr, weight_decay=wd, beta1=beta1,
                   run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for b, lr, wd, beta1 in rs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    rs = runs()
    for b, lr, wd, beta1 in rs:
        print(f'B={b:<3} lr={lr:<7g} wd={wd:<4g} beta1={beta1:g}')
    print(f'{len(rs)} runs')
    if args.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(configs(rs))
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
