"""P1(d) target experiments: Hyperball at 1.2288B tokens, at the predicted LR and three nearby LRs."""
import argparse

EXPERIMENT_KEY = 'a2-p1d-hyperball-D1229m'
TARGET_TOKENS = 1_228_800_000
# Prediction from plot_p1d (local quadratic fits, power law over 153.6M-614.4M): lr* = 0.01048.
PREDICTED_LR = .0105
NEARBY_LRS = (.007, .015, .021)


def runs(lrs):
    from experiments.a2.modal_launcher import config
    return [config(tokens=TARGET_TOKENS, learning_rate=lr, weight_decay=0.0,
                   optimizer_builder='experiments.a2.hyperball:build_hyperball',
                   run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for lr in dict.fromkeys(lrs)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lrs', type=float, nargs='+', default=(PREDICTED_LR, *NEARBY_LRS))
    parser.add_argument('--execute', action='store_true', help='Submit the target runs to Modal.')
    args = parser.parse_args(argv)
    configs = runs(args.lrs)
    for c in configs:
        print(f'tokens={TARGET_TOKENS} lr={c.learning_rate:g} wd={c.weight_decay:g} '
              f'optimizer={c.optimizer_builder} name_suffix={c.run_name_suffix}')
    if args.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(configs)
    else:
        print('Preview only. Add --execute to train these target configurations.')


if __name__ == '__main__':
    main()
