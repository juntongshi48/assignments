"""P2(c) target experiments at 2.4576B tokens: test the product law against two baselines.

  * prediction: peak LR .003 with WD = (product law at 2.4576B) / .003;
  * baseline (i): the lowest-loss LR-WD pair of the 153.6M sweep, without retuning;
  * baseline (ii): peak LR .003 with WD in {.05, .1, .2}; WD .1 is the supplied P1b run, so not rerun.
"""
import argparse

EXPERIMENT_KEY = 'a2-p2c-D2458m'
TARGET_TOKENS = 2_457_600_000
# plot_p2b: eta*lambda = 102.9 * D^-0.593 fit on 153.6M-1.2288B -> 2.782e-4 at 2.4576B; / .003 = 0.0927.
PREDICTED = (.003, .0927)
BASELINE_I = (.0015, 1.6)       # best measured pair at 153.6M (loss 3.1844)
BASELINE_II = ((.003, .05), (.003, .2))  # (.003, .1) reused from P1b (loss 2.7683)


def runs():
    from experiments.a2.modal_launcher import config
    return [config(tokens=TARGET_TOKENS, learning_rate=lr, weight_decay=wd,
                   run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for lr, wd in (PREDICTED, BASELINE_I, *BASELINE_II)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Submit the target runs to Modal.')
    args = parser.parse_args(argv)
    configs = runs()
    for c in configs:
        print(f'tokens={TARGET_TOKENS} lr={c.learning_rate:g} wd={c.weight_decay:g} suffix={c.run_name_suffix}')
    if args.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(configs)
    else:
        print('Preview only. Add --execute to train these target configurations.')


if __name__ == '__main__':
    main()
