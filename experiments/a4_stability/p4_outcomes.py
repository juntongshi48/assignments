"""Problem 4: collect final validation losses and training curves of our runs from W&B.

    uv run python -m experiments.a4_stability.p4_outcomes

Writes results/a4_stability/p4/outcomes.json: one row per (variant, lr) with the finished run's val_loss, or
status 'failed' when no attempt finished, plus val_loss and logged global RMS histories.
"""
import json
from pathlib import Path

from .sweeps import VARIANTS, run_name

OUTPUT = Path('results/a4_stability/p4/outcomes.json')
PROJECT = 'shisteve/cs312-a1'
BASE_URL = 'https://api.forge.coreweave.com'
HISTORY_KEYS = ('val_loss', 'logging/rms/global/activation', 'logging/rms/global/gradient')
OOM = set()


def main():
    import wandb
    api = wandb.Api(overrides={'base_url': BASE_URL}, timeout=120)
    runs = api.runs(PROJECT, filters={'tags': {'$in': ['p4']}})
    by_name = {}
    for run in runs:
        if run.state == 'finished' or run.name not in by_name:
            by_name[run.name] = run
    rows = []
    for name, variant in VARIANTS.items():
        if variant.problem != 4:
            continue
        for remote in (r for r in by_name.values() if r.name.startswith(f'a4-{name}-lr')):
            lr = float(remote.name.rsplit('-lr', 1)[1])
            row = dict(variant=name, lr=lr, run_name=run_name(name, lr), url=remote.url)
            if remote.state == 'finished':
                row.update(status='complete', validation_loss=remote.summary.get('val_loss'), history={
                    key: [(r['optimizer_step'], r[key]) for r in remote.scan_history(keys=['optimizer_step', key])
                          if r.get(key) is not None] for key in HISTORY_KEYS})
            else:
                row.update(status='oom' if name in OOM else remote.state, validation_loss=None)
            rows.append(row)
    rows.sort(key=lambda r: (r['variant'], r['lr']))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows) + '\n')
    for r in rows:
        print(r['run_name'], r['status'], r['validation_loss'])


if __name__ == '__main__':
    main()
