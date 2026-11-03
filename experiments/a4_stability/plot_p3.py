"""Problem 3(a)-(c): residual mixing across depth in d8 and the modern RNN.

    uv run python -m experiments.a4_stability.plot_p3

Reads results/a4_stability/p3/outcomes.json (p3_outcomes), the supplied manifest for the additive baselines,
p3_probe's JSONs (the d8 baseline reuses p1_probe's), and public W&B baseline histories (cached under cache/).
Writes to experiments/a4_stability/plots/p3/:
  sweep.png             final val loss vs LR per variant against the supplied baseline sweep; val curves at best LR
  depth_<family>_<group>.png   activation RMS, gradient RMS for L and L_T, raw branch/input ratio, adjacent
                        cosine across depth entries, at init / step 100 / final (best LR of each variant)
  routing_<family>.png  attention-residual softmax weights: router (depth step) x source slot, step 100 and final
Prints a table of best losses, gaps to the tuned baseline, and parameter counts (results/a4_stability/p3/summary.json).
"""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from experiments.a4_stability.plot_p1a import history as public_history, manifest_runs, savefig as _savefig
from experiments.a4_stability.sweeps import run_name

RESULTS = Path('results/a4_stability/p3')
P1_RESULTS = Path('results/a4_stability/p1')
OUTCOMES = RESULTS/'outcomes.json'
TOKENS_PER_STEP = 64*1024
FAMILIES = {'d8': 'd8 transformer', 'rnn': 'modern 4-layer RNN'}
SCHEMES = ('baseline', 'half', 'ema01', 'ema09', 'uniform', 'attention', 'block2', 'block4')
LABELS = dict(baseline='additive (supplied)', half='half-and-half', ema01='EMA β=0.1', ema09='EMA β=0.9',
              uniform='uniform routing', attention='full AttnRes', block2='block AttnRes B=2',
              block4='block AttnRes B=4')
COLORS = dict(baseline='black', half='tab:blue', ema01='tab:red', ema09='tab:orange', uniform='tab:green',
              attention='tab:purple', block2='tab:brown', block4='tab:pink')
GROUPS = {'fixed': ('baseline', 'half', 'ema09', 'uniform'),
          'ema': ('baseline', 'ema01', 'ema09'),
          'routed': ('baseline', 'uniform', 'attention', 'block2', 'block4')}
STAGES = ('init', 'early', 'final')
STAGE_TITLES = dict(init='init (seed 42)', early='step 100', final='final')
STATS = [('activation_rms', 'full', 'activation RMS', True),
         ('gradient_rms', 'full', 'gradient RMS, L (all tokens)', True),
         ('gradient_rms', 'last', 'gradient RMS, L_T (last token)', True),
         ('raw_branch_input_ratio', 'full', 'RMS F(N(x)) / RMS x', True),
         ('adjacent_cosine', 'full', 'cos(entry j-1, entry j)', False)]
BLOCK = dict(attention=1, block2=2, block4=4)


def savefig(fig, name):
    return _savefig(fig, f'p3/{name}')


def variant(family, scheme):
    return family if scheme == 'baseline' else f'{family}-{scheme}'


def sweep(family, scheme):
    """[(lr, loss or None, status)] for one variant; baselines come from the supplied manifest."""
    if scheme == 'baseline':
        return [(r['lr'], r['validation_loss'], r['status']) for r in manifest_runs(family)]
    rows = json.loads(OUTCOMES.read_text())
    return sorted((r['lr'], r['validation_loss'], r['status']) for r in rows if r['variant'] == variant(family, scheme))


def best_lr(family, scheme):
    done = [(loss, lr) for lr, loss, status in sweep(family, scheme) if status == 'complete' and loss is not None]
    return min(done)[1] if done else None


def probe_lr(family, scheme):
    from experiments.a4_stability.p3_probe import BEST
    return .003 if (family, scheme) == ('d8', 'baseline') else BEST[variant(family, scheme)]


def probe(family, scheme, stage):
    name, lr = variant(family, scheme), probe_lr(family, scheme)
    if family == 'd8' and scheme == 'baseline':
        label = f'a4-d8-init' if stage == 'init' else f'{run_name(name, lr)}-{stage}'
        path = P1_RESULTS/f'{label}.json'
        return json.loads(path.read_text())['depth'] if path.exists() else None
    label = f'{name}-init' if stage == 'init' else f'{run_name(name, lr)}-{stage}'
    path = RESULTS/f'{label}.json'
    return json.loads(path.read_text()) if path.exists() else None


def val_curve(family, scheme):
    lr = best_lr(family, scheme)
    if scheme == 'baseline':
        run = next(r for r in manifest_runs(family) if r['lr'] == lr)
        steps, values = public_history(run)['val_loss']
        return steps, values
    row = next(r for r in json.loads(OUTCOMES.read_text())
               if r['variant'] == variant(family, scheme) and r['lr'] == lr)
    steps, values = zip(*row['history']['val_loss'])
    return np.array(steps, float), np.array(values, float)


def parameter_counts():
    path = RESULTS/'parameters.json'
    if path.exists():
        return json.loads(path.read_text())
    from train import build_model
    from experiments.a4_stability.sweeps import config
    counts = {}
    for family in FAMILIES:
        for scheme in SCHEMES:
            model = build_model(config(variant(family, scheme), precision='fp32'), 'cpu')
            counts[variant(family, scheme)] = sum(p.numel() for p in model.parameters())
    path.write_text(json.dumps(counts, indent=2) + '\n')
    return counts


def plot_sweep():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        notes = []
        for scheme in SCHEMES:
            points = sweep(family, scheme)
            done = [(lr, loss) for lr, loss, status in points if status == 'complete']
            if not done:
                if points:
                    notes.append(f"{LABELS[scheme]}: {points[0][2].upper()} at batch 64, every LR")
                continue
            lrs, losses = zip(*done)
            ax.plot(lrs, losses, 'o-', color=COLORS[scheme], label=LABELS[scheme],
                    lw=2.5 if scheme == 'baseline' else 1.5)
            if min(losses) > 4:
                notes.append(f'{LABELS[scheme]}: {min(losses):.3f} at every LR (off-scale)')
        ax.set_xscale('log')
        ax.set_ylim((2.89, 3.1) if family == 'd8' else (3.48, 3.80))
        ax.set_xlabel('peak learning rate')
        ax.set_ylabel('final validation loss')
        ax.set_title(f'{FAMILIES[family]}: LR sweeps (600k sequences)')
        if notes:
            ax.text(.02, .97, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(fc='white', ec='0.7'))
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8, loc='lower right' if family == 'd8' else 'upper right')

        ax = axes[1, col]
        for scheme in SCHEMES:
            if best_lr(family, scheme) is None:
                continue
            steps, values = val_curve(family, scheme)
            if values[-1] > 4:
                continue
            ax.plot((steps+1)*TOKENS_PER_STEP/1e6, values, color=COLORS[scheme],
                    label=f'{LABELS[scheme]} (lr {best_lr(family, scheme):g})', lw=2.5 if scheme == 'baseline' else 1.2)
        ax.set_xlim(150, 620)
        ax.set_ylim((2.88, 3.15) if family == 'd8' else (3.48, 3.85))
        ax.set_xlabel('training tokens (M)')
        ax.set_ylabel('validation loss')
        ax.set_title(f'{FAMILIES[family]}: validation loss at each best LR')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 'sweep.png')


def depth_axis_labels(family, count):
    """Depth entries are branch inputs (attn/rec, MLP alternating) then the final stream."""
    names = []
    for j in range(count-1):
        names.append(('A' if family == 'd8' else 'R') + str(j//2+1) if j % 2 == 0 else f'M{j//2+1}')
    return names + ['out']


def plot_depth(family, group):
    schemes = GROUPS[group]
    fig, axes = plt.subplots(len(STATS), len(STAGES), figsize=(15, 3.1*len(STATS)), sharex=True)
    for col, stage in enumerate(STAGES):
        for row, (key, objective, title, log) in enumerate(STATS):
            ax = axes[row, col]
            for scheme in schemes:
                data = probe(family, scheme, stage)
                if data is None:
                    continue
                depth = data[objective]['depth']
                xs = [j for j, d in enumerate(depth) if key in d]
                ys = [depth[j][key] for j in xs]
                ax.plot(xs, ys, 'o-', ms=3, color=COLORS[scheme], label=LABELS[scheme],
                        lw=2.2 if scheme == 'baseline' else 1.2)
            if log:
                ax.set_yscale('log')
            if row == 0:
                ax.set_title(STAGE_TITLES[stage])
            if col == 0:
                ax.set_ylabel(title)
            ax.grid(alpha=.3, which='both')
            count = len(depth)
            ax.set_xticks(range(count))
            ax.set_xticklabels(depth_axis_labels(family, count), fontsize=7, rotation=90 if count > 10 else 0)
    axes[-1, 1].set_xlabel('depth entry (branch input; last = final stream before the output norm)')
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f'{FAMILIES[family]}: depth statistics, {group} mixing (probed at '
                 + ', '.join(f'{LABELS[s]} lr {probe_lr(family, s):g}' for s in schemes) + '; '
                 '8 held-out sequences)', y=1.0, fontsize=10)
    fig.tight_layout()
    savefig(fig, f'depth_{family}_{group}.png')


def routing_matrix(weights, block):
    """router j x source slot; slot 0 = embedding, slot k = block k (completed or the partial in progress)."""
    routers = sorted(weights, key=lambda name: int(name.split('.')[1]))
    rows = [weights[name] for name in routers]
    matrix = np.full((len(rows), max(map(len, rows))), np.nan)
    for j, w in enumerate(rows):
        matrix[j, :len(w)] = w
    return matrix


def plot_routing(family):
    schemes = [s for s in ('attention', 'block2', 'block4') if probe(family, s, 'final') is not None]
    if not schemes:
        return
    fig, axes = plt.subplots(len(schemes), 2, figsize=(11, 3.6*len(schemes)), squeeze=False)
    for row, scheme in enumerate(schemes):
        for col, stage in enumerate(('early', 'final')):
            ax = axes[row, col]
            matrix = routing_matrix(probe(family, scheme, stage)['routing'], BLOCK[scheme])
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            branches = matrix.shape[0]-1
            ax.set_yticks(range(branches+1))
            ax.set_yticklabels(depth_axis_labels(family, branches+1), fontsize=7)
            ax.set_xticks(range(matrix.shape[1]))
            ax.set_xticklabels(['emb'] + [f'{"b" if BLOCK[scheme] == 1 else "B"}{k}' for k in range(1, matrix.shape[1])],
                               fontsize=7, rotation=90 if matrix.shape[1] > 10 else 0)
            ax.set_xlabel('source (emb, then branch outputs / blocks; last = partial)', fontsize=9)
            ax.set_ylabel('router (branch input; out = final read)', fontsize=9)
            ax.set_title(f'{LABELS[scheme]}, {STAGE_TITLES[stage]} (lr {probe_lr(family, scheme):g})')
            if BLOCK[scheme] == 1 or matrix.shape[1] <= 18:
                for j in range(matrix.shape[0]):
                    for k in range(matrix.shape[1]):
                        if np.isfinite(matrix[j, k]) and matrix.shape[1] <= 10:
                            ax.text(k, j, f'{matrix[j, k]:.2f}', ha='center', va='center', fontsize=6,
                                    color='white' if matrix[j, k] < .5 else 'black')
            fig.colorbar(image, ax=ax, fraction=.04, label='mean softmax weight')
    fig.suptitle(f'{FAMILIES[family]}: attention-residual routing weights (mean over 8×1024 held-out tokens)')
    fig.tight_layout()
    savefig(fig, f'routing_{family}.png')


def summary():
    counts = parameter_counts()
    rows = []
    for family in FAMILIES:
        base = min(loss for _, loss, status in sweep(family, 'baseline') if status == 'complete')
        for scheme in SCHEMES:
            points = sweep(family, scheme)
            done = [(loss, lr) for lr, loss, status in points if status == 'complete']
            best = min(done) if done else (None, None)
            rows.append(dict(variant=variant(family, scheme), best_loss=best[0], best_lr=best[1],
                             gap=None if best[0] is None else best[0]-base,
                             rates=[lr for lr, _, _ in points],
                             losses=[loss for _, loss, _ in points], statuses=[s for _, _, s in points],
                             parameters=counts[variant(family, scheme)]))
    (RESULTS/'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        loss = '—' if r['best_loss'] is None else f"{r['best_loss']:.4f}"
        gap = '' if r['gap'] is None else f"{r['gap']:+.4f}"
        print(f"{r['variant']:16s} {loss:>8s} {gap:>8s} lr={r['best_lr']} params={r['parameters']:,} "
              + ' '.join(f'{lr:g}:{"—" if l is None else f"{l:.3f}"}' for lr, l in zip(r['rates'], r['losses'])))


if __name__ == '__main__':
    plot_sweep()
    for family in FAMILIES:
        for group in GROUPS:
            plot_depth(family, group)
        plot_routing(family)
    summary()
