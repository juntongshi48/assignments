"""Problem 4: GRU ablations and pre-norm-across-time updates against the supplied tanh RNN.

    uv run python -m experiments.a4_stability.plot_p4

Reads results/a4_stability/p4/outcomes.json (p4_outcomes), the supplied manifest for the tanh RNN, and p4_probe's
JSONs. Writes to experiments/a4_stability/plots/p4/:
  loss.png       final val loss vs LR (tanh sweep, GRU and proposals at 2-3 rates, ablations at the GRU's best)
  time.png       final checkpoints vs time index t: dL_T/dh_t (how far back the last prediction reaches),
                 dL/dh_t, ||h_t||_rms and Pr(|h_t| > .95), each the mean over the four recurrent layers
  stages.png     the same dL_T/dh_t at init / step 100 / final for the tanh-free proposal, the GRU, and z = 1
  gates.png      per layer: distribution over channels of the mean update gate z (new-information fraction)
                 and reset gate r
  context.png    last-token loss with BOS + the most recent k tokens (and without BOS), minus the full-context loss
and results/a4_stability/p4/summary.json with the scalars quoted in the report.
"""
import json
from pathlib import Path

import numpy as np

from experiments.a4_stability.plot_p1a import manifest_runs, plt, savefig as _savefig

RESULTS = Path('results/a4_stability/p4')
MODELS = {  # variant -> (probed run, label, color)
    'rnn': ('a4-rnn-lr0.002', 'tanh RNN (supplied)', 'black'),
    'gru': ('a4-gru-lr0.003', 'GRU', 'tab:blue'),
    'gru-no-reset': ('a4-gru-no-reset-lr0.003', 'GRU, r = 1', 'tab:orange'),
    'gru-input-gates': ('a4-gru-input-gates-lr0.003', 'GRU, gates see x only', 'tab:olive'),
    'gru-fixed-update': ('a4-gru-fixed-update-lr0.003', 'GRU, z = 0.5', 'tab:pink'),
    'gru-no-update': ('a4-gru-no-update-lr0.003', 'GRU, z = 1 (no carry)', 'tab:red'),
    'gru-prenorm': ('a4-gru-prenorm-lr0.006', 'pre-norm GRU', 'tab:cyan'),
    'gru-prenorm-linear': ('a4-gru-prenorm-linear-lr0.003', 'pre-norm GRU, no tanh', 'tab:green'),
}
STAGE_STYLES = {'init': ':', 'early': '--', 'final': '-'}


def savefig(fig, name):
    return _savefig(fig, f'p4/{name}')


def load(run, stage='final'):
    path = RESULTS/f'{run}-{stage}.json'
    return json.loads(path.read_text()) if path.exists() else None


def outcomes():
    rows = {}
    for r in json.loads((RESULTS/'outcomes.json').read_text()):
        if r['status'] == 'complete':
            rows.setdefault(r['variant'], []).append((r['lr'], r['validation_loss']))
    rows['rnn'] = [(r['lr'], r['validation_loss']) for r in manifest_runs('rnn') if r['status'] == 'complete']
    return {k: sorted(v) for k, v in rows.items()}


def layer_mean(rows, log=False):
    a = np.array(rows, dtype=float)
    return np.exp(np.log(np.maximum(a, 1e-30)).mean(0)) if log else a.mean(0)


def plot_loss(sweeps):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, (_, label, color) in MODELS.items():
        pts = sweeps.get(variant, [])
        if not pts:
            continue
        lr, loss = zip(*pts)
        ax.plot(lr, loss, 'o-' if len(pts) > 1 else 's', color=color, label=f'{label}: {min(loss):.3f}')
    ax.set(xscale='log', xlabel='learning rate', ylabel='final val loss', ylim=(3.15, 3.8),
           title='P4: tuned losses (614M tokens, same 4-layer residual pre-norm MLP stack)')
    ax.legend(fontsize=8)
    savefig(fig, 'loss.png')


def plot_time(probes):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    keys = [('last', 'hidden_gradient_rms', 'dL_T/dh_t RMS', True), ('full', 'hidden_gradient_rms', 'dL/dh_t RMS', True),
            ('full', 'hidden_rms', '||h_t|| RMS', False), ('full', 'fraction_abs_gt_095', 'Pr(|h_t| > .95)', False)]
    for ax, (objective, key, title, log) in zip(axes, keys):
        for variant, (run, label, color) in MODELS.items():
            p = probes.get(variant)
            if p is None:
                continue
            y = layer_mean(p['probe']['depth'][objective][key], log)
            ax.plot(np.arange(len(y)), y, color=color, label=label, lw=1)
        ax.set(title=title + ' (mean of 4 layers)', xlabel='time index t', yscale='log' if log else 'linear')
    axes[0].legend(fontsize=7)
    savefig(fig, 'time.png')


def plot_stages():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, variant in zip(axes, ('gru-prenorm-linear', 'gru', 'gru-no-update')):
        run, label, color = MODELS[variant]
        for stage, style in STAGE_STYLES.items():
            p = load(run, stage)
            if p is not None:
                ax.plot(layer_mean(p['probe']['depth']['last']['hidden_gradient_rms'], True), style,
                        color=color, label=stage)
        ax.set(title=f'{label}: dL_T/dh_t', xlabel='time index t', yscale='log')
        ax.legend(fontsize=8)
    savefig(fig, 'stages.png')


def plot_gates(probes):
    gated = [v for v in MODELS if probes.get(v) and any(probes[v]['gates'])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, kind in zip(axes, ('update', 'reset')):
        for i, variant in enumerate(gated):
            run, label, color = MODELS[variant]
            for layer, row in enumerate(probes[variant]['gates']):
                if kind not in row:
                    continue
                q = row[kind]['channel_quantiles']
                x = layer + (i - len(gated)/2) * .08
                ax.plot([x, x], [q[0], q[4]], color=color, lw=1)
                ax.plot([x, x], [q[1], q[3]], color=color, lw=4)
                ax.plot(x, q[2], 'o', color=color, ms=4, label=label if layer == 0 else None)
        ax.set(title=f'{kind} gate: per-channel mean over tokens (5/25/50/75/95%)', xlabel='recurrent layer',
               xticks=range(4), xticklabels=range(1, 5), yscale='log' if kind == 'update' else 'linear')
        ax.legend(fontsize=7)
    savefig(fig, 'gates.png')


def plot_context(probes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, key, title in ((axes[0], 'context_bos', 'BOS + last k tokens'),
                           (axes[1], 'context', 'last k tokens only (no BOS: off-distribution)')):
        for variant, (run, label, color) in MODELS.items():
            p = probes.get(variant)
            if p is None or key not in p:
                continue
            full = np.mean(p[key]['full'])
            keeps = sorted(p[key]['keep'], key=int)
            ax.plot([int(k) for k in keeps], [max(np.mean(p[key]['keep'][k]) - full, 1e-3) for k in keeps], 'o-',
                    color=color, label=f'{label} (full {full:.3f})')
        ax.set(xscale='log', yscale='log', xlabel='tokens of context kept, k', ylabel='L_T(k) - L_T(full)',
               title=title)
        ax.legend(fontsize=7)
    fig.suptitle('Last-token loss lost by truncating the context (64 val rows)')
    savefig(fig, 'context.png')


def summary(sweeps, probes):
    out = {}
    for variant, (run, label, _) in MODELS.items():
        row = dict(label=label, sweep=sweeps.get(variant, []))
        if row['sweep']:
            row['best_lr'], row['best_loss'] = min(row['sweep'], key=lambda p: p[1])
        p = probes.get(variant)
        if p:
            d = p['probe']['depth']
            g = layer_mean(d['last']['hidden_gradient_rms'], True)
            row['grad_LT_ratio_t0_over_tlast'] = float(g[0] / g[-1])
            row['grad_LT_ratio_t_minus_64'] = float(g[-65] / g[-1])
            row['grad_LT_ratio_t_minus_256'] = float(g[-257] / g[-1])
            row['h_rms'] = float(np.mean(d['full']['hidden_rms']))
            row['saturation'] = float(np.mean(d['full']['fraction_abs_gt_095']))
            row['params'] = p['probe']['parameters']
            row['update_gate'] = [r['update'] for r in p['gates'] if 'update' in r] and [
                dict(mean=r['update']['mean'], timescale_median=r['update']['timescale_median'],
                     frac_gt_20=r['update']['frac_channels_timescale_gt_20'],
                     frac_gt_100=r['update']['frac_channels_timescale_gt_100']) for r in p['gates']]
            row['reset_gate_mean'] = [r['reset']['mean'] for r in p['gates'] if 'reset' in r]
            if 'context_bos' in p:
                full = np.mean(p['context_bos']['full'])
                row['context_full'] = float(full)
                row['context_gap'] = {k: float(np.mean(v) - full) for k, v in p['context_bos']['keep'].items()}
                row['context_gap_no_bos'] = {k: float(np.mean(v) - full) for k, v in p['context']['keep'].items()}
        out[variant] = row
    (RESULTS/'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    for variant, row in out.items():
        print(variant, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                        if k not in ('sweep', 'label', 'update_gate', 'context_gap', 'context_gap_no_bos')})
        if row.get('update_gate'):
            print('   z', [{k: round(v, 3) for k, v in r.items()} for r in row['update_gate']])
        if row.get('reset_gate_mean'):
            print('   r', [round(v, 3) for v in row['reset_gate_mean']])
        if 'context_gap' in row:
            print('   context gap', {k: round(v, 3) for k, v in row['context_gap'].items()})


if __name__ == '__main__':
    sweeps = outcomes()
    probes = {v: load(run) for v, (run, _, _) in MODELS.items()}
    plot_loss(sweeps)
    plot_time(probes)
    plot_stages()
    plot_gates(probes)
    plot_context(probes)
    summary(sweeps, probes)
