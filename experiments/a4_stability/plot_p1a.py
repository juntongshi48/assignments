"""Problem 1(a): the supplied default d8 sweep, its depth diagnostics, and its attention patterns.

    uv run python -m experiments.a4_stability.plot_p1a

Reads results/a4_stability/manifest.json (copy of the shared volume's manifest), the probe JSONs written by
p1_probe, and the public W&B histories (cached under cache/a4_p1/). Writes to experiments/a4_stability/plots/:
  p1a_sweep.png       final val loss vs LR; val loss and logged global RMS over training, with the end of the
                      linear warmup / start of linear decay marked. The RMS logger has only step 0 inside warmup
                      (dashed to the next sample); validation is first logged after warmup.
  p1a_depth.png       activation RMS, gradient RMS for L and L_T, branch/input ratio, adjacent cosine across depth
                      (top: init / step 100 / final at the best LR; bottom: final checkpoint of every LR)
  p1a_positions.png   stream RMS by position (BOS spike) and gradient RMS by lag from the final token
  p1a_attention.png   head-averaged mass by lag, first-token and self mass by query vs uniform, layer x head maps
  p1a_heads.png       every head's mass-by-lag profile, one panel per layer
  p1a_sink.png        sink mass across stages and LRs, BOS value/stream RMS, loss with key 0 blocked, no-BOS control
"""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from experiments.a4_stability.p1_probe import LATE, OUTPUT as RESULTS
from experiments.a4_stability.sweeps import PUBLIC_PROJECT, learning_rates, run_name

PLOT_DIR = Path(__file__).resolve().parent / 'plots'
CACHE = Path('cache/a4_p1')
MANIFEST = Path('results/a4_stability/manifest.json')
TOKENS_PER_STEP = 64 * 1024
HISTORY_KEYS = ('val_loss', 'train_loss', 'learning_rate', 'logging/rms/global/activation',
                'logging/rms/global/gradient')
STAGE_COLORS = {'init': '0.6', 'early': 'tab:orange', 'final': 'tab:blue'}
STAGE_LABELS = {'init': 'init (seed 42)', 'early': 'step 100', 'final': 'final'}


def savefig(fig, name):
    path = PLOT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(path.resolve())
    return path


def tokens_seen(step):
    """Training tokens (M) consumed through optimizer step `step`, counting its own batch."""
    return (np.asarray(step) + 1) * TOKENS_PER_STEP / 1e6


def manifest_runs(variant):
    runs = json.loads(MANIFEST.read_text())['runs']
    return sorted((r for r in runs if r['variant'] == variant), key=lambda r: r['lr'])


def history(run):
    """{key: (steps, values)} for one public run, cached after the first download."""
    path = CACHE / f"{run['run_name']}.json"
    if not path.exists() or set(HISTORY_KEYS) - set(json.loads(path.read_text())):
        import wandb
        api = wandb.Api(timeout=120)
        # Some links point at a mirror host; the run id is the same in the public project.
        remote = api.run(f"{PUBLIC_PROJECT}/{run['url'].rstrip('/').split('/')[-1]}")
        data = {}
        for key in HISTORY_KEYS:
            # val_loss rows carry optimizer_step but not step.
            rows = [r for r in remote.scan_history(keys=['optimizer_step', key]) if r.get(key) is not None]
            data[key] = ([r['optimizer_step'] for r in rows], [r[key] for r in rows])
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return {k: (np.array(s, float), np.array(v, float)) for k, (s, v) in json.loads(path.read_text()).items()}


def probe(label):
    return json.loads((RESULTS / f'{label}.json').read_text())


def lr_colors(rates):
    return dict(zip(rates, plt.cm.viridis(np.linspace(0, .9, len(rates)))))


def plain_number(v, _=None):
    return f'{v:.10g}' if abs(v) >= 1e-3 else f'{v:.0e}'


def readable_log_axis(ax):
    """Log y ticks at 1, 2, 5 x 10^k as plain numbers, with a light grid to read values across."""
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1., 2., 5.)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(plain_number))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(which='major', axis='y', alpha=.3)


def label_points(ax, values, color, entries=(0, 2, -1)):
    for j in entries:
        j = j % len(values)
        ax.annotate(f'{values[j]:.2g}', (j, values[j]), textcoords='offset points', xytext=(4, -9 if j else 5),
                    fontsize=7, color=color, zorder=5,
                    bbox=dict(boxstyle='round,pad=.1', fc='w', ec='none', alpha=.8))


def depth_ticks(ax, entries):
    ax.set_xticks(range(0, entries, 2), [f'L{j}' for j in range(entries // 2)] + ['out'])
    ax.set_xticks(range(entries), minor=True)
    ax.set_xlabel('depth: attn input of block Lj (minor tick = MLP input)', fontsize=8)


def sweep_figure(variant, runs):
    rates = [r['lr'] for r in runs]
    colors = lr_colors(rates)
    best = min((r for r in runs if r['status'] == 'complete'), key=lambda r: r['validation_loss'])
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    ax = axes[0]
    done = [r for r in runs if r['status'] == 'complete']
    ax.plot([r['lr'] for r in done], [r['validation_loss'] for r in done], 'o-', color='k')
    for r in runs:
        if r['status'] != 'complete':
            ax.axvline(r['lr'], color='tab:red', ls=':', label=f"lr {r['lr']:g} {r['status']}")
        else:
            ax.annotate(f"{r['validation_loss']:.4f}", (r['lr'], r['validation_loss']), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=8)
    ax.plot(best['lr'], best['validation_loss'], '*', ms=16, color='tab:red', label=f"best lr {best['lr']:g}")
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='final validation loss',
           title=f'{variant}: supplied five-rate sweep')
    ax.set_xticks(rates, [f'{lr:g}' for lr in rates])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend(fontsize=8)
    histories = {r['lr']: history(r) for r in runs}
    for ax, key, ylabel, title in [
            (axes[1], 'val_loss', 'validation loss', 'validation loss (first logged at step 93, after warmup)'),
            (axes[2], 'logging/rms/global/activation', 'global activation RMS', 'RMS logger: activations'),
            (axes[3], 'logging/rms/global/gradient', 'global parameter-gradient RMS', 'RMS logger: gradients')]:
        for lr, h in histories.items():
            steps, values = h[key]
            # x = tokens seen through this step, so step 0 (inside warmup) sits on the log axis.
            x = tokens_seen(steps)
            ax.plot(x[1:], values[1:], color=colors[lr], lw=1.2, label=f'lr {lr:g}')
            if steps[0] == 0 and len(steps) > 1 and steps[1] > 1:
                # Nothing is logged between step 0 and the next sample: mark the gap as such.
                ax.plot(x[:2], values[:2], color=colors[lr], lw=1, ls='--', alpha=.6)
                ax.plot(x[0], values[0], 'o', ms=4, color=colors[lr])
        ax.set(xscale='log', yscale='log' if 'rms' in key else 'linear', xlabel='training tokens (M)',
               ylabel=ylabel, title=title)
        if 'rms' in key:
            readable_log_axis(ax)
    # Every run shares the schedule: linear warmup to the peak, then linear decay to 0.
    steps, rate = histories[best['lr']]['learning_rate']
    decay = tokens_seen(steps[rate.argmax()])
    for ax in axes[1:]:
        ax.axvline(decay, color='tab:red', ls=':', lw=1.2, label=f'LR decay starts ({decay:.1f}M)')
        ax.set_xlim(left=.05)
    axes[1].set_ylim(best['validation_loss'] - .05, 4.0)
    axes[1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(.1))
    axes[1].legend(fontsize=7, loc='lower left')
    axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(.01))
    for ax in axes[:2]:
        ax.grid(which='major', axis='y', alpha=.3)
    savefig(fig, 'p1a_sweep.png')
    return best


def depth_rows(label, objective, key):
    return np.array([row.get(key, np.nan) for row in probe(label)['depth'][objective]['depth']])


def depth_figure(variant, best_lr, rates):
    stages = {'init': f'a4-{variant}-init', 'early': f'{run_name(variant, best_lr)}-early',
              'final': f'{run_name(variant, best_lr)}-final'}
    finals = {lr: f'{run_name(variant, lr)}-final' for lr in rates if (RESULTS / f'{run_name(variant, lr)}-final.json').exists()}
    colors = lr_colors(rates)
    panels = [('full', 'activation_rms', 'activation RMS', 'log'),
              ('full', 'gradient_rms', 'gradient RMS for L (full sequence)', 'log'),
              ('last', 'gradient_rms', 'gradient RMS for L_T (final token)', 'log'),
              ('full', 'raw_branch_input_ratio', 'branch / input RMS  ||F(N(x))|| / ||x||', 'log'),
              ('full', 'stream_and_update', 'stream RMS ||x|| vs update RMS ||F(N(x))||', 'log'),
              ('full', 'adjacent_cosine', 'cosine(entry j-1, entry j)', 'linear')]
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 8.5))
    for row, curves in enumerate([{STAGE_LABELS[s]: (l, STAGE_COLORS[s]) for s, l in stages.items()},
                                  {f'lr {lr:g}' + (' (best)' if lr == best_lr else ''): (l, colors[lr])
                                   for lr, l in finals.items()}]):
        for ax, (objective, key, title, scale) in zip(axes[row], panels):
            for name, (label, color) in curves.items():
                if key == 'stream_and_update':
                    # Update added at entry j: ||F(N(x_j))|| = ratio * ||x_j||; the final entry has none.
                    stream = depth_rows(label, objective, 'activation_rms')
                    ax.plot(np.arange(len(stream)), stream, 'o-', ms=3, color=color, label=name)
                    values = stream * depth_rows(label, objective, 'raw_branch_input_ratio')
                    ax.plot(np.arange(len(values)), values, 's--', ms=3, color=color, alpha=.8)
                    continue
                values = depth_rows(label, objective, key)
                ax.plot(np.arange(len(values)), values, 'o-', ms=3, color=color, label=name)
                if key == 'activation_rms':
                    label_points(ax, values, color)
            ax.set(yscale=scale, title=title)
            if key == 'stream_and_update':
                ax.legend([mpl.lines.Line2D([], [], color='k', marker='o', ms=3),
                           mpl.lines.Line2D([], [], color='k', marker='s', ms=3, ls='--')],
                          ['residual stream x entering j', 'update F(N(x)) added at j'], fontsize=7, loc='lower right')
            if key in ('activation_rms', 'raw_branch_input_ratio', 'stream_and_update'):
                readable_log_axis(ax)
            else:
                ax.grid(which='major', axis='y', alpha=.3)
            depth_ticks(ax, len(values))
        axes[row, 0].legend(fontsize=8, title=f'lr {best_lr:g}' if row == 0 else 'final checkpoints')
    savefig(fig, 'p1a_depth.png')


def positions_figure(variant, best_lr):
    label = f'{run_name(variant, best_lr)}-final'
    positions = probe(label)['positions']
    activation = np.array(positions['activation'])
    entries = activation.shape[0]
    shown = [0, 2, 4, 6, 8, 12, entries - 1]
    colors = dict(zip(shown, plt.cm.plasma(np.linspace(0, .85, len(shown)))))
    name = lambda j: 'final stream' if j == entries - 1 else f"{'attn' if j % 2 == 0 else 'MLP'} input L{j // 2}"
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    x = np.arange(activation.shape[1]) + 1
    for j in shown:
        axes[0].plot(x, activation[j], color=colors[j], lw=1, label=name(j))
    axes[0].set(xscale='log', yscale='log', xlabel='position + 1 (BOS at 1)', ylabel='stream RMS over batch, width',
                title=f'activation RMS by position (lr {best_lr:g}, final)')
    axes[0].legend(fontsize=7)
    for stage, run in [('init', f'a4-{variant}-init'), ('early', f'{run_name(variant, best_lr)}-early'), ('final', label)]:
        a = np.array(probe(run)['positions']['activation'])
        axes[1].plot(range(entries), a[:, 0] / np.median(a[:, 1:], 1), 'o-', ms=3, color=STAGE_COLORS[stage],
                     label=STAGE_LABELS[stage])
    axes[1].axhline(1, color='k', lw=.8, ls=':')
    axes[1].set(yscale='log', ylabel='RMS at BOS / median RMS of other positions', title='BOS massive activation')
    depth_ticks(axes[1], entries)
    axes[1].legend(fontsize=8)
    for ax, key, title in [(axes[2], 'gradient_last', 'L_T gradient RMS vs lag from final token'),
                           (axes[3], 'gradient_full', 'L gradient RMS vs lag from final token')]:
        g = np.array(positions[key])
        lag = np.arange(g.shape[1])[::-1] + 1
        for j in shown:
            ax.plot(lag, g[j], color=colors[j], lw=1, label=name(j))
        ax.set(xscale='log', yscale='log', xlabel='lag from final input position + 1', ylabel='gradient RMS', title=title)
    savefig(fig, 'p1a_positions.png')


def heat(ax, values, title, cmap='viridis', norm=None, fmt='{:.2f}'):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=6.5,
                color='w' if image.norm(v) < .55 else 'k')
    ax.set(xticks=range(values.shape[1]), yticks=range(values.shape[0]), xlabel='head', ylabel='layer', title=title)
    plt.colorbar(image, ax=ax, fraction=.046)


def attention_figure(variant, best_lr):
    attention = probe(f'{run_name(variant, best_lr)}-final')['attention']
    layers = len(attention)
    colors = plt.cm.viridis(np.linspace(0, .95, layers))
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    lag = np.arange(LATE)
    for j, a in enumerate(attention):
        axes[0, 0].plot(lag + 1, np.mean(a['lag_mass'], 0), color=colors[j], lw=1.2, label=f'layer {j}')
    axes[0, 0].axhline(attention[0]['lag_uniform'], color='k', ls='--', lw=1, label='uniform over prefix')
    axes[0, 0].set(xscale='log', yscale='log', xlabel='lag q - k  (+1)', ylabel='mean attention mass',
                   title=f'mass by lag, head average, queries q >= {LATE}')
    axes[0, 0].legend(fontsize=7, ncol=2)
    for ax, key, title in [(axes[0, 1], 'first_mass', 'mass on key 0 (BOS) by query position'),
                           (axes[0, 2], 'self_mass', 'mass on the current token by query position')]:
        for j, a in enumerate(attention):
            ax.plot(np.array(a['query_points']) + 1, np.mean(a[key], 0), color=colors[j], lw=1.2, label=f'layer {j}')
        ax.plot(np.array(attention[0]['query_points']) + 1, attention[0]['uniform_point'], 'k--', lw=1,
                label='uniform 1/(q+1)')
        ax.set(xscale='log', yscale='log', xlabel='query position q (+1)', ylabel='attention mass (head average)', title=title)
    axes[0, 1].legend(fontsize=7, ncol=2)
    first = np.array([a['first_late'] for a in attention]) / attention[0]['first_uniform_late']
    heat(axes[1, 0], first, f'BOS mass / uniform (q >= {LATE})', 'magma',
         mpl.colors.LogNorm(vmin=max(first.min(), .1), vmax=first.max()), '{:.0f}')
    heat(axes[1, 1], np.array([[h[0] for h in a['lag_mass']] for a in attention]),
         f'self (lag 0) mass, q >= {LATE}', 'Blues')
    heat(axes[1, 2], np.array([a['entropy_late'] for a in attention]) / attention[0]['entropy_uniform_late'],
         'entropy / uniform-prefix entropy', 'viridis')
    savefig(fig, 'p1a_attention.png')


def heads_figure(variant, best_lr):
    attention = probe(f'{run_name(variant, best_lr)}-final')['attention']
    heads = len(attention[0]['lag_mass'])
    colors = plt.cm.tab10(np.arange(heads))
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True, sharey=True)
    lag = np.arange(LATE) + 1
    for j, (ax, a) in enumerate(zip(axes.flat, attention)):
        for h in range(heads):
            ax.plot(lag, np.clip(a['lag_mass'][h], 1e-6, None), color=colors[h], lw=.9,
                    label=f"h{h}: BOS {a['first_late'][h]:.2f}, self {a['lag_mass'][h][0]:.2f}")
        ax.axhline(a['lag_uniform'], color='k', ls='--', lw=.8)
        ax.set(xscale='log', yscale='log', title=f'layer {j}', ylim=(1e-6, 1))
        ax.legend(fontsize=6, loc='upper right')
    for ax in axes[1]:
        ax.set_xlabel('lag q - k (+1)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'mass, queries q >= {LATE}')
    fig.suptitle(f'd8 lr {best_lr:g} final: per-head attention mass by lag (dashed: uniform); legend gives BOS and self mass')
    savefig(fig, 'p1a_heads.png')


def sink_figure(variant, best_lr, rates):
    colors = lr_colors(rates)
    stages = [('init', f'a4-{variant}-init'), ('early', f'{run_name(variant, best_lr)}-early'),
              ('final', f'{run_name(variant, best_lr)}-final')]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ax = axes[0, 0]
    for stage, label in stages:
        a = probe(label)['attention']
        ax.plot([np.mean(x['first_late']) for x in a], 'o-', color=STAGE_COLORS[stage], label=f'lr {best_lr:g} {STAGE_LABELS[stage]}')
    for lr in rates:
        if lr != best_lr:
            a = probe(f'{run_name(variant, lr)}-final')['attention']
            ax.plot([np.mean(x['first_late']) for x in a], '.--', lw=1, color=colors[lr], label=f'lr {lr:g} final')
    ax.axhline(a[0]['first_uniform_late'], color='k', ls=':', label='uniform')
    ax.set(yscale='log', xlabel='layer', ylabel=f'BOS mass, head average (q >= {LATE})', title='sink strength')
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for stage, label in stages:
        a = probe(label)['attention']
        ax.plot([np.mean(np.array(x['value_rms_first']) / np.array(x['value_rms_rest'])) for x in a], 'o-',
                color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
    ax.axhline(1, color='k', ls=':')
    ax.set(xlabel='layer', ylabel='value RMS at BOS / at other positions', title='BOS values shrink: a no-op sink')
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for stage, label in stages:
        attractors = probe(label)['attention']
        ax.plot([x['attractors'][0]['key0_ratio'] for x in attractors], 'o-', color=STAGE_COLORS[stage],
                label=f'BOS, {STAGE_LABELS[stage]}')
        ax.plot([x['attractors'][0]['top'][0]['ratio'] for x in attractors], 's:', color=STAGE_COLORS[stage],
                label=f'strongest other key, {STAGE_LABELS[stage]}')
    ax.axhline(1, color='k', lw=.8)
    ax.set(yscale='log', xlabel='layer', ylabel='received mass / uniform (queries >= 8 back)',
           title='does any other position attract mass?')
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    edges = np.unique(np.geomspace(1, 1022, 24).astype(int))
    for lr in rates:
        s = probe(f'{run_name(variant, lr)}-final')['sink']
        delta = np.array(s['block_key0']['by_position']) - np.array(s['normal']['by_position'])
        mids = [np.sqrt(lo * hi) for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mids, [delta[lo - 1:hi - 1].mean() for lo, hi in zip(edges[:-1], edges[1:])], 'o-', ms=3,
                color=colors[lr], label=f"lr {lr:g}: mean +{s['block_key0']['mean'] - s['normal']['mean']:.3f}")
    ax.axhline(0, color='k', lw=.8)
    ax.set(xscale='log', xlabel='target position', ylabel='loss increase (nats)', title='block attention to key 0 (final)')
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    s = probe(f'{run_name(variant, best_lr)}-final')
    ax.plot([np.mean(x['first_late']) for x in s['attention']], 'o-', color='tab:blue', label='with BOS: mass on BOS')
    ax.plot([np.mean(x) for x in s['sink']['drop_bos_first_late']], 's-', color='tab:red',
            label='BOS dropped: mass on new first token')
    ax.axhline(s['attention'][0]['first_uniform_late'], color='k', ls=':', label='uniform')
    ax.set(yscale='log', xlabel='layer', ylabel=f'first-key mass, head average (q >= {LATE})',
           title=f'is the sink BOS or "first"? (lr {best_lr:g} final)')
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for stage, label in stages:
        a = np.array(probe(label)['positions']['activation'])
        ax.plot(range(a.shape[0]), a[:, 0] / np.median(a[:, 1:], 1), 'o-', ms=3, color=STAGE_COLORS[stage],
                label=STAGE_LABELS[stage])
    ax.axhline(1, color='k', ls=':')
    ax.set(yscale='log', ylabel='stream RMS at BOS / median other positions', title='BOS stream norm across depth')
    depth_ticks(ax, a.shape[0])
    ax.legend(fontsize=8)
    savefig(fig, 'p1a_sink.png')


def token_pieces(ids):
    from huggingface_hub import hf_hub_download
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(hf_hub_download('Ryukijano/parameter-golf-sp4096', 'fineweb_4096_bpe.model',
                                   subfolder='tokenizers', repo_type='dataset'))
    return [tokenizer.id_to_piece(int(i)).replace('▁', '␣') for i in ids]


def attention_grid_figure(variant, best_lr, length=20, row=0, layer=None, head=None):
    """A[i, j] for the first `length` positions: mean over layers and heads, or one (layer, head), 0-indexed."""
    from experiments.a4_stability.p1_probe import attention_matrices, held_out, trained_model
    tokens = held_out()
    a = attention_matrices(trained_model(run_name(variant, best_lr), 'final'), tokens, length)
    if layer is None:
        per_row, which, name = a.mean((0, 1)).numpy(), f'mean over {a.shape[0]} layers x {a.shape[1]} heads', ''
    else:
        per_row, which, name = a[layer, head].numpy(), f'layer {layer} head {head}', f'_l{layer}h{head}'
    pieces = token_pieces(tokens[row, :length])
    grids = [(per_row.mean(0), f'{which}, mean over {a.shape[2]} sequences', False),
             (per_row[row], f'{which}, sequence {row}', True)]
    fig, axes = plt.subplots(2, 2, figsize=(17, 16.5), gridspec_kw=dict(hspace=.32))
    # Top: full rows (each sums to 1). Bottom: same weights without BOS (row and column 0), on their own color scale.
    for axes_row, first, vmax, label in [
            (axes[0], 0, 1., 'attention weight A[i, j]  (each row sums to 1)'),
            (axes[1], 1, max(g[1:, 1:].max() for g, _, _ in grids), 'A[i, j], BOS column removed  (row sums to 1 - A[i, 0])')]:
        for ax, (grid, title, named) in zip(axes_row, grids):
            grid = grid[first:, first:]
            image = ax.imshow(np.ma.masked_where(np.triu(np.ones_like(grid, bool), 1), grid),
                              cmap='viridis', vmin=0, vmax=vmax)
            for (i, j), v in np.ndenumerate(grid):
                if j <= i:
                    text = f'{v:.2f}'
                    ax.text(j, i, '1' if text == '1.00' else text[1:], ha='center', va='center', fontsize=6,
                            color='k' if v > .6 * vmax else 'w')
            positions = range(first, length)
            names = [f'{t} {pieces[t]}' if named else str(t) for t in positions]
            ax.set_xticks(range(len(names)), names, rotation=90 if named else 0, fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set(xlabel='key position j (attended to)', ylabel='query position i (attending)',
                   title=title + (' — BOS ignored' if first else ''))
            ax.set_facecolor('0.92')
        fig.colorbar(image, ax=axes_row, fraction=.025, pad=.02, label=label)
    fig.suptitle(f'{variant} lr {best_lr:g} final: attention among the first {length} positions (grey: masked, j > i)', y=.92)
    savefig(fig, f'p1a_attention_grid{name}.png')


def report(variant, runs, best_lr):
    print(f'\n{variant} sweep')
    for r in runs:
        print(f"  lr {r['lr']:<8g} {r['status']:<9} val {r['validation_loss']:.4f}")
    for lr in learning_rates(variant):
        for stage in ('early', 'final'):
            label = f'{run_name(variant, lr)}-{stage}'
            if not (RESULTS / f'{label}.json').exists():
                continue
            s = probe(label)
            full, last = s['depth']['full']['depth'], s['depth']['last']['depth']
            sink = s['sink']
            print(f"  {label:<28} probe L {s['depth']['full']['loss']:.3f}  act RMS {full[0]['activation_rms']:.3g}->"
                  f"{full[-1]['activation_rms']:.3g}  grad bottom/top L {full[0]['gradient_rms'] / full[-1]['gradient_rms']:.1f}"
                  f" (entry1/top {full[1]['gradient_rms'] / full[-1]['gradient_rms']:.1f})"
                  f"  L_T {last[0]['gradient_rms'] / last[-1]['gradient_rms']:.1f}"
                  f"  BOS mass max layer {max(np.mean(a['first_late']) for a in s['attention']):.3f}"
                  f"  block-key0 dL {sink['block_key0']['mean'] - sink['normal']['mean']:+.3f}")


def main(variant='d8'):
    runs = manifest_runs(variant)
    best = sweep_figure(variant, runs)
    rates = [r['lr'] for r in runs]
    depth_figure(variant, best['lr'], rates)
    positions_figure(variant, best['lr'])
    attention_figure(variant, best['lr'])
    heads_figure(variant, best['lr'])
    sink_figure(variant, best['lr'], rates)
    report(variant, runs, best['lr'])


if __name__ == '__main__':
    main()
