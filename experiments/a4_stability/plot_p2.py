"""Problem 2(a)-(d): plain and modern RNN sweeps, and their depth/time diagnostics.

    uv run python -m experiments.a4_stability.plot_p2

Reads the manifest, public W&B histories (cache/a4_p1/, shared with P1) and the probe JSONs from p2_probe
(run p2_probe first). Writes to experiments/a4_stability/plots/p2/:
  a_sweep.png               plain 1- and 4-layer sweeps: final val loss vs LR, the unigram plateau in training
  b_time_<model>.png        per layer vs time index: dL/dh_t, dL_T/dh_t, ||h_t||, Pr(|h_t| > .95), by stage or LR
  b_depth.png               plain RNNs across depth: activation and gradient RMS at init / step 100 / final
  b_jacobian.png            rnn4 one-step temporal Jacobian factors: mean tanh' and W_h singular values
  b_escape.png              per-layer W&B RMS-logger series around the rnn4 plateau, escaping vs stuck LRs
  b_flow.png                log10 ||dL_T / dx|| over (depth entry, lag): the joint depth x time picture
  c_sweep.png               modern 4-layer sweep vs the plain models; its training never visits the plateau
  c_depth.png               modern RNN across depth by stage and LR, against the plain rnn4
  c_depth_vs_rnn4.png       plain rnn4 vs modern rnn at their best LRs: depth profiles at init / step 100 / final
  d_summary.png             every variant's sweep, and its loss change next to the transformer counterpart's
  d_time.png                temporal diagnostics of every variant at its best LR vs the modern baseline
  d_depth.png               depth diagnostics of every variant at its best LR vs the modern baseline
  d_lr_edge.png             what happens at the top of the LR range (post-norm plateau, no-pre-norm divergence)
  d_gates.png               gate openness of the three gated variants
  e_similarity.png          modern RNN vs d8: stream RMS and adjacent-layer / across-position cosine (needs p2_context)
  e_removal.png             whether similar layers and tokens are removable: branch skip / mean, token deletion
  f_context.png             useful context (loss vs kept prefix, with and without BOS) against gradient reach
  summary.md               one row per variant: losses, LRs and the scalar diagnostics quoted in the report
"""
import json
from pathlib import Path

import numpy as np

from experiments.a4_stability.plot_p1a import (
    PLOT_DIR, STAGE_COLORS, STAGE_LABELS, history, manifest_runs, mpl, plt, readable_log_axis, savefig, tokens_seen)
from experiments.a4_stability.p2_probe import OUTPUT as RESULTS
from experiments.a4_stability.sweeps import run_name

BASE = 'rnn'
UNIGRAM_ENTROPY = 6.844  # entropy of the val targets' token frequencies (nats): an input-blind predictor
FLOOR = 1e-30            # gradients below this (fp32 underflow reaches exact 0) are drawn at the floor
ABLATIONS = ('rnn-no-residual', 'rnn-no-prenorm', 'rnn-rope', 'rnn-final-norm', 'rnn-postnorm',
             'rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm')
# Transformer counterpart of each RNN change, the component it toggles, and whether the RNN / d8 variant
# removes that component from its baseline (P1 removes RoPE and the final norm from d8; P2 adds them to the RNN).
COUNTERPART = {
    'rnn-no-residual': ('d8-no-residual', 'depth residuals', True, True),
    'rnn-no-prenorm': ('d8-no-prenorm', 'pre-norm', True, True),
    'rnn-rope': ('d8-no-rope', 'RoPE', False, True),
    'rnn-final-norm': ('d8-no-final-norm', 'final output norm', False, True),
    'rnn-postnorm': ('d8-postnorm', 'post-norm instead of pre-norm', False, False),
    'rnn-gated-residual': ('d8-gated-residual', 'gated residuals', False, False),
    'rnn-gated-output': ('d8-gated-attention', 'gated output / attention', False, False),
    'rnn-gated-norm': ('d8-gated-norm', 'gated norm', False, False),
}


def component_effects(v):
    """L(with component) - L(without component) for the RNN and d8, each at its own tuned LR."""
    other, name, rnn_removes, d8_removes = COUNTERPART[v]
    def effect(variant, baseline, removes):
        delta = best(variant)[1] - best(baseline)[1]
        return -delta if removes else delta
    return name, effect(v, BASE, rnn_removes), effect(other, 'd8', d8_removes)
VARIANT_COLORS = dict(zip(ABLATIONS, plt.cm.tab10(np.arange(len(ABLATIONS)))))


def probe(label):
    return json.loads((RESULTS / f'{label}.json').read_text())


def has(label):
    return (RESULTS / f'{label}.json').exists()


def label(variant, lr, stage='final'):
    return f'a4-{variant}-init' if stage == 'init' else f'{run_name(variant, lr)}-{stage}'


def best(variant):
    done = [r for r in manifest_runs(variant) if r['status'] == 'complete']
    r = min(done, key=lambda r: r['validation_loss'])
    return r['lr'], r['validation_loss']


def layers(lab, key, objective='full'):
    return np.array(probe(lab)['depth'][objective][key], float)       # [layer, time]


def depth(lab, key, objective='full'):
    return np.array([row.get(key, np.nan) for row in probe(lab)['depth'][objective]['depth']])


def floor(values):
    return np.maximum(np.asarray(values, float), FLOOR)


def layer_colors(n):
    return plt.cm.viridis(np.linspace(0, .85, n)) if n > 1 else ['tab:blue']


def reach(gradient, fraction):
    """Largest lag k whose ||dL_T/dh_{T-1-k}|| is at least `fraction` of the layer's largest value."""
    g = np.asarray(gradient)[::-1]
    above = np.nonzero(g >= fraction * g.max())[0]
    return int(above.max()) if len(above) else 0


def log_ticks(ax, axis='x'):
    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}'))


# ---------------------------------------------------------------------------------------------- (a)

def sweep_panel(ax, variants, top=4.6, title='supplied five-rate sweeps'):
    for variant, color, style in variants:
        runs = manifest_runs(variant)
        done = [r for r in runs if r['status'] == 'complete']
        b_lr, b_loss = best(variant)
        ax.plot([r['lr'] for r in done], [min(r['validation_loss'], top) for r in done], style, color=color, ms=5,
                label=f'{variant}: best {b_loss:.4f} at lr {b_lr:g}')
        for r in runs:
            if r['status'] != 'complete':
                ax.plot(r['lr'], top, 'x', ms=10, mew=2.2, color=color, clip_on=False)
                ax.annotate('diverged', (r['lr'], top), xytext=(0, 8), textcoords='offset points', ha='center',
                            fontsize=7, color=color)
            elif r['validation_loss'] > top:
                ax.plot(r['lr'], top, '^', ms=8, color=color, clip_on=False)
                ax.annotate(f"{r['validation_loss']:.2f}", (r['lr'], top), xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=7, color=color)
        ax.plot(b_lr, b_loss, '*', ms=14, color=color, zorder=5)
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='final validation loss', ylim=(None, top))
    ax.set_title(title, pad=30)
    rates = sorted({r['lr'] for v, _, _ in variants for r in manifest_runs(v)})
    ax.set_xticks(rates, [f'{lr:g}' for lr in rates], rotation=30)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='lower left')


def train_panel(ax, variant, title, extra=(), steps_max=None):
    runs = manifest_runs(variant)
    colors = dict(zip([r['lr'] for r in runs], plt.cm.viridis(np.linspace(0, .9, len(runs)))))
    for r in runs:
        steps, values = history(r)['train_loss']
        finite = np.isfinite(values)
        keep = finite if steps_max is None else finite & (steps <= steps_max)
        tag = '' if r['status'] == 'complete' else f" ({r['status']} after step {steps[finite][-1]:.0f})"
        ax.plot(tokens_seen(steps[keep]), values[keep], color=colors[r['lr']], lw=.6, alpha=.85,
                label=f"lr {r['lr']:g}{tag}")
        if r['status'] != 'complete':
            ax.plot(tokens_seen(steps[finite][-1]), values[finite][-1], 'x', ms=10, mew=2, color=colors[r['lr']])
    for other, lr, color in extra:
        r = next(r for r in manifest_runs(other) if r['lr'] == lr)
        steps, values = history(r)['train_loss']
        keep = np.isfinite(values) if steps_max is None else np.isfinite(values) & (steps <= steps_max)
        ax.plot(tokens_seen(steps[keep]), values[keep], color=color, lw=.6, alpha=.85, label=f'{other} lr {lr:g}')
    ax.axhline(np.log(4096), color='0.4', ls=':', label='ln 4096 = 8.32: uniform prediction')
    ax.axhline(UNIGRAM_ENTROPY, color='tab:red', ls='--', lw=1, label=f'{UNIGRAM_ENTROPY}: unigram (input-blind) entropy')
    ax.set(xscale='log', xlabel='training tokens (M, log)', ylabel='training loss (every step)', title=title)
    ax.grid(which='major', axis='y', alpha=.3)
    ax.legend(fontsize=7, loc='lower left')


def figure_a():
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.6))
    sweep_panel(axes[0], [('rnn1', 'tab:green', 'o-'), ('rnn4', 'tab:purple', 's-')], top=4.6,
                title='plain RNN sweeps (▲ = above 4.6, value printed)')
    train_panel(axes[1], 'rnn4', 'plain 4-layer RNN: training loss (log tokens)')
    train_panel(axes[2], 'rnn1', 'plain 1-layer RNN: training loss (log tokens)')
    savefig(fig, 'p2/a_sweep.png')


# ---------------------------------------------------------------------------------------------- (b)

def time_figure(rows, name, title):
    """rows: [(row title, probe label)]. Columns are the four Setup statistics against time, one line per layer."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(24, 3.9 * len(rows)), squeeze=False)
    for i, (row_title, lab) in enumerate(rows):
        d = probe(lab)['depth']
        full, last = np.array(d['full']['hidden_gradient_rms']), np.array(d['last']['hidden_gradient_rms'])
        h, sat = np.array(d['full']['hidden_rms']), np.array(d['full']['fraction_abs_gt_095'])
        T = h.shape[1]
        t = np.arange(1, T + 1)
        lag = np.arange(T)
        colors = layer_colors(len(h))
        for l, c in enumerate(colors):
            name_l = f'layer {l + 1}'
            axes[i, 0].plot(t, floor(full[l]), color=c, lw=.8, label=name_l)
            axes[i, 1].plot(lag + 1, floor(last[l][::-1]), color=c, lw=1, label=f'{name_l}: reach(1e-2) {reach(last[l], 1e-2)}')
            axes[i, 2].plot(t, h[l], color=c, lw=1, label=name_l)
            axes[i, 3].plot(t, sat[l], color=c, lw=1, label=name_l)
        loss = f"L={d['full']['loss']:.3f}, L_T={d['last']['loss']:.3f}"
        axes[i, 0].set(xscale='log', yscale='log', ylabel=f'{row_title}\n\n' + r'$\|\partial L/\partial h_t\|_{rms}$',
                       title=f'full-sequence loss gradient ({loss})')
        axes[i, 1].set(xscale='log', yscale='log', ylabel=r'$\|\partial L_T/\partial h_t\|_{rms}$',
                       ylim=(FLOOR / 3, None), title='last-token loss gradient (reach = lag where it falls to 1% of max)')
        axes[i, 2].set(xscale='log', ylabel=r'$\|h_t\|_{rms}$', ylim=(0, 1.05), title='hidden-state RMS')
        axes[i, 3].set(xscale='log', ylabel=r'Pr$(|h_t| > 0.95)$', ylim=(-.02, 1.02), title='tanh saturation fraction')
        for ax in axes[i]:
            ax.grid(alpha=.3)
            log_ticks(ax)
            ax.legend(fontsize=7)
        if i == len(rows) - 1:
            for j in (0, 2, 3):
                axes[i, j].set_xlabel('time index t (log; t = 1 is the first input)')
            axes[i, 1].set_xlabel('lag T - t from the final position (log, +1)')
        axes[i, 1].axhline(FLOOR, color='0.5', ls=':', lw=.8)
    fig.suptitle(title, y=1.0, fontsize=13)
    fig.tight_layout()
    savefig(fig, f'p2/{name}.png')


def stages(variant, lr):
    return [(f'{variant} {STAGE_LABELS[s]}' + (f' (lr {lr:g})' if s != 'init' else ''), label(variant, lr, s))
            for s in ('init', 'early', 'final')]


def jacobian_panel(ax, entries):
    """One-step temporal Jacobian diag(1 - h^2) W_h: mean tanh' and W_h singular values, per layer."""
    width = .8 / len(entries)
    for k, (name, lab, color) in enumerate(entries):
        rec = probe(lab)['recurrence']
        x = np.arange(len(rec)) + (k - (len(entries) - 1) / 2) * width
        ax.bar(x, [r['tanh_derivative_mean'] for r in rec], width, color=color, alpha=.8, label=f"{name}: mean tanh'")
        ax.plot(x, [r['W_h_singular_median'] for r in rec], 'o', color=color, mec='k', ms=6)
        ax.plot(x, [r['W_h_singular_max'] for r in rec], '^', color=color, mec='k', ms=6)
    ax.plot([], [], 'o', color='w', mec='k', label=r'median singular value of $W_h$')
    ax.plot([], [], '^', color='w', mec='k', label=r'largest singular value of $W_h$')
    ax.set_xticks(range(len(rec)), [f'layer {l + 1}' for l in range(len(rec))])
    ax.set(yscale='log', ylabel="mean tanh'(h) (bar);  W_h singular values (markers)",
           title=r"one-step Jacobian $\partial h_t/\partial h_{t-1}$ = diag(1-$h_t^2$) $W_h$")
    readable_log_axis(ax)
    ax.legend(fontsize=7)


def plain_gradients(lab, objective):
    """Embedding gradient, then RMS over time of each layer's true h_t gradient. (A plain layer's stacked output,
    which profile() hooks as the next layer's input, only sees the depth path, not the h_t -> h_t+1 path.)"""
    hidden = np.array(probe(lab)['depth'][objective]['hidden_gradient_rms'])
    return np.concatenate([[depth(lab, 'gradient_rms', objective)[0]], np.sqrt(np.mean(hidden ** 2, 1))])


STAGE_STYLES = {'init': ':', 'early': '--', 'final': '-'}


def decade_log_axis(ax):
    """One labeled tick per decade (10^k), unlabeled minor ticks, grid on the decades only."""
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=30))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(which='major', axis='y', alpha=.4)


def plain_depth_figure(models):
    """models: [(panel title, [(run name, variant, lr, color)])]. One column per model, one row per statistic;
    color = run, line style = checkpoint stage."""
    rows = [('activation RMS', lambda lab: depth(lab, 'activation_rms'), False),
            (r'$\|\partial L/\partial h\|_{rms}$', lambda lab: plain_gradients(lab, 'full'), True),
            (r'$\|\partial L_T/\partial h\|_{rms}$', lambda lab: plain_gradients(lab, 'last'), True)]
    # Width = x-range of each panel (entries - 0.3, see set_xlim), so one depth step is equally wide in every column.
    widths = [len(depth(label(*runs[0][1:3]), 'activation_rms')) - .3 for _, runs in models]
    style = {'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 13,
             'ytick.labelsize': 13, 'legend.fontsize': 12}
    with plt.rc_context(style):
        fig, axes = plt.subplots(len(rows), len(models), figsize=(2.5 + 1.25 * sum(widths), 3.3 * len(rows)),
                                 sharey='row', squeeze=False, gridspec_kw=dict(width_ratios=widths))
        for j, (title, runs) in enumerate(models):
            for i, (ylabel, values, log) in enumerate(rows):
                ax = axes[i, j]
                for k, (name, variant, lr, color) in enumerate(runs):
                    for stage, ls in STAGE_STYLES.items():
                        y = values(label(variant, lr, stage))
                        if stage == 'init':  # runs of one model share this init: interleave the dots so each color shows
                            ls = (2 * k, (1, 2 * len(runs) + 1))
                        ax.plot(np.arange(len(y)), floor(y) if log else y, linestyle=ls, marker='o', ms=5, lw=2,
                                color=color)
                entries = len(values(label(*runs[0][1:3])))
                ax.set_xticks(range(entries), (['embed'] + [f'h{l}' for l in range(1, entries)]))
                ax.set_xlim(-.35, entries - .65)
                if log:
                    ax.set_yscale('log')
                    decade_log_axis(ax)
                ax.grid(alpha=.3)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == 0:
                    ax.set_title(title)
                if i == len(rows) - 1:
                    ax.set_xlabel('depth')
        handles = [mpl.lines.Line2D([], [], color=color, lw=3, label=name) for _, runs in models
                   for name, _, _, color in runs]
        handles += [mpl.lines.Line2D([], [], color='0.3', ls=ls, marker='o', ms=5, lw=2, label=STAGE_LABELS[stage])
                    for stage, ls in STAGE_STYLES.items()]
        fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, .965), frameon=False)
        fig.suptitle("plain RNNs across depth (embedding, then each layer's $h_t$; RMS over batch, width and t)",
                     fontsize=15, y=.995)
        fig.tight_layout(rect=(0, 0, 1, .96))
        savefig(fig, 'p2/b_depth.png')


def figure_b():
    (lr1, _), (lr4, _) = best('rnn1'), best('rnn4')
    time_figure(stages('rnn1', lr1), 'b_time_rnn1', f'plain 1-layer RNN (best lr {lr1:g}) by checkpoint stage')
    time_figure(stages('rnn4', lr4), 'b_time_rnn4', f'plain 4-layer RNN (best lr {lr4:g}) by checkpoint stage')
    time_figure([(f'rnn4 final, lr {lr:g}', label('rnn4', lr)) for lr in (.0005, .001, .002, .004)],
                'b_time_rnn4_lr', 'plain 4-layer RNN, final checkpoints: lr <= 0.001 trains, lr >= 0.002 stays on the unigram plateau')
    plain_depth_figure([(f'1-layer RNN', [(f'rnn1 lr {lr1:g} (best)', 'rnn1', lr1, 'tab:green')]),
                        (f'4-layer RNN', [(f'rnn4 lr {lr4:g} (best)', 'rnn4', lr4, 'tab:blue'),
                                          ('rnn4 lr 0.004 (unigram plateau)', 'rnn4', .004, 'tab:red')])])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    jacobian_panel(ax, [('rnn4 init', label('rnn4', lr4, 'init'), STAGE_COLORS['init']),
                        (f'rnn4 lr {lr4:g} step 100 (inside the plateau)', label('rnn4', lr4, 'early'), STAGE_COLORS['early']),
                        (f'rnn4 lr {lr4:g} final', label('rnn4', lr4), STAGE_COLORS['final']),
                        ('rnn4 lr 0.004 final (never left the plateau)', label('rnn4', .004), 'tab:red')])
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1))
    savefig(fig, 'p2/b_jacobian.png')
    figure_b_escape()
    lr = best(BASE)[0]
    flow_figure([(f'rnn1 lr {lr1:g} final', label('rnn1', lr1), ['embed', 'h1']),
                 (f'rnn4 lr {lr4:g} final', label('rnn4', lr4), ['embed', 'h1', 'h2', 'h3', 'h4']),
                 ('rnn4 lr 0.004 final (plateau)', label('rnn4', .004), ['embed', 'h1', 'h2', 'h3', 'h4']),
                 (f'modern rnn lr {lr:g} final', label(BASE, lr), modern_entries())], 'b_flow')


RMS_CACHE = Path('cache/a4_p2')
RMS_KEYS = ([f'stack.{l}/activation' for l in range(4)] +
            [f'stack.{l}.cell.{m}/{q}' for l in range(4) for m in ('hidden', 'input') for q in ('gradient', 'parameter')] +
            ['embedding/activation', 'embedding/gradient', 'lm_head/gradient', 'lm_head/parameter', 'global/gradient'])


def module_rms(run):
    """{key: (steps, values)} of the RMS logger's per-module series (sampled every 100 steps) for one public run."""
    path = RMS_CACHE / f"{run['run_name']}-rms.json"
    if not path.exists():
        import wandb
        from experiments.a4_stability.sweeps import PUBLIC_PROJECT
        remote = wandb.Api(timeout=120).run(f"{PUBLIC_PROJECT}/{run['url'].rstrip('/').split('/')[-1]}")
        keys = [k for k in ('logging/rms/' + k for k in RMS_KEYS) if k in remote.summary.keys()]
        rows = list(remote.scan_history(keys=['optimizer_step'] + keys))
        RMS_CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: [[r['optimizer_step'], r[k]] for r in rows if r.get(k) is not None] for k in keys}))
    data = json.loads(path.read_text())
    return {k.removeprefix('logging/rms/'): tuple(np.array(v, float).T) for k, v in data.items()}


def plateau_steps(run, level=6.9, band=.12, smooth=21):
    """First and last step whose smoothed training loss is within `band` of the plateau level (None if absent)."""
    steps, values = history(run)['train_loss']
    smoothed = np.convolve(np.nan_to_num(values, nan=99.), np.ones(smooth) / smooth, mode='same')
    inside = np.nonzero((np.abs(smoothed - level) < band) & (steps > 10))[0]
    return (steps[inside[0]], steps[inside[-1]]) if len(inside) >= 15 else None


def figure_b_escape(variant='rnn4', rates=(.00025, .001, .002, .004), last_step=1000, zero=1e-10):
    """Per-layer RMS-logger series around the plateau: training loss, input-weight gradient, layer output RMS."""
    runs = {r['lr']: r for r in manifest_runs(variant)}
    colors = layer_colors(4)
    style = {'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 12,
             'ytick.labelsize': 12, 'legend.fontsize': 11}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, len(rates), figsize=(4.6 * len(rates), 11), sharex=True, sharey='row')
        for j, lr in enumerate(rates):
            run = runs[lr]
            rms = module_rms(run)
            window = plateau_steps(run)
            steps, values = history(run)['train_loss']
            keep = steps <= last_step
            ax = axes[0, j]
            ax.plot(steps[keep], values[keep], color='0.3', lw=.7)
            ax.axhline(UNIGRAM_ENTROPY, color='tab:red', ls='--', lw=1)
            ax.set_title(f'lr {lr:g}: ' + ('escapes' if window and window[1] < last_step else 'stays on the plateau'))
            for ax in axes[:, j]:
                ax.axvline(93, color='0.5', ls=':', lw=1)
                if window:
                    ax.axvspan(window[0], min(window[1], last_step), color='tab:red', alpha=.08)
                ax.grid(alpha=.3)
            for l, c in enumerate(colors):
                for row, key in ((1, f'stack.{l}.cell.input/gradient'), (2, f'stack.{l}/activation')):
                    x, y = rms[key]
                    keep = x <= last_step
                    x, y = x[keep], y[keep]
                    if row == 1:
                        axes[row, j].plot(x, np.maximum(y, zero), 'o-', color=c, ms=5, lw=2, label=f'layer {l + 1}')
                        axes[row, j].plot(x[y == 0], np.full((y == 0).sum(), zero), 'v', color=c, ms=9)
                    else:
                        axes[row, j].plot(x, y, 'o-', color=c, ms=5, lw=2, label=f'layer {l + 1}')
            x, y = rms['embedding/gradient']
            keep = x <= last_step
            axes[1, j].plot(x[keep], np.maximum(y[keep], zero), 's--', color='k', ms=4, lw=1.4, label='embedding')
            axes[1, j].set_yscale('log')
            decade_log_axis(axes[1, j])
            axes[2, j].set_xlabel('optimizer step')
        axes[0, 0].set(ylabel='training loss (every step)', ylim=(5, 8.6))
        axes[1, 0].set_ylabel(r'gradient RMS of $W_x$' + '\n(▼ = exactly 0)')
        axes[2, 0].set(ylabel='layer output RMS, $\\|h\\|_{rms}$', ylim=(0, 1.05))
        axes[1, 0].set_ylim(zero / 2, 3e-3)
        handles = [mpl.lines.Line2D([], [], color=c, marker='o', lw=2, label=f'layer {l + 1}') for l, c in enumerate(colors)]
        handles += [mpl.lines.Line2D([], [], color='k', marker='s', ls='--', label='embedding'),
                    mpl.lines.Line2D([], [], color='tab:red', ls='--', label=f'unigram entropy {UNIGRAM_ENTROPY}'),
                    mpl.lines.Line2D([], [], color='0.5', ls=':', label='end of warmup (step 93)'),
                    mpl.patches.Patch(color='tab:red', alpha=.15, label='plateau (training loss within 0.12 of 6.9)')]
        fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(.5, .97), frameon=False)
        fig.suptitle(f'plain 4-layer RNN around the plateau: W&B RMS logger, sampled every 100 steps', y=.997)
        fig.tight_layout(rect=(0, 0, 1, .935))
        savefig(fig, 'p2/b_escape.png')


def modern_entries():
    return [f'{kind}{l}' for l in range(1, 5) for kind in ('rec in ', 'mlp in ')] + ['out']


def flow_figure(models, name):
    """log10 ||dL_T/dx_{j,t}|| for every depth-stream entry j and lag: how far back and how deep the final
    prediction reaches."""
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    norm = mpl.colors.Normalize(-12, -2)
    for ax, (title, lab, names) in zip(axes[0], models):
        g = np.array(probe(lab)['positions']['gradient_last'])
        if names[1] == 'h1':  # plain RNN: replace the stacked outputs with the true h_t nodes (see plain_gradients)
            g = np.concatenate([g[:1], probe(lab)['depth']['last']['hidden_gradient_rms']])
        g = g[:, ::-1]                                                                 # [entry, lag]
        lags = np.arange(g.shape[1])
        edges = np.concatenate([[.5], lags + 1.5])
        mesh = ax.pcolormesh(edges, np.arange(len(g) + 1) - .5, np.log10(np.maximum(g, FLOOR)), cmap='magma',
                             norm=norm, shading='flat')
        ax.set(xscale='log', title=title, xlabel='lag T - t (log, +1)')
        ax.set_yticks(range(len(names)), names, fontsize=8)
        log_ticks(ax)
    axes[0, 0].set_ylabel('depth-stream entry (bottom = embedding)')
    fig.colorbar(mesh, ax=axes[0], label=r'$\log_{10}\|\partial L_T/\partial x_{j,t}\|_{rms}$ (clipped at -12)')
    savefig(fig, f'p2/{name}.png')


# ---------------------------------------------------------------------------------------------- (c)

def modern_depth_panels(axes, entries):
    """entries: [(name, label, color, style)] of 4-block modern RNNs (9 stream entries)."""
    for name, lab, color, style in entries:
        act = depth(lab, 'activation_rms')
        gl, gt = depth(lab, 'gradient_rms'), depth(lab, 'gradient_rms', 'last')
        ratio, cos = depth(lab, 'raw_branch_input_ratio'), depth(lab, 'adjacent_cosine')
        x = np.arange(len(act))
        axes[0].plot(x, act, style, color=color, ms=4, label=name)
        axes[1].plot(x, floor(gl), style, color=color, ms=4, label=f'{name}: top/bottom {gl[-1] / gl[0]:.2g}')
        axes[2].plot(x, floor(gt), style, color=color, ms=4, label=f'{name}: top/bottom {gt[-1] / gt[0]:.2g}')
        axes[3].plot(x[:-1], ratio[:-1], style, color=color, ms=4, label=name)
        axes[4].plot(x[1:], cos[1:], style, color=color, ms=4, label=name)
    for ax, ylabel, title, log in [
            (axes[0], 'activation RMS', 'depth-stream RMS', True),
            (axes[1], r'$\|\partial L/\partial x\|_{rms}$', 'gradient RMS for L', True),
            (axes[2], r'$\|\partial L_T/\partial x\|_{rms}$', 'gradient RMS for $L_T$', True),
            (axes[3], r'$\|F(N(x))\|/\|x\|$', 'raw branch output / branch input', True),
            (axes[4], 'cosine(x_j, x_{j-1})', 'adjacent-depth cosine', False)]:
        ax.set_xticks(range(0, 9, 2), ['L1', 'L2', 'L3', 'L4', 'out'])
        ax.set_xticks(range(9), minor=True)
        ax.set(ylabel=ylabel, title=title, xlabel='depth: recurrent input of block Lj (minor tick = MLP input)')
        if log:
            ax.set_yscale('log')
            readable_log_axis(ax)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=6)
    axes[2].legend(fontsize=6)


def hidden_rms(lab):
    """RMS over batch, width and time of each recurrent layer's tanh output h_t."""
    return np.sqrt(np.mean(np.square(layers(lab, 'hidden_rms')), 1))


def figure_c_depth_vs_plain():
    """Plain rnn4 vs modern rnn at their best LRs. Per recurrent layer: tanh output RMS and Pr(|h_t| > .95) (bars;
    hatch = stage). Then gradient RMS on one depth axis (block boundaries; the modern RNN's MLP-branch inputs at the
    half steps). Color = run, line style = stage."""
    (lr4, _), (lr, _) = best('rnn4'), best(BASE)
    runs = [(f'plain rnn4, lr {lr4:g}', 'rnn4', lr4, 'tab:purple', 1.),
            (f'modern rnn, lr {lr:g}', BASE, lr, 'k', .5)]
    gradients = [(r'$\|\partial L/\partial x\|_{rms}$', 'full', 'gradient RMS for L'),
                 (r'$\|\partial L_T/\partial x\|_{rms}$', 'last', 'gradient RMS for $L_T$')]
    style = {'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12,
             'ytick.labelsize': 13, 'legend.fontsize': 12}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), gridspec_kw=dict(width_ratios=(.8, .9, 1, 1)))
        ax = axes[0]
        for name, variant, rate, color, _ in runs:
            for stage, ls in STAGE_STYLES.items():
                y = hidden_rms(label(variant, rate, stage))
                ax.plot(np.arange(1, len(y) + 1), y, linestyle=ls, marker='o', ms=5, lw=2, color=color)
        ax.set(ylim=(0, 1.02), xlim=(.7, 4.3), ylabel=r'$\|h_t\|_{rms}$', xlabel='recurrent layer',
               title='tanh output of each recurrent layer')
        ax.set_xticks(range(1, 5), [f'L{l}' for l in range(1, 5)])
        ax.grid(alpha=.3)
        ax = axes[1]
        hatches = {'init': '...', 'early': '///', 'final': None}
        width = .8 / (len(runs) * len(STAGE_STYLES))
        for k, (name, variant, rate, color, _) in enumerate(runs):
            for m, stage in enumerate(STAGE_STYLES):
                y = layers(label(variant, rate, stage), 'fraction_abs_gt_095').mean(1)
                x = np.arange(1, len(y) + 1) + (k * len(STAGE_STYLES) + m + .5) * width - .4
                solid = hatches[stage] is None
                ax.bar(x, y, width, facecolor=color if solid else 'white', edgecolor=color, hatch=hatches[stage],
                       lw=1.2)
        ax.set(ylim=(0, 1), xlim=(.5, 4.5), ylabel=r'Pr$(|h_t| > 0.95)$', xlabel='recurrent layer',
               title='tanh saturation fraction\n(missing bars are ≈ 0: plain init, plain final L1)')
        ax.set_xticks(range(1, 5), [f'L{l}' for l in range(1, 5)])
        ax.grid(axis='y', alpha=.3)
        for ax, (ylabel, objective, title) in zip(axes[2:], gradients):
            for name, variant, rate, color, step in runs:
                for stage, ls in STAGE_STYLES.items():
                    lab = label(variant, rate, stage)
                    y = plain_gradients(lab, objective) if variant == 'rnn4' else depth(lab, 'gradient_rms', objective)
                    ax.plot(np.arange(len(y)) * step, floor(y), linestyle=ls, marker='o', ms=5, lw=2, color=color)
            ax.set_yscale('log')
            decade_log_axis(ax)
            ax.set(ylabel=ylabel, title=title, xlim=(-.25, 4.25), xlabel='depth (minor ticks: modern MLP inputs)')
            ax.set_xticks(range(5), ['embed', 'in L2', 'in L3', 'in L4', 'out'])
            ax.set_xticks(np.arange(.5, 4, 1), minor=True)
            ax.grid(alpha=.3)
        handles = [mpl.lines.Line2D([], [], color=color, lw=3, label=name) for name, _, _, color, _ in runs]
        handles += [mpl.lines.Line2D([], [], color='0.3', ls=ls, marker='o', ms=5, lw=2, label=STAGE_LABELS[stage])
                    for stage, ls in STAGE_STYLES.items()]
        handles += [mpl.patches.Patch(facecolor='0.3' if hatches[stage] is None else 'white', edgecolor='0.3',
                                      hatch=hatches[stage], label=f'{STAGE_LABELS[stage]} (bars)')
                    for stage in STAGE_STYLES]
        fig.legend(handles=handles, loc='upper center', ncol=8, bbox_to_anchor=(.5, .945), frameon=False)
        fig.suptitle('plain vs modern 4-layer RNN at their best LR (gradients: plain = embedding then each '
                     "layer's $h_t$; modern = residual stream; RMS over batch, width and t)", fontsize=13, y=.995)
        fig.tight_layout(rect=(0, 0, 1, .915))
        savefig(fig, 'p2/c_depth_vs_rnn4.png')


def figure_c():
    lr, _ = best(BASE)
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.6))
    sweep_panel(axes[0], [('rnn1', 'tab:green', 'o--'), ('rnn4', 'tab:purple', 's--'), (BASE, 'k', 'D-')], top=4.6,
                title='modern 4-layer RNN vs the plain models (▲ = above 4.6)')
    train_panel(axes[1], BASE, 'modern 4-layer RNN: training loss (log tokens)',
                extra=[('rnn4', best('rnn4')[0], 'tab:purple')])
    ax = axes[2]
    runs = manifest_runs(BASE)
    colors = dict(zip([r['lr'] for r in runs], plt.cm.viridis(np.linspace(0, .9, len(runs)))))
    for r in runs:
        steps, values = history(r)['val_loss']
        ax.plot(tokens_seen(steps), values, color=colors[r['lr']], lw=1.2, label=f"lr {r['lr']:g} (final {r['validation_loss']:.4f})")
    ax.set(xlabel='training tokens (M)', ylabel='validation loss', ylim=(3.45, 4.5),
           title='modern RNN validation loss: high LR leads early, low LR catches up during decay')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    savefig(fig, 'p2/c_sweep.png')
    time_figure(stages(BASE, lr), 'c_time_rnn', f'modern 4-layer RNN (best lr {lr:g}) by checkpoint stage')
    time_figure([(f'rnn final, lr {r:g}', label(BASE, r)) for r in (.00025, .001, .002, .004)],
                'c_time_rnn_lr', 'modern 4-layer RNN, final checkpoints across the sweep')
    time_figure([(f"plain rnn4 final (lr {best('rnn4')[0]:g})", label('rnn4', best('rnn4')[0])),
                 (f'modern rnn final (lr {lr:g})', label(BASE, lr))],
                'c_time_plain_vs_modern', 'plain vs modern 4-layer RNN at their best LR')
    fig, axes = plt.subplots(1, 6, figsize=(34, 5.2))
    entries = [(f'lr {lr:g} {STAGE_LABELS[s]}', label(BASE, lr, s), STAGE_COLORS[s], 'o-')
               for s in ('init', 'early', 'final')]
    entries += [(f'lr {r:g} final', label(BASE, r), c, 'o--')
                for r, c in ((.00025, 'tab:cyan'), (.004, 'tab:red'))]
    modern_depth_panels(axes[:5], entries)
    jacobian_panel(axes[5], [(f'modern lr {lr:g} final', label(BASE, lr), 'k'),
                             (f"plain rnn4 lr {best('rnn4')[0]:g} final", label('rnn4', best('rnn4')[0]), 'tab:purple'),
                             ('modern init', label(BASE, lr, 'init'), '0.6')])
    savefig(fig, 'p2/c_depth.png')
    figure_c_depth_vs_plain()


# ---------------------------------------------------------------------------------------------- (d)

def figure_d_summary():
    fig, axes = plt.subplots(1, 2, figsize=(24, 6.8), gridspec_kw=dict(width_ratios=(1, 1.1), wspace=.75))
    ax = axes[0]
    ref = best(BASE)[1]
    for v in (BASE,) + ABLATIONS:
        runs = manifest_runs(v)
        done = [r for r in runs if r['status'] == 'complete']
        color = 'k' if v == BASE else VARIANT_COLORS[v]
        top = 3.9
        ax.plot([r['lr'] for r in done], [min(r['validation_loss'], top) for r in done], 'o-', ms=4, color=color,
                lw=2.2 if v == BASE else 1.2, label=f'{v} (best {best(v)[1]:.4f} at lr {best(v)[0]:g})')
        for r in runs:
            if r['status'] != 'complete':
                ax.plot(r['lr'], top, 'x', ms=10, mew=2.2, color=color, clip_on=False)
                ax.annotate('diverged', (r['lr'], top), xytext=(0, 8), textcoords='offset points', ha='center',
                            fontsize=7, color=color)
            elif r['validation_loss'] > top:
                ax.plot(r['lr'], top, '^', ms=8, color=color, clip_on=False)
                ax.annotate(f"{r['validation_loss']:.2f}", (r['lr'], top), xytext=(6, -4), textcoords='offset points',
                            fontsize=7, color=color)
    ax.axhline(ref, color='k', ls=':', lw=.8)
    rates = [r['lr'] for r in manifest_runs(BASE)]
    ax.set(xscale='log', ylim=(3.46, 3.9), xlabel='peak learning rate', ylabel='final validation loss',
           title='modern RNN ablations: supplied sweeps (▲ = above 3.9, value printed; x = diverged)')
    ax.set_xticks(rates, [f'{r:g}' for r in rates])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax = axes[1]
    d8 = best('d8')[1]
    rows = [component_effects(v) for v in ABLATIONS]
    y = np.arange(len(rows))
    limit = .15
    ax.barh(y - .2, [np.clip(r[1], -limit, limit) for r in rows], .38, color=[VARIANT_COLORS[v] for v in ABLATIONS],
            label='modern 4-layer RNN')
    ax.barh(y + .2, [np.clip(r[2], -limit, limit) for r in rows], .38, color='0.75', hatch='//', edgecolor='0.4',
            label='d8 transformer')
    for i, (_, a, b) in enumerate(rows):
        for dy, value in ((-.2, a), (.2, b)):
            clipped = abs(value) > limit
            ax.text(np.clip(value, -limit, limit) + (.004 if value >= 0 else -.004), i + dy,
                    f'{value:+.3f}' + (' (clipped)' if clipped else ''), va='center',
                    ha='left' if value >= 0 else 'right', fontsize=9.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    for i in y[1:]:
        ax.axhline(i - .5, color='0.85', lw=.8)
    ax.axvline(0, color='k', lw=.8)
    ax.axvspan(-.01, .01, color='0.9', zorder=0, label='±0.01: inside LR-grid / seed resolution')
    low = min(min(r[1], r[2]) for r in rows)
    ax.set(xlim=(max(low, -limit) - .09, limit + .1), ylim=(len(rows) - .5, -.5),
           xlabel='L(with component) − L(without component), best val loss (nats; < 0 = component helps)',
           title=f'effect of each component, each model at its own tuned LR\n(RNN baseline {ref:.4f}, d8 baseline {d8:.4f})')
    ax.grid(axis='x', alpha=.3)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.11), ncol=3, frameon=False)
    savefig(fig, 'p2/d_summary.png')


def figure_d_time():
    """Rows = ablations; columns = last-token gradient vs lag, full-loss gradient, ||h_t||, saturation.
    Each panel shows the top and bottom recurrent layer of the variant (color) and of the baseline (black)."""
    base = label(BASE, best(BASE)[0])
    fig, axes = plt.subplots(len(ABLATIONS), 4, figsize=(24, 3.3 * len(ABLATIONS)))
    for i, v in enumerate(ABLATIONS):
        lab = label(v, best(v)[0])
        for name, source, color in ((f'baseline lr {best(BASE)[0]:g}', base, 'k'), (f'{v} lr {best(v)[0]:g}', lab, VARIANT_COLORS[v])):
            d = probe(source)['depth']
            full, last = np.array(d['full']['hidden_gradient_rms']), np.array(d['last']['hidden_gradient_rms'])
            h, sat = np.array(d['full']['hidden_rms']), np.array(d['full']['fraction_abs_gt_095'])
            t = np.arange(1, h.shape[1] + 1)
            for l, ls in ((0, ':'), (3, '-')):
                tag = f'{name} layer {l + 1}'
                axes[i, 0].plot(t, floor(last[l][::-1]), ls, color=color, lw=1,
                                label=f'{tag}: reach {reach(last[l], 1e-2)}')
                axes[i, 1].plot(t, floor(full[l]), ls, color=color, lw=.8, label=tag)
                axes[i, 2].plot(t, h[l], ls, color=color, lw=1, label=tag)
                axes[i, 3].plot(t, sat[l], ls, color=color, lw=1, label=tag)
        axes[i, 0].set(xscale='log', yscale='log', ylim=(1e-14, 1e-1), ylabel=f'{v}\n\n' + r'$\|\partial L_T/\partial h\|$')
        axes[i, 1].set(xscale='log', yscale='log', ylabel=r'$\|\partial L/\partial h_t\|$')
        axes[i, 2].set(xscale='log', ylim=(0, 1.02), ylabel=r'$\|h_t\|_{rms}$')
        axes[i, 3].set(xscale='log', ylim=(-.02, 1.02), ylabel=r'Pr$(|h_t|>.95)$')
        for ax in axes[i]:
            ax.grid(alpha=.3)
            log_ticks(ax)
        axes[i, 0].legend(fontsize=6, loc='lower left')
        axes[i, 3].legend(fontsize=6, loc='lower right')
    for ax, title in zip(axes[0], ('last-token gradient vs lag (reach = lag at 1% of max)', 'full-loss gradient vs t',
                                   'hidden-state RMS vs t', 'tanh saturation vs t')):
        ax.set_title(title)
    axes[-1, 0].set_xlabel('lag T - t (log, +1)')
    for ax in axes[-1, 1:]:
        ax.set_xlabel('time index t (log)')
    fig.suptitle('ablations at their best LR, final checkpoint: dotted = layer 1, solid = layer 4; black = modern baseline',
                 y=1.0, fontsize=13)
    fig.tight_layout()
    savefig(fig, 'p2/d_time.png')


def figure_d_depth():
    lr = best(BASE)[0]
    fig, axes = plt.subplots(1, 5, figsize=(30, 5.6))
    entries = [(f'baseline lr {lr:g}', label(BASE, lr), 'k', 'o-')]
    entries += [(f'{v} lr {best(v)[0]:g}', label(v, best(v)[0]), VARIANT_COLORS[v], 'o-') for v in ABLATIONS]
    modern_depth_panels(axes, entries)
    for ax in axes[1:3]:
        ax.legend(fontsize=6)
    fig.suptitle('depth diagnostics of every ablation at its best LR (final checkpoint)', fontsize=13)
    savefig(fig, 'p2/d_depth.png')


def figure_d_lr_edge():
    """Top of the LR range: post-norm and no-pre-norm against the baseline, in training and in the probes."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 10.5))
    train_panel(axes[0, 0], 'rnn-postnorm', 'post-norm: training loss (lr >= 0.002 visits the unigram plateau)')
    train_panel(axes[0, 1], 'rnn-no-prenorm', 'no pre-norm: training loss (lr 0.004 diverges)')
    train_panel(axes[0, 2], BASE, 'modern baseline: training loss')
    # Step-100 probes: is the state frozen (temporal cosine near 1) and saturated?
    ax = axes[1, 0]
    groups = [('rnn4', 'goldenrod'), (BASE, 'k'), ('rnn-postnorm', VARIANT_COLORS['rnn-postnorm']),
              ('rnn-no-prenorm', VARIANT_COLORS['rnn-no-prenorm'])]
    for v, color in groups:
        for stage, marker in (('early', 'o--'), ('final', 's-')):
            rates, values = [], []
            for r in manifest_runs(v):
                if has(label(v, r['lr'], stage)):
                    rates.append(r['lr'])
                    values.append(np.mean([c['1'] for c in probe(label(v, r['lr'], stage))['depth']['temporal_cosine']]))
            ax.plot(rates, values, marker, color=color, ms=5, label=f'{v} {STAGE_LABELS[stage]}')
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='cosine(h_t, h_{t-1}), mean over layers',
           title='frozen state: lag-1 temporal cosine of h (1 = h does not depend on the input)', ylim=(0, 1.05))
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    for v, color in groups:
        for stage, marker in (('early', 'o--'), ('final', 's-')):
            rates, values = [], []
            for r in manifest_runs(v):
                if has(label(v, r['lr'], stage)):
                    rates.append(r['lr'])
                    values.append(np.mean(probe(label(v, r['lr'], stage))['depth']['full']['fraction_abs_gt_095']))
            ax.plot(rates, values, marker, color=color, ms=5, label=f'{v} {STAGE_LABELS[stage]}')
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='Pr(|h| > .95), mean over layers and t',
           title='tanh saturation by LR, step 100 (dashed) and final (solid)', ylim=(0, 1))
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    ax = axes[1, 2]
    for v, color in groups:
        for stage, marker in (('early', 'o--'), ('final', 's-')):
            rates, values = [], []
            for r in manifest_runs(v):
                if has(label(v, r['lr'], stage)):
                    rates.append(r['lr'])
                    values.append(probe(label(v, r['lr'], stage))['output']['output_entropy'])
            ax.plot(rates, values, marker, color=color, ms=5, label=f'{v} {STAGE_LABELS[stage]}')
    ax.axhline(UNIGRAM_ENTROPY, color='tab:red', ls='--', lw=1, label='unigram entropy')
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='mean entropy of the predicted distribution (nats)',
           title='predicted-distribution entropy: the plateau predicts the unigram')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    for ax in axes[1]:
        rates = [r['lr'] for r in manifest_runs(BASE)]
        ax.set_xticks(rates, [f'{r:g}' for r in rates])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    savefig(fig, 'p2/d_lr_edge.png')


def figure_d_gates():
    fig, axes = plt.subplots(1, 3, figsize=(21, 4.8))
    for ax, v in zip(axes, ('rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm')):
        lr = best(v)[0]
        for stage, color in (('early', STAGE_COLORS['early']), ('final', STAGE_COLORS['final'])):
            gates = probe(label(v, lr, stage))['output']['gates']
            names = list(gates)
            x = np.arange(len(names))
            ax.plot(x, [gates[n]['rest'] for n in names], 'o-', color=color, label=f'mean gate, {STAGE_LABELS[stage]}')
            ax.plot(x, [gates[n]['closed_rest'] for n in names], 'x--', color=color,
                    label=f'fraction of gates < 0.1, {STAGE_LABELS[stage]}')
        short = [f"L{int(n.split('.')[1]) + 1} {'rec' if '.recurrent' in n else 'mlp'}" for n in names]
        ax.set_xticks(x, short, rotation=45, fontsize=7)
        ax.set(ylim=(0, 1), ylabel='sigmoid gate value (non-BOS positions)', title=f'{v} (lr {lr:g})')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    savefig(fig, 'p2/d_gates.png')


def frozen_layers(lab, threshold=.95):
    """Recurrent layers whose state barely moves with the input: lag-1 cosine(h_t, h_{t-1}) above threshold."""
    return sum(c['1'] > threshold for c in probe(lab)['depth']['temporal_cosine'])


def figure_d_scatter():
    """Across every finite final checkpoint of the modern family: loss against frozen layers, saturation and reach."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.8))
    rows = []
    for v in (BASE,) + ABLATIONS:
        color = 'k' if v == BASE else VARIANT_COLORS[v]
        for r in manifest_runs(v):
            if r['status'] != 'complete' or not has(label(v, r['lr'])):
                continue
            lab = label(v, r['lr'])
            d = probe(lab)['depth']
            last = d['last']['hidden_gradient_rms']
            stats = (frozen_layers(lab), np.mean(d['full']['fraction_abs_gt_095']),
                     np.median([reach(g, 1e-2) for g in last]))
            rows.append((r['validation_loss'], *stats))
            size = 20 + 25 * np.log2(r['lr'] / .00025)
            jitter = (np.log2(r['lr'] / .001)) * .05
            for ax, x in zip(axes, (stats[0] + jitter, stats[1], stats[2] + 1)):
                ax.scatter(x, min(r['validation_loss'], 4.0), s=size, color=color, alpha=.8,
                           label=v if r['lr'] == manifest_runs(v)[0]['lr'] else None)
    rows = np.array(rows)
    keep = rows[:, 0] < 4.0
    for ax, j, xlabel, title in [
            (axes[0], 1, 'recurrent layers with lag-1 cosine(h_t, h_{t-1}) > 0.95 (of 4; jittered by LR)', 'frozen (input-independent) layers'),
            (axes[1], 2, 'Pr(|h| > .95), mean over layers and t', 'tanh saturation'),
            (axes[2], 3, 'layer-1 last-token reach at 1% of max (lags, log, +1)', 'bottom-layer gradient reach')]:
        x = rows[keep, j]
        rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(rows[keep, 0])))[0, 1]
        ax.set(xlabel=xlabel, ylabel='final validation loss (clipped at 4.0)',
               title=f'{title}\nSpearman rho = {rho:+.2f} over the {keep.sum()} runs below 4.0')
        ax.grid(alpha=.3)
    axes[2].set_xscale('log')
    log_ticks(axes[2])
    axes[0].legend(fontsize=7, title='marker size = LR (0.00025 ... 0.004)', title_fontsize=7)
    fig.suptitle('modern RNN family, every finite final checkpoint: which diagnostics track loss? (correlation, not cause)',
                 fontsize=13, y=1.04)
    savefig(fig, 'p2/d_scatter.png')


# ---------------------------------------------------------------------------------------------- (e, f)

# (title, run, color); the first two are the competitive pair compared in (e), rnn4 is a reference in (f).
CONTEXT_MODELS = (('modern RNN (lr 0.002)', 'a4-rnn-lr0.002', 'tab:blue'),
                  ('d8 transformer (lr 0.003)', 'a4-d8-lr0.003', 'tab:red'),
                  ('plain rnn4 (lr 0.001)', 'a4-rnn4-lr0.001', 'goldenrod'))
POSITION_GROUPS = (('BOS (t = 0)', slice(0, 1), ':'), ('t = 1-15', slice(1, 16), '--'),
                   ('t = 16-255', slice(16, 256), '-.'), ('t = 256-1022', slice(256, None), '-'))
NEAR = 32  # targets s+2 .. s+1+NEAR count as "near" a deleted token s


def context_probe(run):
    from experiments.a4_stability.p2_context import OUTPUT as CONTEXT
    return json.loads((CONTEXT / f'context-{run}.json').read_text())


def stream_names(run):
    """Depth-stream entries (branch inputs, then the output-norm input) and the branch between j and j+1."""
    if '-d8' in run:
        return ([f'{kind} in {l}' for l in range(1, 9) for kind in ('attn', 'mlp')] + ['out'],
                [f'{kind}{l}' for l in range(1, 9) for kind in ('A', 'M')])
    return modern_entries(), [f'{kind}{l}' for l in range(1, 5) for kind in ('R', 'M')]


def figure_e_similarity():
    """Rows = the RNN and d8. Activation RMS and adjacent-layer cosine across depth, split by position;
    cosine between positions vs lag, raw and mean-centred, against a different-sequence control."""
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 10}):
        fig, axes = plt.subplots(2, 3, figsize=(20, 9.5), gridspec_kw=dict(width_ratios=(1.15, 1.15, 1)))
        for row, (title, run, color) in zip(axes, CONTEXT_MODELS[:2]):
            sim = context_probe(run)['similarity']
            names, _ = stream_names(run)
            x = np.arange(len(names))
            ax = row[0]
            for group, part, style in POSITION_GROUPS:
                ax.plot(x, [np.mean(s['rms'][part]) for s in sim], style, color=color, marker='o', ms=3.5, label=group)
            ax.set(yscale='log', ylabel='activation RMS', title=f'{title}: depth-stream RMS')
            decade_log_axis(ax)
            ax.legend(loc='upper left')
            ax = row[1]
            mid = np.arange(len(names) - 1) + .5
            for key, style, name in (('adjacent_cosine', '-', 'raw'), ('adjacent_cosine_centred', '--', 'mean-centred')):
                ax.plot(mid, [np.mean(s[key][1:]) for s in sim[:-1]], style, color=color, marker='o', ms=4,
                        label=f'{name}, t ≥ 1')
            ax.plot(mid, [s['adjacent_cosine'][0] for s in sim[:-1]], ':', color='k', marker='x', ms=5, label='raw, BOS')
            ax.set(ylim=(-.05, 1.02), ylabel='cos(entry j, entry j+1)', title='similarity of adjacent depth entries')
            ax.grid(alpha=.3)
            ax.legend(loc='lower right')
            for ax in row[:2]:
                ax.set_xticks(x, names, rotation=60, ha='right', fontsize=9)
                ax.set_xlim(-.5, len(names) - .5)
            ax = row[2]
            lags = [int(k) for k in sim[0]['lag_cosine']]
            picks = (1, len(sim) // 2, len(sim) - 1)
            shades = layer_colors(len(picks))
            for j, shade in zip(picks, shades):
                s = sim[j]
                ax.plot(lags, [s['lag_cosine'][str(k)] for k in lags], '-', color=shade, marker='o', ms=3.5,
                        label=f'{names[j]}: raw')
                ax.plot(lags, [s['lag_cosine_centred'][str(k)] for k in lags], '--', color=shade, marker='o', ms=3.5,
                        label=f'{names[j]}: centred')
                ax.axhline(s['other_sequence'], color=shade, lw=.8, ls=':')
            ax.plot([], [], ':', color='0.4', label='raw cos to another sequence')
            ax.set(xscale='log', ylim=(-.03, .6), xlabel='lag between positions', ylabel='cos(x_t, x_{t-lag})',
                   title='similarity between positions (t ≥ 1)')
            log_ticks(ax)
            ax.grid(alpha=.3)
            ax.legend(fontsize=9, loc='upper right', ncol=1)
        fig.tight_layout()
        savefig(fig, 'p2/e_similarity.png')


def deletion_effects(probe_data, entry):
    """Per deleted token: its cosine with the previous token at `entry`, the loss change on the first target
    after it (s+1), and the mean change over the next NEAR targets."""
    cos, first, near = [], [], []
    for d in probe_data['delete']:
        delta = np.array(d['delta'])
        if delta.shape[1] <= NEAR + 1:
            continue
        cos.append(d['cosine_previous'][entry])
        first.append(delta[:, 0])
        near.append(delta[:, 1:NEAR + 1].mean(1))
    return np.ravel(cos), np.ravel(first), np.ravel(near)


def figure_e_removal():
    """Does high similarity mean removable? Branch removal against adjacent-layer cosine, and token
    deletion against the deleted token's cosine with its predecessor."""
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        ax = axes[0]
        for title, run, color in CONTEXT_MODELS[:2]:
            p = context_probe(run)
            cos = [np.mean(s['adjacent_cosine'][1:]) for s in p['similarity'][:-1]]
            _, branches = stream_names(run)
            skip, mean = np.array(p['skip']['skip']), np.array(p['skip']['mean'])
            ax.scatter(cos, np.maximum(skip, 1e-3), s=55, color=color, label=f'{title}: branch skipped')
            ax.scatter(cos, np.maximum(mean, 1e-3), s=55, facecolor='none', edgecolor=color, lw=1.5,
                       label=f'{title}: replaced by its mean update')
            for c, v, m, name in zip(cos, skip, mean, branches):
                if v > .5 or c > .975:
                    ax.annotate(name, (c, v), xytext=(5, 3), textcoords='offset points', fontsize=10, color=color)
                if v > 5 * m:  # the branch is mostly a constant offset: removing it hurts, averaging it does not
                    ax.annotate(f'{name} mean', (c, max(m, 1e-3)), xytext=(6, -4), textcoords='offset points',
                                fontsize=10, color=color)
        ax.set(yscale='log', xlabel='cos(branch input, branch output), mean over t ≥ 1',
               ylabel='full-sequence loss increase (nats; < 1e-3 drawn at 1e-3)',
               title='removing one residual branch')
        decade_log_axis(ax)
        ax.legend(loc='upper left', fontsize=9)
        for ax, key, ylabel, where in ((axes[1], 'first', 'loss change on target s+1 (nats)', 'the next target'),
                                       (axes[2], 'near', f'mean loss change on targets s+2..s+{NEAR + 1} (nats)',
                                        f'the {NEAR} targets after that')):
            for title, run, color in CONTEXT_MODELS[:2]:
                p = context_probe(run)
                cos, first, near = deletion_effects(p, len(p['similarity']) - 1)
                value = first if key == 'first' else near
                edges = np.quantile(cos, np.linspace(0, 1, 11))
                which = np.clip(np.searchsorted(edges, cos, side='right') - 1, 0, 9)
                centres = [cos[which == b].mean() for b in range(10)]
                means = [value[which == b].mean() for b in range(10)]
                errors = [value[which == b].std() / np.sqrt((which == b).sum()) for b in range(10)]
                ax.errorbar(centres, means, errors, color=color, marker='o', ms=5, capsize=3, label=title)
            ax.axhline(0, color='k', lw=.8)
            ax.set(xlabel='cos(x_s, x_{s-1}) at the final stream (decile means)', ylabel=ylabel,
                   title=f'deleting input token s: effect on {where}\n(every 16th position s x 8 held-out rows; later positions preserved)')
            ax.grid(alpha=.3)
            ax.legend()
        fig.tight_layout()
        savefig(fig, 'p2/e_removal.png')


def context_curve(p, keep, first, rows=None):
    c = p['context']
    full = np.array(c['full'])
    increase = np.array(c['keep'][f'{keep}_first_{first}']) - full
    if rows is not None:
        increase = increase[rows]
    return increase.mean(), increase.std() / np.sqrt(len(increase)), np.median(increase)


def figure_f_context():
    """Loss increase on the final target when only the last `keep` tokens are visible, with and without BOS,
    against the last-token gradient reach on the embeddings."""
    from experiments.a4_stability.p2_context import CONTEXT_ROWS, KEEPS, REACH_ROWS
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        for title, run, color in CONTEXT_MODELS:
            p = context_probe(run)
            for ax, first, style in ((axes[0], True, '-'), (axes[1], False, '--')):
                mean, error, median = np.array([context_curve(p, k, first) for k in KEEPS]).T
                ax.errorbar(KEEPS, np.maximum(mean, 1e-4), error, fmt=style, color=color, marker='o', ms=4,
                            capsize=2, label=f'{title}: mean ± s.e.')
                ax.plot(KEEPS, np.maximum(median, 1e-4), style, color=color, alpha=.45, lw=1, marker='.',
                        label=f'{title}: median')
            g = np.array(p['reach'])[::-1]  # lag 0 = the final input token
            lags = np.arange(len(g))
            axes[2].plot(lags[:-1] + 1, g[:-1] / g[0], color=color, lw=1.4, label=title)
            axes[2].plot(lags[-1] + 1, g[-1] / g[0], '*', color=color, ms=13)
        for ax, title in ((axes[0], 'BOS kept at position 0'), (axes[1], 'BOS removed too')):
            ax.axhline(.01, color='k', lw=.8, ls=':')
            ax.set(xscale='log', yscale='log', xlabel='recent tokens kept (state reset; positions preserved)',
                   ylabel='last-token loss increase vs full context (nats)',
                   title=f'removing older context: {title}\n({CONTEXT_ROWS} val rows; < 1e-4 drawn at 1e-4)')
            log_ticks(ax)
            decade_log_axis(ax)
            ax.legend(fontsize=8.5, loc='lower left')
        ax = axes[2]
        ax.axhline(.01, color='k', lw=.8, ls=':', label='1% of lag 0')
        ax.plot([], [], '*', color='0.4', ms=11, label='BOS position')
        ax.set(xscale='log', yscale='log', ylim=(1e-7, 2), xlabel='lag T - t (+1)',
               ylabel=r'$\|\partial L_T/\partial e_t\|_{rms}$ / value at lag 0',
               title=f'last-token gradient reach on the embeddings\n({REACH_ROWS} val rows)')
        log_ticks(ax)
        decade_log_axis(ax)
        ax.legend(loc='lower left')
        fig.tight_layout()
        savefig(fig, 'p2/f_context.png')


# ---------------------------------------------------------------------------------------------- table

def summary_table():
    ref, d8 = best(BASE)[1], best('d8')[1]
    lines = ['| variant | params | best lr | best val | Δ vs modern | per-LR val (0.25x..4x center) | '
             'L_T reach L1/L4 (1%) | frozen layers | mean Pr(|h|>.95) | mean tanh\' | h RMS (t>100) | dL/dx top/bottom | dL_T/dx top/bottom |',
             '|' + '---|' * 13]
    for v in ('rnn1', 'rnn4', BASE) + ABLATIONS:
        lr, loss = best(v)
        lab = label(v, lr)
        p = probe(lab)
        d = p['depth']
        last = d['last']['hidden_gradient_rms']
        sat = np.mean(d['full']['fraction_abs_gt_095'])
        hrms = np.mean([np.mean(h[100:]) for h in d['full']['hidden_rms']])
        tanh = np.mean([r['tanh_derivative_mean'] for r in p['recurrence']])
        plain = v in ('rnn1', 'rnn4')
        gl = plain_gradients(lab, 'full') if plain else depth(lab, 'gradient_rms')
        gt = plain_gradients(lab, 'last') if plain else depth(lab, 'gradient_rms', 'last')
        per_lr = ' / '.join('div' if r['status'] != 'complete' else f"{r['validation_loss']:.3f}" for r in manifest_runs(v))
        lines.append(f"| {v} | {p['parameters'] / 1e6:.2f}M | {lr:g} | {loss:.4f} | {loss - ref:+.4f} | {per_lr} | "
                     f"{reach(last[0], 1e-2)}/{reach(last[-1], 1e-2)} | {frozen_layers(lab)}/{len(last)} | {sat:.3f} | {tanh:.3f} | {hrms:.3f} | "
                     f"{gl[-1] / gl[0]:.3g} | {gt[-1] / gt[0]:.3g} |")
    lines += ['', f'Transformer d8 best {d8:.4f}. Component effects, L(with) - L(without), each at its tuned LR:', '',
              '| component | RNN | d8 |', '|---|---|---|']
    for v in ABLATIONS:
        name, rnn_effect, d8_effect = component_effects(v)
        lines.append(f'| {name} | {rnn_effect:+.4f} | {d8_effect:+.4f} |')
    path = PLOT_DIR / 'p2' / 'summary.md'
    path.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    figure_a()
    figure_b()
    figure_c()
    figure_d_summary()
    figure_d_time()
    figure_d_depth()
    figure_d_lr_edge()
    figure_d_gates()
    figure_d_scatter()
    figure_e_similarity()
    figure_e_removal()
    figure_f_context()
    summary_table()
