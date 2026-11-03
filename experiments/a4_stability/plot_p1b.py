"""Problem 1(b): one hypothesis per supplied variant, tested against the default d8.

    uv run python -m experiments.a4_stability.plot_p1b

Reads the manifest, public W&B histories (cache/a4_p1/) and the probe JSONs from p1_probe (run p1_probe first;
it measures only downloaded checkpoints). Writes to experiments/a4_stability/plots/p1b/:
  summary_sweep.png            final val loss vs LR for every variant, and best loss relative to the baseline
  <variant>/<hypothesis>.png   one figure per variant, built around what that change should affect most
The baseline in every figure is d8 at its best LR (0.003).
"""
import json

import numpy as np

from experiments.a4_stability.plot_p1a import (
    RESULTS, depth_ticks, history, manifest_runs, mpl, plt, probe, readable_log_axis, savefig, tokens_seen)
from experiments.a4_stability.sweeps import run_name

BASE = 'd8'
BASE_LR = .003
UNIGRAM_ENTROPY = 6.844  # entropy of the val targets' token frequencies (nats): an input-blind predictor
VARIANT_COLOR = 'tab:red'


def label(variant, lr, stage='final'):
    return f'{run_name(variant, lr)}-{stage}' if stage != 'init' else f'a4-{variant}-init'


def has(variant, lr, stage='final'):
    return (RESULTS / f'{label(variant, lr, stage)}.json').exists()


def depth(run, key, objective='full'):
    return np.array([row.get(key, np.nan) for row in probe(run)['depth'][objective]['depth']])


def per_layer(run, key, reduce=np.mean):
    return np.array([reduce(a[key]) for a in probe(run)['attention']])


def baseline(ax, values, x=None, **kwargs):
    x = np.arange(len(values)) if x is None else x
    ax.plot(x, values, 'o-', ms=3, color='k', label=kwargs.pop('label', f'baseline d8 lr {BASE_LR:g}'), **kwargs)


def train_curves(ax, variant, title):
    """Per-step training loss for every LR (the RMS logger samples only every 100 steps), ending where a run diverged."""
    runs = manifest_runs(variant)
    colors = dict(zip([r['lr'] for r in runs], plt.cm.viridis(np.linspace(0, .9, len(runs)))))
    for r in runs:
        steps, values = history(r)['train_loss']
        finite = np.isfinite(values)
        tag = '' if r['status'] == 'complete' else f' (diverged at step {steps[finite][-1] + 1:.0f})'
        ax.plot(tokens_seen(steps[finite]), values[finite], color=colors[r['lr']], lw=.6, alpha=.8, label=f"lr {r['lr']:g}{tag}")
        if r['status'] != 'complete':
            ax.plot(tokens_seen(steps[finite][-1]), values[finite][-1], 'x', ms=10, mew=2, color=colors[r['lr']])
    base = next(r for r in manifest_runs(BASE) if r['lr'] == BASE_LR)
    steps, values = history(base)['train_loss']
    ax.plot(tokens_seen(steps), values, color='k', lw=.6, alpha=.8, label=f'baseline d8 lr {BASE_LR:g}')
    ax.axhline(np.log(4096), color='0.4', ls=':', label='ln 4096: uniform prediction')
    ax.set(xscale='log', xlabel='training tokens (M)', ylabel='training loss (every step)', title=title)
    ax.grid(which='major', axis='y', alpha=.3)
    ax.legend(fontsize=7)


def val_curves(ax, variant, title=None, unigram=False, logy=False, skip=(), baselines=(BASE_LR,), finals=False):
    """Validation loss over training for every LR of a variant (except `skip`), with baseline runs dashed.

    finals=True adds the final val loss of the variant's best run and of the baselines to their legend labels."""
    runs = manifest_runs(variant)
    best = min((r for r in runs if r['status'] == 'complete'), key=lambda r: r['validation_loss'])
    colors = dict(zip([r['lr'] for r in runs], plt.cm.viridis(np.linspace(0, .9, len(runs)))))
    for r in runs:
        if r['lr'] in skip:
            continue
        steps, values = history(r)['val_loss']
        tag = '' if r['status'] == 'complete' else f" ({r['status']})"
        if finals and r is best:
            tag += f" (best, final {r['validation_loss']:.4f})"
        ax.plot(tokens_seen(steps), values, color=colors[r['lr']], lw=1.2, label=f"lr {r['lr']:g}{tag}")
    for lr, color in zip(baselines, ('k', '0.55')):
        base = next(r for r in manifest_runs(BASE) if r['lr'] == lr)
        steps, values = history(base)['val_loss']
        tag = f" (final {base['validation_loss']:.4f})" if finals else ''
        ax.plot(tokens_seen(steps), values, color=color, lw=1.2, ls='--', label=f'baseline d8 lr {lr:g}{tag}')
    if unigram:
        ax.axhline(UNIGRAM_ENTROPY, color='tab:red', ls=':', label=f'token-frequency entropy {UNIGRAM_ENTROPY:.2f}')
    ax.set(xscale='log', xlabel='training tokens (M)', ylabel='validation loss', title=title or f'{variant}: validation loss')
    if logy:
        ax.set_yscale('log')
        readable_log_axis(ax)
    ax.grid(which='major', axis='y', alpha=.3)
    ax.legend(fontsize=7)


def finish_depth(ax, title, ylabel, log=True):
    ax.set(title=title, ylabel=ylabel)
    if log:
        ax.set_yscale('log')
        readable_log_axis(ax)
    else:
        ax.grid(which='major', axis='y', alpha=.3)
    depth_ticks(ax, 17)
    ax.legend(fontsize=7)


def sublayer_ticks(ax, sublayers=16):
    """Axis for per-sublayer quantities (16 = 8 blocks x attention, MLP), unlike depth_ticks' stream entries."""
    ax.set_xticks(range(0, sublayers, 2), [f'L{i}' for i in range(sublayers // 2)])
    ax.set_xticks(range(sublayers), minor=True)
    ax.set_xlim(-.5, sublayers - .5)
    ax.set_xlabel('sublayer: Lj = attention of block j (minor tick = its MLP)', fontsize=8)


def summary_figure(variants):
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=(1.15, 1), wspace=.55))
    colors = dict(zip(variants, plt.cm.tab10(np.arange(len(variants)))))
    base_best = min(r['validation_loss'] for r in manifest_runs(BASE) if r['status'] == 'complete')
    top = 3.45
    bests = {}
    ax = axes[0]
    for v in variants:
        runs = manifest_runs(v)
        done = [r for r in runs if r['status'] == 'complete']
        best = min(done, key=lambda r: r['validation_loss'])
        bests[v] = (best['validation_loss'], best['lr'])
        if v == 'd8-no-residual':
            continue  # 6.87 at every LR: shown in the bar chart only.
        style = dict(color='k', lw=2.2) if v == BASE else dict(color=colors[v], lw=1.2)
        ax.plot([r['lr'] for r in done], [min(r['validation_loss'], top) for r in done], 'o-', ms=4, label=v, **style)
        for r in runs:
            if r['status'] != 'complete':
                ax.plot(r['lr'], top, 'x', ms=10, mew=2.2, color=style['color'], clip_on=False)
                ax.annotate('diverged', (r['lr'], top), xytext=(0, 19), textcoords='offset points', ha='center',
                            fontsize=7, color=style['color'])
            elif r['validation_loss'] > top:
                ax.plot(r['lr'], top, '^', ms=8, color=style['color'], clip_on=False)
                ax.annotate(f"{r['validation_loss']:.2f}", (r['lr'], top), xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=7, color=style['color'])
    ax.set(xscale='log', ylim=(2.89, top), xlabel='peak learning rate', ylabel='final validation loss')
    ax.set_title(f'supplied sweeps (▲ = above {top}, value printed; x = diverged)', pad=32)
    ax.text(.98, .03, 'no-residual omitted: 6.87 at every LR', transform=ax.transAxes, ha='right', fontsize=8)
    ax.set_xticks([.00025, .0005, .001, .002, .004, .008, .016, .032], ['0.00025', '0.0005', '0.001', '0.002',
                                                                       '0.004', '0.008', '0.016', '0.032'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    order = sorted(variants, key=lambda v: bests[v][0])
    delta = np.array([bests[v][0] - base_best for v in order])
    limit = .3
    ax = axes[1]
    ax.barh(range(len(order)), np.minimum(delta, limit), color=['k' if v == BASE else colors[v] for v in order])
    for i, (v, d) in enumerate(zip(order, delta)):
        text = f'{d:+.3f}  (best {bests[v][0]:.4f} at lr {bests[v][1]:g})' + ('  bar clipped' if d > limit else '')
        ax.text(min(max(d, 0), limit) + .005, i, text, va='center', fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color='k', lw=.8)
    ax.set(xlim=(-.03, limit + .25), xlabel='best final val loss minus baseline best (nats)',
           title=f'best LR per variant vs baseline best {base_best:.4f}')
    ax.grid(axis='x', alpha=.3)
    ax.invert_yaxis()
    savefig(fig, 'p1b/summary_sweep.png')
    for v in order:
        print(f'  {v:22} best {bests[v][0]:.4f} at lr {bests[v][1]:<8g} delta {bests[v][0] - base_best:+.4f}')


def no_residual():
    """Without the identity path, nothing ties a token's representation to its own input across 16 sublayers:
    positions should become indistinguishable and the model should fall back to token frequencies."""
    v, lr = 'd8-no-residual', .003
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.2))
    val_curves(axes[0], v, 'validation loss: every LR stalls at the token-frequency entropy', unigram=True)
    ax = axes[1]
    baseline(ax, probe(label(BASE, BASE_LR))['positions']['position_cosine'])
    for stage, ls in [('init', ':'), ('early', '--'), ('final', '-')]:
        ax.plot(probe(label(v, lr, stage))['positions']['position_cosine'], 'o', ls=ls, ms=3, color=VARIANT_COLOR,
                label=f'no residual lr {lr:g}, {stage}')
    ax.set_ylim(0, 1.02)
    finish_depth(ax, 'do different positions still look different?', 'mean cosine between positions of a sequence', log=False)
    ax = axes[2]
    # The raw input to each sublayer: the residual stream for the baseline, the previous F(N(x)) without residuals.
    baseline(ax, depth(label(BASE, BASE_LR), 'activation_rms'))
    for stage, ls in [('init', ':'), ('early', '--'), ('final', '-')]:
        ax.plot(depth(label(v, lr, stage), 'activation_rms'), 'o', ls=ls, ms=3, color=VARIANT_COLOR,
                label=f'no residual lr {lr:g}, {stage}')
    finish_depth(ax, 'scale of the raw input to each sublayer', 'RMS of sublayer input (baseline: residual stream)')
    ax = axes[3]
    for run, color, name in [(label(BASE, BASE_LR, 'init'), '0.5', 'baseline init'),
                             (label(v, lr, 'init'), 'tab:orange', 'no residual init'),
                             (label(v, lr, 'early'), 'tab:pink', f'no residual lr {lr:g} step 100'),
                             (label(BASE, BASE_LR), 'k', f'baseline lr {BASE_LR:g} final'),
                             (label(v, lr), VARIANT_COLOR, f'no residual lr {lr:g} final')]:
        ax.plot(depth(run, 'gradient_rms'), 'o-', ms=3, color=color, label=name)
    finish_depth(ax, 'does the loss gradient reach the bottom?', 'activation-gradient RMS for L')
    ax.set_yscale('log')
    savefig(fig, f'p1b/{v}/collapse.png')


def no_prenorm():
    """Branches see the raw stream, so their outputs scale with it: the stream should grow multiplicatively
    with depth and over training, which is what limits the usable LR."""
    v, lr = 'd8-no-prenorm', .001
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.2))
    ax = axes[0]
    # Same line style per stage for both models: init (shared by every LR, seed 42) dotted, step 100 dashed, final solid.
    for stage, ls in [('init', ':'), ('early', '--'), ('final', '-')]:
        name = 'init (seed 42)' if stage == 'init' else f'lr {{:g}}, {"step 100" if stage == "early" else "final"}'
        ax.plot(depth(label(BASE, BASE_LR, stage), 'activation_rms'), 'o', ls=ls, ms=3, color='k',
                label='baseline ' + name.format(BASE_LR))
        ax.plot(depth(label(v, lr, stage), 'activation_rms'), 'o', ls=ls, ms=3, color=VARIANT_COLOR,
                label='no pre-norm ' + name.format(lr))
    if has(v, .0005):
        ax.plot(depth(label(v, .0005), 'activation_rms'), 'o-', ms=3, color='tab:orange', label='no pre-norm lr 0.0005, final')
    finish_depth(ax, 'stream size across depth', 'activation RMS (residual stream)')
    ax.set_ylim(top=1000)  # room above the curves for the legend
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax = axes[1]
    baseline(ax, depth(label(BASE, BASE_LR), 'raw_branch_input_ratio'))
    ax.plot(depth(label(v, lr), 'raw_branch_input_ratio'), 'o-', ms=3, color=VARIANT_COLOR, label=f'no pre-norm lr {lr:g}, final')
    finish_depth(ax, 'how big is each update relative to the stream?', '||F(x)|| / ||x||')
    train_curves(axes[2], v, 'training loss up to divergence (no warning in the logged RMS)')
    ax = axes[3]
    # Init is shared by every LR (seed 42); the diverged runs (lr 0.002, 0.004) left no checkpoint to probe.
    for run, color, ls, name in [(label(BASE, BASE_LR, 'init'), 'k', ':', 'baseline init (seed 42)'),
                                 (label(BASE, BASE_LR), 'k', '-', f'baseline lr {BASE_LR:g}, final'),
                                 (label(v, lr, 'init'), VARIANT_COLOR, ':', 'no pre-norm init (seed 42, every LR)'),
                                 (label(v, lr), VARIANT_COLOR, '-', f'no pre-norm lr {lr:g}, final')]:
        ax.plot(depth(run, 'gradient_rms'), 'o', ls=ls, ms=3, color=color, label=name)
    finish_depth(ax, 'does the loss gradient reach the bottom?\n(lr 0.002 / 0.004 diverged without a checkpoint)',
                 'activation-gradient RMS for L')
    savefig(fig, f'p1b/{v}/scale.png')


def no_qknorm():
    """Without QK norm, attention logits are q.k of unnormalized vectors and can grow with the weights:
    at high LR attention should become nearly one-hot (large logits, low entropy)."""
    v = 'd8-no-qknorm'
    # The best LR and the highest one; 0.006 sits between them and only crowds the panels.
    rates = [lr for lr in (.003, .012) if has(v, lr)]
    colors = dict(zip(rates, (VARIANT_COLOR, 'orange')))
    fig, axes = plt.subplots(1, 5, figsize=(32, 5.2))
    for ax, key, title, ylabel in [
            (axes[0], 'max_logit_late', 'largest attention logit', 'max_k q.k/sqrt(d), mean over heads and late queries'),
            (axes[1], 'entropy_late', 'attention entropy', 'entropy / uniform-prefix entropy')]:
        scale = probe(label(BASE, BASE_LR))['attention'][0]['entropy_uniform_late'] if key == 'entropy_late' else 1
        ax.plot(per_layer(label(BASE, BASE_LR, 'early'), key) / scale, 'o--', ms=3, color='k',
                label=f'baseline d8 lr {BASE_LR:g}, early')
        baseline(ax, per_layer(label(BASE, BASE_LR), key) / scale, label=f'baseline d8 lr {BASE_LR:g}, final')
        for lr in rates:
            for stage, ls in [('early', '--'), ('final', '-')]:
                if has(v, lr, stage):
                    ax.plot(per_layer(label(v, lr, stage), key) / scale, 'o', ls=ls, ms=3, color=colors[lr],
                            label=f'no QK-norm lr {lr:g}, {stage}')
        ax.set(xlabel='layer', ylabel=ylabel, title=title)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axes[0].set_yscale('log')
    readable_log_axis(axes[0])
    axes[1].set_ylim(-.02, 1.08)  # room above the curves for the legend
    axes[1].legend(fontsize=7, ncol=2, loc='upper center')
    ax = axes[2]
    # Same LR, the only difference is QK-norm: the paired comparison for the high-LR failure.
    scale = probe(label(BASE, BASE_LR))['attention'][0]['entropy_uniform_late']
    for variant, color, name in [(BASE, '0.45', 'baseline (with QK-norm)'), (v, 'orange', 'no QK-norm')]:
        for stage, ls in [('early', '--'), ('final', '-')]:
            ax.plot(per_layer(label(variant, .012, stage), 'entropy_late') / scale, 'o', ls=ls, ms=3, color=color,
                    label=f"{name} lr 0.012, {'step 100' if stage == 'early' else 'final'}")
    ax.set(xlabel='layer', ylabel='entropy / uniform-prefix entropy', ylim=(-.02, 1.08),
           title='attention entropy at the same lr 0.012')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2, loc='upper center')
    val_curves(axes[3], v, skip=(.0015,), baselines=(BASE_LR, .012))
    best_gap_panel(axes[4], v, 'no QK-norm')
    savefig(fig, f'p1b/{v}/attention_logits.png')


def best_gap_panel(ax, variant, name, start=300, extra_lr=None):
    """Zoom on the end of training: the variant's best run against the baseline, with the final val-loss gap.

    extra_lr adds both models' runs at that LR (dashed), for a same-LR comparison away from the optimum."""
    best = min((r for r in manifest_runs(variant) if r['status'] == 'complete'), key=lambda r: r['validation_loss'])
    base = next(r for r in manifest_runs(BASE) if r['lr'] == BASE_LR)
    curves = [(base, 'k', '-', f'baseline d8 lr {BASE_LR:g}'), (best, VARIANT_COLOR, '-', f"best {name}: lr {best['lr']:g}")]
    if extra_lr is not None:
        curves += [(next(r for r in manifest_runs(BASE) if r['lr'] == extra_lr), '0.55', '--', f'baseline d8 lr {extra_lr:g}'),
                   (next(r for r in manifest_runs(variant) if r['lr'] == extra_lr), 'tab:orange', '--', f'{name} lr {extra_lr:g}')]
    ends = []
    for run, color, ls, tag in curves:
        steps, values = history(run)['val_loss']
        x = tokens_seen(steps)
        keep = x >= start
        ax.plot(x[keep], values[keep], 'o', ls=ls, ms=2.5, lw=1.2, color=color, label=f"{tag}  (final {run['validation_loss']:.4f})")
        ends.append((values[-1], x[-1], run['validation_loss'], color))
    # End labels, pushed apart vertically when final values are close.
    low, high = ax.get_ylim()
    gap, previous = .035*(high-low), -np.inf
    for y, x, value, color in sorted(ends):
        text_y = max(y, previous + gap)
        ax.annotate(f'{value:.4f}', (x, y), xytext=(x + 8, text_y), va='center', fontsize=8, color=color)
        previous = text_y
    gap = best['validation_loss'] - base['validation_loss']
    ax.set(xlabel='training tokens (M)', ylabel='validation loss', xlim=(start, 680),
           title=f'end of training: best {name} - baseline = {gap:+.4f} nats')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper right')


def no_rope():
    """Without RoPE, attention scores carry no relative position; only the causal mask hints at order.
    Fixed-offset heads (previous token, current token) should weaken, and the loss cost should sit in
    contexts that need order."""
    v, lr = 'd8-no-rope', .003
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.2))
    for ax, d, title in [(axes[0], 1, 'previous-token attention'), (axes[1], 0, 'current-token attention')]:
        for run, color, name in [(label(BASE, BASE_LR), 'k', f'baseline lr {BASE_LR:g}'), (label(v, lr), VARIANT_COLOR, f'no RoPE lr {lr:g}')]:
            lag = [np.array(a['lag_mass'])[:, d] for a in probe(run)['attention']]
            ax.plot([x.max() for x in lag], 'o-', ms=4, color=color, label=f'{name}: strongest head')
            ax.plot([x.mean() for x in lag], 's--', ms=3, color=color, alpha=.6, label=f'{name}: head average')
        ax.set(xlabel='layer', ylabel=f'attention weight at lag {d} (late queries)', title=title, ylim=(0, 1))
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    ax = axes[2]
    lags = np.arange(512) + 1
    for run, color, name in [(label(BASE, BASE_LR), 'k', f'baseline lr {BASE_LR:g}'), (label(v, lr), VARIANT_COLOR, f'no RoPE lr {lr:g}')]:
        a = probe(run)['attention']
        ax.plot(lags, np.mean([np.mean(x['lag_mass'], 0) for x in a], 0), color=color, lw=1.2, label=f'{name}: all layers, heads')
        ax.axhline(a[0]['lag_uniform'], color='0.5', ls=':', lw=1)
    ax.set(xscale='log', yscale='log', xlabel='distance back d = q - k (drawn at d+1)', ylabel='mean attention weight (late queries)',
           title='attention by distance (dotted: uniform)')
    ax.legend(fontsize=7)
    ax = axes[3]
    scale = probe(label(BASE, BASE_LR))['attention'][0]['entropy_uniform_late']
    for variant, color, name in [(BASE, 'k', 'baseline'), (v, VARIANT_COLOR, 'no RoPE')]:
        for stage, ls in [('early', '--'), ('final', '-')]:
            ax.plot(per_layer(label(variant, lr, stage), 'entropy_late') / scale, 'o', ls=ls, ms=3, color=color,
                    label=f"{name} lr {lr:g}, {'step 100' if stage == 'early' else 'final'}")
    ax.set(xlabel='layer', ylabel='entropy / uniform-prefix entropy (head average)', ylim=(-.02, 1.08),
           title='attention entropy across depth (late queries)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2, loc='upper center')
    savefig(fig, f'p1b/{v}/position.png')


def no_final_norm():
    """Logits become lm_head(x_out) directly, so the model itself must keep the top of the stream at a usable
    scale. The top-layer stream and the logit scale should be where this variant differs, more so at high LR."""
    v = 'd8-no-final-norm'
    rates = [lr for lr in (.003, .012) if has(v, lr)]
    colors = dict(zip(rates, [VARIANT_COLOR, 'tab:orange']))
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.2))
    ax = axes[0]
    baseline(ax, depth(label(BASE, BASE_LR), 'activation_rms'))
    if has(BASE, .012):
        ax.plot(depth(label(BASE, .012), 'activation_rms'), 'o--', ms=3, color='0.5', label='baseline lr 0.012')
    for lr in rates:
        ax.plot(depth(label(v, lr), 'activation_rms'), 'o-', ms=3, color=colors[lr], label=f'no final norm lr {lr:g}')
    finish_depth(ax, 'stream size across depth (final checkpoints)', 'activation RMS (residual stream)')
    ax = axes[1]
    stages = ['init', 'early', 'final']
    for variant, lr, color, ls in [(BASE, BASE_LR, 'k', '-'), (BASE, .012, '0.5', '--')] + \
            [(v, lr, colors[lr], '-') for lr in rates]:
        if has(variant, lr):
            name = ('baseline' if variant == BASE else 'no final norm') + f' lr {lr:g}'
            ax.plot(range(3), [probe(label(variant, lr, s))['output']['logit_rms'] for s in stages], 'o', ls=ls,
                    color=color, label=name)
    ax.set_xticks(range(3), ['init (seed 42)', 'step 100', 'final'])
    ax.set(yscale='log', ylabel='RMS of output logits (held-out batch)', title='output logit scale during training')
    readable_log_axis(ax)
    ax.legend(fontsize=7)
    val_curves(axes[2], v, logy=True, baselines=(BASE_LR, .012))
    best_gap_panel(axes[3], v, 'no final norm', extra_lr=.012)
    savefig(fig, f'p1b/{v}/output_scale.png')


def postnorm():
    """N(x + F(x)) removes the identity path for gradients: each of the 16 norms rescales the backward signal,
    so lower layers should receive smaller gradients than in pre-norm, and a lower LR should be needed."""
    v = 'd8-postnorm'
    fig, axes = plt.subplots(1, 5, figsize=(32, 5.2))
    ax = axes[0]
    # Scale-free gradient: activation RMS x gradient RMS, because post-norm entries are normalized (RMS ~ 1).
    for run, color, ls, name in [(label(BASE, BASE_LR, 'init'), '0.5', ':', 'baseline init'),
                                 (label(BASE, BASE_LR), 'k', '-', f'baseline lr {BASE_LR:g} final'),
                                 (label(v, .001, 'init'), 'tab:orange', ':', 'post-norm init'),
                                 (label(v, .001, 'early'), 'tab:orange', '--', 'post-norm lr 0.001, step 100'),
                                 (label(v, .001), VARIANT_COLOR, '-', 'post-norm lr 0.001 final')]:
        ax.plot(depth(run, 'activation_rms') * depth(run, 'gradient_rms'), 'o', ls=ls, ms=3, color=color, label=name)
    finish_depth(ax, 'gradient reaching each depth (scale-free)', 'activation RMS x gradient RMS for L')
    ax = axes[1]
    base = probe(label(BASE, BASE_LR))['positions']['norm_input_rms']
    baseline(ax, base, label=f'baseline lr {BASE_LR:g}: stream (pre-norm input)')
    for lr, color in [(.001, VARIANT_COLOR), (.004, 'tab:purple')]:
        if has(v, lr):
            ax.plot(probe(label(v, lr))['positions']['norm_input_rms'], 'o-', ms=3, color=color,
                    label=f'post-norm lr {lr:g}: x + F(x) before its norm')
    finish_depth(ax, 'RMS of x + F(x) before each norm (x itself has RMS ~ 1)\n'
                     'large value = the branch drowns out the skip', 'RMS')
    sublayer_ticks(ax)
    ax = axes[2]
    # Post-norm sublayers see x itself, so this is ||F(x)|| / ||x||; for the baseline ||F(N(x))|| / ||x||.
    baseline(ax, depth(label(BASE, BASE_LR), 'raw_branch_input_ratio')[:16], label=f'baseline lr {BASE_LR:g}: ||F(N(x))|| / ||x||')
    for lr, stage, color, ls in [(.001, 'init', VARIANT_COLOR, ':'), (.001, 'early', VARIANT_COLOR, '--'),
                                 (.001, 'final', VARIANT_COLOR, '-'), (.004, 'final', 'tab:purple', '-')]:
        if has(v, lr, stage):
            tag = 'init (seed 42)' if stage == 'init' else f"lr {lr:g}, {'step 100' if stage == 'early' else 'final'}"
            ax.plot(depth(label(v, lr, stage), 'raw_branch_input_ratio')[:16], 'o', ls=ls, ms=3, color=color,
                    label=f'post-norm {tag}')
    ax.axhline(1, color='0.5', lw=.8)
    finish_depth(ax, 'size of each update relative to its input  ||F(x)|| / ||x||\n'
                     '(above 1: the branch outweighs the skip)', 'branch output RMS / input RMS')
    sublayer_ticks(ax)
    ax = axes[3]
    # RMS of the MLP's own output F: the ratio above times the RMS of the MLP's input.
    for variant, lr, stage, color, ls, name in [
            (BASE, BASE_LR, 'final', 'k', '-', f'baseline lr {BASE_LR:g}: MLP(N(x)), final'),
            (v, .001, 'init', VARIANT_COLOR, ':', 'post-norm init (seed 42)'),
            (v, .001, 'early', VARIANT_COLOR, '--', 'post-norm lr 0.001, step 100'),
            (v, .001, 'final', VARIANT_COLOR, '-', 'post-norm lr 0.001, final'),
            (v, .004, 'final', 'tab:purple', '-', 'post-norm lr 0.004, final')]:
        run = label(variant, lr, stage)
        output = depth(run, 'raw_branch_input_ratio') * depth(run, 'activation_rms')
        ax.plot(range(8), output[1:16:2], 'o', ls=ls, ms=4, color=color, label=name)
    ax.axhline(1, color='0.5', lw=.8)
    ax.set(xlabel='block', ylabel='RMS of MLP output F(x)', yscale='log',
           title='scale of the MLP output itself\n(post-norm MLP input has RMS ~ 0.75-1)')
    readable_log_axis(ax)
    ax.legend(fontsize=7)
    val_curves(axes[4], v, unigram=True, finals=True)
    savefig(fig, f'p1b/{v}/gradients.png')


def gates_by_layer(run, kind):
    gates = probe(run)['output']['gates']
    rows = sorted((name, g) for name, g in gates.items() if f'.{kind}.' in name)
    return [g['bos'] for _, g in rows], [g['rest'] for _, g in rows], [g['closed_rest'] for _, g in rows]


def gate_panel(ax, run, kinds, title):
    for kind, color in zip(kinds, ('tab:blue', 'tab:green')):
        bos, rest, closed = gates_by_layer(run, kind)
        if not rest:
            continue
        ax.plot(rest, 'o-', color=color, label=f'{kind} gate, non-BOS tokens (mean)')
        ax.plot(bos, 's--', color=color, alpha=.7, label=f'{kind} gate, at BOS (mean)')
        ax.plot(closed, '^:', color=color, alpha=.5, label=f'{kind} gate, fraction of values < 0.1')
    ax.set(xlabel='layer', ylabel='sigmoid gate value', ylim=(0, 1), title=title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)


def sink_panels(axes, v, lr, name):
    """BOS mass per layer and the BOS stream ratio across depth, against the baseline."""
    ax = axes[0]
    base = probe(label(BASE, BASE_LR))['attention']
    ax.plot([np.mean(a['first_late']) for a in base], 'o-', color='k', label=f'baseline lr {BASE_LR:g}')
    ax.plot([np.mean(a['first_late']) for a in probe(label(v, lr))['attention']], 'o-', color=VARIANT_COLOR, label=f'{name} lr {lr:g}')
    ax.axhline(base[0]['first_uniform_late'], color='0.5', ls=':', label='uniform')
    ax.set(yscale='log', xlabel='layer', ylabel='attention weight on BOS (head avg, late queries)', title='attention sink on BOS')
    ax.legend(fontsize=7)
    ax = axes[1]
    if ax is None:
        return
    for run, color, tag in [(label(BASE, BASE_LR), 'k', f'baseline lr {BASE_LR:g}'), (label(v, lr), VARIANT_COLOR, f'{name} lr {lr:g}')]:
        a = np.array(probe(run)['positions']['activation'])
        ax.plot(a[:, 0] / np.median(a[:, 1:], 1), 'o-', ms=3, color=color, label=tag)
    ax.axhline(1, color='0.5', ls=':')
    finish_depth(ax, 'size of the BOS stream relative to other positions', 'RMS at BOS / median RMS elsewhere')


def gated_residual():
    """A per-token, per-channel gate on each update lets the model shrink updates where they are not useful.
    The gate values themselves, and whether the BOS sink is still needed, are what this should change."""
    v, lr = 'd8-gated-residual', .003
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.8), gridspec_kw=dict(hspace=.35))
    axes = axes.ravel()
    gate_panel(axes[0], label(v, lr), ('attention', 'mlp'), f'residual gates (lr {lr:g}, final)')
    ax = axes[1]
    # What is actually added to the stream: ||g * F(N(x))|| / ||x||, against the ungated ||F(N(x))|| / ||x||.
    for stage, ls in [('early', ':'), ('final', '-')]:
        when = 'step 100' if stage == 'early' else 'final'
        gated = json.loads((RESULTS/f'{label(v, lr, stage)}-gated_updates.json').read_text())
        ax.plot(depth(label(BASE, BASE_LR, stage), 'raw_branch_input_ratio')[:16], 'o', ls=ls, ms=3, color='k',
                label=f'baseline lr {BASE_LR:g}, {when}: ||F(N(x))|| / ||x||')
        ax.plot(gated['gated_update_ratio'], 'o', ls=ls, ms=3, color=VARIANT_COLOR,
                label=f'gated residual lr {lr:g}, {when}: ||g * F(N(x))|| / ||x||')
        if stage == 'final':
            ax.plot(gated['raw_update_ratio'], 'o', ls='--', ms=3, color=VARIANT_COLOR, alpha=.45,
                    label=f'gated residual lr {lr:g}, final: ungated ||F(N(x))|| / ||x||')
    ax.axhline(1, color='0.5', lw=.8)
    ax.set(yscale='log', ylabel='update norm / stream norm',
           title='effective residual update relative to the stream\n(what each sublayer actually adds)')
    readable_log_axis(ax)
    sublayer_ticks(ax)
    ax.legend(fontsize=7)
    sink_panels(axes[2:4], v, lr, 'gated residual')
    ax = axes[4]
    # Same-LR comparison: final val loss of gated residual minus the baseline at every swept LR.
    base = {r['lr']: r['validation_loss'] for r in manifest_runs(BASE)}
    runs = [r for r in manifest_runs(v) if r['status'] == 'complete' and r['lr'] in base]
    rates, gaps = [r['lr'] for r in runs], [r['validation_loss'] - base[r['lr']] for r in runs]
    ax.plot(rates, gaps, 'o-', color=VARIANT_COLOR)
    for x, y in zip(rates, gaps):
        ax.annotate(f'{y:+.3f}', (x, y), xytext=(0, -12), textcoords='offset points', ha='center', fontsize=8)
    ax.axhline(0, color='0.5', lw=.8)
    ax.set(xscale='log', xlabel='peak learning rate', ylabel='val loss: gated residual - baseline',
           ylim=(min(gaps) - .01, .005), title='final val-loss difference at the same LR\n(below 0: gated residual is better)')
    ax.set_xticks(rates, [f'{x:g}' for x in rates])
    ax.minorticks_off()
    ax.grid(alpha=.3)
    best_gap_panel(axes[5], v, 'gated residual')
    savefig(fig, f'p1b/{v}/gates_and_sink.png')
    gate_histograms(v, lr)
    gate_by_position(v, lr)


def gate_histograms(v='d8-gated-residual', lr=.003, stage='final'):
    """Distribution of every residual gate value (token x channel) in each layer, attention and MLP, BOS separately."""
    run = label(v, lr, stage)
    gated = json.loads((RESULTS/f'{run}-gated_updates.json').read_text())
    edges = np.linspace(0, 1, gated['gate_bins'] + 1)
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, gridspec_kw=dict(hspace=.3))
    means = {kind: gates_by_layer(run, kind)[1] for kind in ('attention', 'mlp')}
    for layer, ax in enumerate(axes.ravel()):
        for kind, j, color in [('attention', 2*layer, 'tab:blue'), ('mlp', 2*layer + 1, 'tab:green')]:
            rest, bos = (100*np.array(gated[f'gate_hist_{part}'][j]) for part in ('rest', 'bos'))
            ax.stairs(rest, edges, fill=True, alpha=.35, color=color, label=f'{kind} gate, non-BOS tokens')
            ax.stairs(rest, edges, color=color, lw=1)
            ax.stairs(bos, edges, color=color, lw=1.2, ls='--', label=f'{kind} gate, at BOS')
        ax.set(title=f"layer {layer}   (non-BOS mean g: attention {means['attention'][layer]:.2f}, "
                     f"MLP {means['mlp'][layer]:.2f})", xlim=(0, 1))
        ax.title.set_fontsize(10)
        ax.grid(alpha=.3)
        if layer % 4 == 0:
            ax.set_ylabel(f'% of gate values per bin (bin width {1/gated["gate_bins"]:g})')
        if layer >= 4:
            ax.set_xlabel('sigmoid gate value g')
    fig.suptitle(f'gated residual lr {lr:g}, {stage}: distribution of gate values g over all tokens x 512 channels '
                 f'(8 held-out sequences; BOS = position 0)', y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(.5, .955))
    savefig(fig, f'p1b/{v}/gate_histograms.png')


def gate_by_position(v='d8-gated-residual', lr=.003, stage='final'):
    """Channel-mean residual gate of each token against its position, per layer: mean over the 8 sequences and their range."""
    gated = json.loads((RESULTS/f'{label(v, lr, stage)}-gated_updates.json').read_text())
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, sharey=True, gridspec_kw=dict(hspace=.25))
    for layer, ax in enumerate(axes.ravel()):
        for kind, j, color in [('attention', 2*layer, 'tab:blue'), ('mlp', 2*layer + 1, 'tab:green')]:
            g = np.array(gated['gate_token_mean'][j])  # [sequence, position]
            x = np.arange(1, g.shape[1] + 1)  # position + 1, so BOS (position 0) sits at 1 on the log axis
            ax.fill_between(x, g.min(0), g.max(0), color=color, alpha=.2, lw=0)
            ax.plot(x, g.mean(0), color=color, lw=1, label=f'{kind} gate')
            ax.plot(1, g[:, 0].mean(), 'o', color=color, ms=6, mec='k')
        ax.set(title=f'layer {layer}', xscale='log', ylim=(0, 1))
        ax.grid(alpha=.3)
        if layer % 4 == 0:
            ax.set_ylabel('gate averaged over 512 channels')
        if layer >= 4:
            ax.set_xlabel('token position + 1  (1 = BOS, circled)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + [mpl.patches.Patch(color='0.5', alpha=.3)],
               [f'{l}: mean over the 8 sequences' for l in labels] + ['range (min-max) over the 8 sequences'],
               loc='upper center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .955))
    fig.suptitle(f'gated residual lr {lr:g}, {stage}: per-token gate, averaged over the 512 channels, vs token position',
                 y=.99)
    savefig(fig, f'p1b/{v}/gate_by_position.png')


def relative_sweep_panel(ax, variants):
    """Final val loss of every complete run minus the best d8 run, against LR: offset and LR sensitivity in one view."""
    reference = min(r['validation_loss'] for r in manifest_runs(BASE) if r['status'] == 'complete')
    for variant, color in variants:
        runs = [r for r in manifest_runs(variant) if r['status'] == 'complete']
        gaps = [r['validation_loss'] - reference for r in runs]
        ax.plot([r['lr'] for r in runs], gaps, 'o-', color=color, label=variant)
        j = int(np.argmin(gaps))
        ax.annotate(f'{gaps[j]:+.3f}', (runs[j]['lr'], gaps[j]), xytext=(0, -13), textcoords='offset points',
                    ha='center', fontsize=8, color=color)
    ax.axhline(0, color='0.5', lw=.8, ls=':')
    ax.set(xscale='log', xlabel='peak learning rate',
           ylabel=f'final val loss minus best d8 ({reference:.4f}, lr {BASE_LR:g})',
           title='final val loss relative to the best baseline run\n(labels: each variant\'s best; below 0 = beats d8)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)


def gated_attention():
    """A per-head output gate gives each head a way to output nothing without parking attention on BOS
    (Qiu et al. 2025): the sink, and the large BOS stream that supports it, should shrink or vanish."""
    v, lr = 'd8-gated-attention', .006
    fig, axes = plt.subplots(1, 4, figsize=(27, 5.2))
    sink_panels(axes[:2], v, lr, 'gated attention')
    if has(v, .003):
        axes[0].plot([np.mean(a['first_late']) for a in probe(label(v, .003))['attention']], 'o--',
                     color='tab:orange', label='gated attention lr 0.003 (same LR as baseline)')
        axes[0].legend(fontsize=7)
    ax = axes[2]
    gate_panel(ax, label(v, lr), ('operation',), f'head output gates (lr {lr:g}, final)')
    ax.legend([h for h in ax.get_lines()], ['head gate at non-BOS positions (mean)', 'head gate at the BOS position (mean)',
                                           'fraction of non-BOS gate values < 0.1'], fontsize=7)
    relative_sweep_panel(axes[3], [(BASE, 'k'), (v, VARIANT_COLOR)])
    savefig(fig, f'p1b/{v}/sink.png')


def gated_norm():
    """Gating the normalized branch input lets each branch scale its own input per channel, so the
    effective step on the branch weights depends less on the peak LR: the sweep should be flatter."""
    v = 'd8-gated-norm'
    lr = min((r for r in manifest_runs(v) if r['status'] == 'complete'), key=lambda r: r['validation_loss'])['lr']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    relative_sweep_panel(axes[0], [(BASE, 'k'), ('d8-gated-residual', 'tab:blue'), (v, VARIANT_COLOR)])
    gate_panel(axes[1], label(v, lr), ('attention', 'mlp'), f'norm gates (lr {lr:g}, final)')
    sink_panels([axes[2], None], v, lr, 'gated norm')
    savefig(fig, f'p1b/{v}/lr_sensitivity.png')


FIGURES = dict(no_residual=no_residual, no_prenorm=no_prenorm, no_qknorm=no_qknorm, no_rope=no_rope,
               no_final_norm=no_final_norm, postnorm=postnorm, gated_residual=gated_residual,
               gated_attention=gated_attention, gated_norm=gated_norm)


if __name__ == '__main__':
    summary_figure([BASE] + ['d8-' + key.replace('_', '-') for key in FIGURES])
    for make in FIGURES.values():
        make()
