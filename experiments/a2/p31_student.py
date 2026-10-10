"""Student entry point for P4.1: initialization and Adam parameter groups for each parameterization."""
import argparse
from functools import partial

from experiments.a2.stress import StressConfig, load_tokens, run


# Width policies (P4.1 a, b): Kaiming and standard muP, reference width n0 (512 for width runs); m = width / n0.
# Depth policies (P4.1 c, d): standard muP, Depth-muP, CompleteP on top of the muP width rules, reference depth 2;
# r = L / base_depth. Block multipliers (residual c_L, LR, Adam eps) apply to hidden matrices and within-block norms.
POLICIES = ('kaiming', 'mup', 'depth-mup', 'completep')
BASE_WIDTH = 512
BASE_DEPTH = 2
ADAM_EPS = 1e-8
HIDDEN = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')


def _is_hidden(name):
    """Hidden matrices: every block's attention and feedforward projections (fan-in and fan-out scale with width)."""
    parts = name.split('.')
    return parts[0] == 'blocks' and parts[2] in HIDDEN


def _is_block_norm(name):
    parts = name.split('.')
    return parts[0] == 'blocks' and 'norm' in parts[2]  # norm1, norm2, qnorm, knorm


def depth_multipliers(policy, depth, base_depth=BASE_DEPTH):
    """(residual c_L, block LR multiplier, block eps multiplier) for relative depth r = depth / base_depth."""
    r = depth / base_depth
    return {'kaiming': (1., 1., 1.), 'mup': (1., 1., 1.),
            'depth-mup': (r ** -.5, r ** -.5, r ** -.5), 'completep': (1 / r, 1., 1 / r)}[policy]


def _check(policy):
    if policy not in POLICIES:
        raise ValueError(f'policy must be one of {POLICIES}')


def initialize(model, policy='mup', base_width=BASE_WIDTH, base_depth=BASE_DEPTH):
    """Initialize all weights/gains for `policy` and set its forward multipliers.

    All policies: embedding variance 1, hidden-matrix variance 1/fan-in, normalization gains 1. Readout: Kaiming
    uses variance 1/n and multiplier 1; the muP family uses variance 1/n0 and multiplier 1/m. Residual branch
    multiplier c_L: 1 (Kaiming, muP), r^-1/2 (Depth-muP), 1/r (CompleteP).
    """
    _check(policy)
    m = model.config.width / base_width
    for name, p in model.named_parameters():
        if name == 'embed.weight':
            p.normal_(0., 1.)
        elif name == 'head.weight':
            fan_in = p.shape[1]
            p.normal_(0., (1 / fan_in if policy == 'kaiming' else 1 / base_width) ** .5)
        elif _is_hidden(name):
            p.normal_(0., (1 / p.shape[1]) ** .5)
        elif 'norm' in name.split('.')[-2]:  # norm1, norm2, qnorm, knorm, final norm
            p.fill_(1.)
        else:
            raise ValueError(f'no initialization rule for {name}')
    model.output_multiplier = 1. if policy == 'kaiming' else 1 / m
    residual, _, _ = depth_multipliers(policy, model.config.depth, base_depth)
    for block in model.blocks:
        block.residual_multiplier = residual


def parameter_groups(model, base_lr, policy='mup', base_width=BASE_WIDTH, base_depth=BASE_DEPTH):
    """Adam groups. Hidden matrices: base_lr (Kaiming) or base_lr / m (muP family), times the block LR
    multiplier. Within-block norms: base_lr times the block LR multiplier. Both use eps 1e-8 times the block eps
    multiplier. Embedding, readout, and final norm: base_lr and eps 1e-8."""
    _check(policy)
    m = model.config.width / base_width
    _, lr_mult, eps_mult = depth_multipliers(policy, model.config.depth, base_depth)
    named = list(model.named_parameters())
    hidden = [p for n, p in named if _is_hidden(n)]
    block_norms = [p for n, p in named if _is_block_norm(n)]
    other = [p for n, p in named if not _is_hidden(n) and not _is_block_norm(n)]
    hidden_lr = base_lr if policy == 'kaiming' else base_lr / m
    return [dict(params=hidden, lr=hidden_lr * lr_mult, eps=ADAM_EPS * eps_mult),
            dict(params=block_norms, lr=base_lr * lr_mult, eps=ADAM_EPS * eps_mult),
            dict(params=other, lr=base_lr, eps=ADAM_EPS)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-path', required=True)
    p.add_argument('--val-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--policy', choices=POLICIES, default='mup')
    p.add_argument('--base-width', type=int, default=BASE_WIDTH, help='reference width n0 (64 for depth runs)')
    p.add_argument('--base-depth', type=int, default=BASE_DEPTH)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--precision', choices=('fp32', 'mp'), default='fp32',
                   help='fp32 for width comparisons; mp for depth comparisons')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-alignment', action='store_true')
    p.add_argument('--no-wandb', action='store_true', help='Save diagnostics locally without W&B')
    a = p.parse_args(argv)
    c = StressConfig(width=a.width, depth=a.depth, head_dim=a.head_dim,
                     microbatch=a.microbatch, seed=a.seed, precision=a.precision)
    c.validate()
    train = load_tokens(a.train_path, c.steps * c.batch, c.context)
    val = load_tokens(a.val_path, c.batch, c.context)
    result = run(c, train, val, base_lr=a.lr, initialize_fn=partial(initialize, policy=a.policy, base_width=a.base_width,
                                       base_depth=a.base_depth),
                 groups_fn=partial(parameter_groups, policy=a.policy, base_width=a.base_width,
                                   base_depth=a.base_depth), device=a.device,
                 alignment=not a.no_alignment, output=a.output)
    if not a.no_wandb:
        import wandb
        from pathlib import Path
        from utils import WANDB_ENTITY, WANDB_PROJECT
        from experiments.a2.wandb_diagnostics import log_records
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                        name=Path(a.output).stem, tags=['a2', 'p4.1'],
                        config={**result['config'], 'base_lr': a.lr, 'policy': a.policy,
                                'base_width': a.base_width, 'base_depth': a.base_depth,
                                'parameter_groups': result['parameter_groups'],
                                'data_sha256': result['data_sha256']}) as wb:
            wandb.define_metric('optimizer_step')
            wandb.define_metric('*', step_metric='optimizer_step')
            for row in result['history']:
                wandb.log({'optimizer_step': row['step'],
                           **{key: row[key] for key in ('val_loss', 'train_loss', 'logit_rms') if key in row}})
            log_records(wandb, result['alignment'], result['history'])
            wandb.save(a.output, base_path=str(Path(a.output).parent), policy='now')
            print('WANDB_RUN_URL=' + wb.url)
    print('Final validation loss:', result['history'][-1]['val_loss'])


if __name__ == '__main__':
    main()
