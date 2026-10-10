"""Problem 4.2: muP for the course model, plus the width sweep launcher for 4.2(a).

muP changes only four settings relative to the course baseline (reference width n0 = 512, m = n / n0):
embedding init G/n -> G/n0, readout init G/sqrt(n) -> G/sqrt(n0), readout forward multiplier 1 -> 1/m, and
hidden-matrix peak LR eta -> eta/m. G is the baseline's shared truncated-normal draw, so muP rescales the
baseline's own embedding/readout tensors and both prescriptions see the same G. Everything else (hidden init
1/sqrt(fan-in), RMSNorm, WD .1 on matrices, clipping at 1, schedule, data) is unchanged.

The forward multiplier is applied after the final RMSNorm, so the probes on `model.norm` see features before it.
"""
import argparse
import math

import torch

import torch.nn.functional as F

from model_config import LMConfig
from modeling import AutoregressiveLM, LlamaDecoderLayer, initialize_model
from optimizers import ADAMW_EPSILON, should_apply_weight_decay

BASE_WIDTH = 512
EXPERIMENT_KEY = 'a2-p42a-D154m'
TOKENS = 153_600_000
SOURCE_LR = .003  # Problem 1's best sampled LR at 153.6M tokens (width 512)


BASE_DEPTH = 8
DEPTH_RULES = ('mup', 'depth-mup', 'completep')


def depth_multipliers(rule, depth, base_depth=BASE_DEPTH):
    """(residual c_L, block LR multiplier, block Adam eps multiplier) for r = depth / base_depth (4.1 rules)."""
    r = depth / base_depth
    return {'mup': (1., 1., 1.), 'depth-mup': (r ** -.5, r ** -.5, r ** -.5), 'completep': (1 / r, 1., 1 / r)}[rule]


class ScaledDecoderLayer(LlamaDecoderLayer):
    """Course decoder layer whose attention and MLP branches are multiplied by residual_multiplier before the
    residual addition (no new parameters, so checkpoints load into the plain layer's state dict)."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        c = self.residual_multiplier
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
                                       attention_mask=attention_mask)
        hidden_states = residual + c * F.dropout(hidden_states, p=self.dropout, training=self.training)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + c * F.dropout(hidden_states, p=self.dropout, training=self.training)


class MuPLM(AutoregressiveLM):
    """Course model with muP embedding/readout initialization and readout multiplier n0 / n, plus an optional
    depth rule (standard muP, Depth-muP, CompleteP) setting the residual branch multiplier for r = L / base_depth."""

    def __init__(self, config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.0,
                 base_width=BASE_WIDTH, depth_rule='mup', base_depth=BASE_DEPTH):
        if tie_word_embeddings:
            raise ValueError('muP here assumes untied embedding and readout')
        super().__init__(config, dtype=dtype, qk_norm=qk_norm, tie_word_embeddings=False, dropout=dropout)
        self.base_width = base_width
        self.output_multiplier = base_width / self.config.hidden_size
        residual, _, _ = depth_multipliers(depth_rule, self.config.num_hidden_layers, base_depth)
        for layer in self.model.layers:
            layer.__class__ = ScaledDecoderLayer
            layer.residual_multiplier = residual

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids is None:
            raise ValueError('input_ids must be provided.')
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.lm_head(hidden * self.output_multiplier)

    def initialize_parameters(self):
        """Baseline init (embedding G/n, readout G/sqrt(n)), then rescale both to the reference width."""
        initialize_model(self)
        n = self.config.hidden_size
        with torch.no_grad():
            self.get_input_embeddings().weight.mul_(n / self.base_width)
            self.get_output_embeddings().weight.mul_(math.sqrt(n / self.base_width))


def build_mup_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2, base_width=BASE_WIDTH,
                        depth_rule='mup', base_depth=BASE_DEPTH):
    """AdamW groups: readout (eta, WD); hidden matrices (eta / m x block LR multiplier, WD, eps x block eps
    multiplier); within-block norms (eta x block LR multiplier, no WD, scaled eps); embedding and final norm
    (eta, no WD). Same WD masking as the baseline. The readout group comes first so the logged LR is eta."""
    if optimizer_name != 'adamw':
        raise ValueError('muP optimizer builder supports adamw only')
    m = model.config.hidden_size / base_width
    _, lr_mult, eps_mult = depth_multipliers(depth_rule, model.config.num_hidden_layers, base_depth)
    block_modules = {id(x) for layer in model.model.layers for x in layer.modules()}
    readout, hidden, block_norms, other = [], [], [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if module is model.lm_head:
                readout.append(p)
            elif isinstance(module, torch.nn.Linear):
                hidden.append(p)
            elif id(module) in block_modules:
                block_norms.append(p)
            else:
                other.append(p)
            if not isinstance(module, torch.nn.Linear) and should_apply_weight_decay(module, name):
                raise ValueError(f'unexpected decayed parameter outside matrices: {name}')
    groups = [dict(params=readout, lr=learning_rate, weight_decay=weight_decay, eps=ADAMW_EPSILON),
              dict(params=hidden, lr=learning_rate / m * lr_mult, weight_decay=weight_decay,
                   eps=ADAMW_EPSILON * eps_mult),
              dict(params=block_norms, lr=learning_rate * lr_mult, weight_decay=0.0, eps=ADAMW_EPSILON * eps_mult),
              dict(params=other, lr=learning_rate, weight_decay=0.0, eps=ADAMW_EPSILON)]
    kwargs = dict(lr=learning_rate, betas=(beta1, beta2), eps=ADAMW_EPSILON, weight_decay=0.0)
    if any(p.is_cuda for g in groups for p in g['params']):
        kwargs['fused'] = True
    return torch.optim.AdamW(groups, **kwargs)


def model_config(width, depth=8):
    """The course model at another width/depth: head dim 64, FFN 3.5 x width."""
    return LMConfig(f'a2-d{depth}-w{width}', 4096, 1024, width, int(3.5 * width), depth, width // 64, width // 64)


# 4.2(a): widths 128 and 256, both prescriptions, a factor-2 LR ladder through the source LR .003.
# muP should keep the width-512 optimum; the baseline's hidden updates grow with width at fixed eta, so its
# optimum should move up at smaller width (4.1(a) suggests ~m^-1 to m^-1.5), hence a ladder shifted upward.
SWEEP = {
    'mup': (.00075, .0015, .003, .006, .012),
    'sp': (.0015, .003, .006, .012, .024),
}
WIDTHS = (128, 256)


def runs():
    return [(p, w, lr) for p, lrs in SWEEP.items() for w in WIDTHS for lr in lrs]


# Extra points: the baseline's width-128 curve was still flat at .024, and Problem 1's width-512 curve has no
# point below .0015. At width 512 the prescriptions coincide (m = 1), so one baseline run extends the shared curve.
EXTRA = [('sp', 128, .048), ('sp', 512, .00075)]


def configs(rs):
    from experiments.a2.modal_launcher import config
    out = []
    for prescription, width, lr in rs:
        extra = dict(model_builder='experiments.a2.mup:MuPLM', model_builder_kwargs=dict(base_width=BASE_WIDTH),
                     optimizer_builder='experiments.a2.mup:build_mup_optimizer',
                     optimizer_kwargs=dict(base_width=BASE_WIDTH)) if prescription == 'mup' else {}
        out.append(config(tokens=TOKENS, batch=64, learning_rate=lr, model_config=model_config(width),
                          run_name_suffix=f'{EXPERIMENT_KEY}-{prescription}',
                          wandb_tags=(EXPERIMENT_KEY, f'{EXPERIMENT_KEY}-{prescription}'), **extra))
    return out


# 4.2(c): held-out width 1024. Power laws fitted to the 4.2(b) optima at widths 128/256/512 predict LR* at 1024
# (SP: 0.5617 n^-0.880 -> .00126; muP: 0.00588 n^-0.161 -> .00193). Local sweep: prediction x (1/2, 1/sqrt2, 1,
# sqrt2, 2), plus the directly transferred source LR .003.
HELDOUT_WIDTH = 1024
HELDOUT = {'sp': (.00063, .00089, .00126, .00178, .00252, SOURCE_LR),
           'mup': (.00097, .00137, .00193, .00273, .00386, SOURCE_LR)}


def heldout_runs():
    return [(p, HELDOUT_WIDTH, lr) for p, lrs in HELDOUT.items() for lr in lrs]


# 4.2(d): width 512, depths 4 and 16 (reference depth 8: Problem 1's curve is the shared source), pilot
# factor-2 grid through the source LR for each prescription.
DEPTH_KEY = 'a2-p42d-D154m'
DEPTHS = (4, 16)
DEPTH_LRS = (.0015, .003, .006)


# Fill-in: the three depth curves whose best point was .0015 (the low edge) get one more point below.
DEPTH_EDGE = [('mup', 16, .00075), ('completep', 16, .00075), ('depth-mup', 4, .00075)]


def depth_runs():
    return [(rule, depth, lr) for rule in DEPTH_RULES for depth in DEPTHS for lr in DEPTH_LRS]


def depth_configs(rs):
    from experiments.a2.modal_launcher import config
    return [config(tokens=TOKENS, batch=64, learning_rate=lr, model_config=model_config(BASE_WIDTH, depth),
                   model_builder='experiments.a2.mup:MuPLM',
                   model_builder_kwargs=dict(base_width=BASE_WIDTH, depth_rule=rule, base_depth=BASE_DEPTH),
                   optimizer_builder='experiments.a2.mup:build_mup_optimizer',
                   optimizer_kwargs=dict(base_width=BASE_WIDTH, depth_rule=rule, base_depth=BASE_DEPTH),
                   run_name_suffix=f'{DEPTH_KEY}-{rule}', wandb_tags=(DEPTH_KEY, f'{DEPTH_KEY}-{rule}'))
            for rule, depth, lr in rs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extra', action='store_true', help='only the EXTRA fill-in points')
    p.add_argument('--depth', action='store_true', help='4.2(d) depth pilot instead of the 4.2(a) width sweep')
    p.add_argument('--heldout', action='store_true', help='4.2(c) held-out width 1024 runs')
    p.add_argument('--depth-edge', action='store_true', help='4.2(d) .00075 fill-in for edge-bracketed curves')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    rs = DEPTH_EDGE if a.depth_edge else depth_runs() if a.depth else heldout_runs() if a.heldout else EXTRA if a.extra else runs()
    unit = 'depth' if a.depth or a.depth_edge else 'width'
    for prescription, size, lr in rs:
        print(f'{prescription:<9} {unit} {size:<4} peak LR {lr:g}' + ('  <- source LR' if lr == SOURCE_LR else ''))
    print(f'{len(rs)} runs')
    if a.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(depth_configs(rs) if a.depth or a.depth_edge else configs(rs))
    else:
        print('Preview only. Add --execute to launch.')


if __name__ == '__main__':
    main()
