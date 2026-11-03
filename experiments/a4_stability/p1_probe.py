"""Problem 1: fixed-input depth and attention probes for supplied transformer checkpoints, run locally.

    uv run python -m experiments.a4_stability.p1_probe

Checkpoints are local copies of the shared volume (`modal volume get -e cs312-shared-data hard-dl-dclm-v1
a4_stability/<run> ckpts/a4_stability/`); the held-out batch is the first BATCH rows of the ds42 val split.
For every run this writes results/a4_stability/p1/<run>-{early,final}.json, plus one <variant>-init.json
rebuilt with model seed 42. Each file holds
  depth:      diagnostics.profile (activation / gradient RMS for L and L_T, branch ratios, adjacent cosine)
  positions:  per-position stream RMS and gradient RMS for L and L_T at every depth entry
  attention:  per layer and head, mass by lag vs. uniform, first/self mass by query, entropy, attractor keys
  sink:       losses with key 0 blocked and with BOS dropped, and where the first-token mass goes without BOS
"""
from contextlib import contextmanager
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import profile, rms, save
from .p1_transformer_depth import Attention, GatedResidual

CKPT_ROOT = Path('ckpts/a4_stability')
OUTPUT = Path('results/a4_stability/p1')
BATCH = 8
LATE = 512      # Queries q >= LATE see every lag 0..LATE-1 without touching key 0.
MIN_LAG = 8     # Attractor scores ignore the local window, which every key receives.
MIN_QUERIES = 128
GATE_BINS = 50
TOP = 12
QUERY_POINTS = 48


def held_out(batch=BATCH):
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    return torch.tensor(validation[:batch]['input_ids'].astype('int64'))


def initial_model(name):
    from train import build_model
    from .sweeps import config
    return build_model(config(name, precision='fp32'), 'cpu').eval()


def trained_model(run, stage):
    from model_io import load_model
    path = CKPT_ROOT/run/('early' if stage == 'early' else '')
    return load_model(path, device='cpu', precision='fp32').eval()


@contextmanager
def capture_attention(model):
    """Collect the q/k that SDPA actually sees (after RoPE and QK norm) and the per-head values."""
    seen, handles = {}, []
    layers = [m for m in model.modules() if isinstance(m, Attention)]
    for j, module in enumerate(layers):
        seen[j] = {}
        for label, sub in [('q', module.q_norm), ('k', module.k_norm), ('v', module.v_proj)]:
            def hook(m, args, output, j=j, label=label):
                seen[j][label] = output.detach()
            handles.append(sub.register_forward_hook(hook))
    try:
        yield seen, layers
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def block_first_key():
    """Causal SDPA in which queries 1.. cannot attend to key 0 (query 0 still sees itself)."""
    original = F.scaled_dot_product_attention

    def blocked(q, k, v, attn_mask=None, is_causal=False, **kwargs):
        length = q.shape[-2]
        allowed = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        if attn_mask is not None or not is_causal:
            raise ValueError('block_first_key expects unpadded causal attention.')
        allowed[1:, 0] = False
        return original(q, k, v, attn_mask=allowed, **kwargs)

    F.scaled_dot_product_attention = blocked
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def probabilities(q, k):
    scores = q.float() @ k.float().transpose(-1, -2)/math.sqrt(q.shape[-1])
    causal = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
    return scores.masked_fill(~causal, -torch.inf).softmax(-1)


def query_points(length):
    return np.unique(np.geomspace(1, length-1, QUERY_POINTS).astype(int)).tolist()


def attention_layer(q, k, v, tokens):
    """Statistics of one layer's [batch, head, query, key] attention, averaged over the batch."""
    k = k.repeat_interleave(q.shape[1]//k.shape[1], 1)
    p = probabilities(q, k)
    batch, heads, length, _ = p.shape
    late = torch.arange(LATE, length)
    # Logit scale for late queries: the largest score over allowed keys, and its gap to their mean score.
    scores = (q[:, :, LATE:].float() @ k.float().transpose(-1, -2)/math.sqrt(q.shape[-1]))
    allowed = torch.ones(length, length, dtype=torch.bool).tril()[LATE:]
    max_logit = scores.masked_fill(~allowed, -torch.inf).amax(-1)
    mean_logit = (scores*allowed).sum(-1)/allowed.sum(-1)
    uniform_late = (1/(late.double()+1)).mean().item()
    # Mass at lag d for late queries; key 0 lies beyond every lag here and is reported as the sink.
    lag = torch.stack([p.diagonal(-d, -2, -1)[..., LATE-d:].mean((0, 2)) for d in range(LATE)], -1)
    first_late = p[:, :, LATE:, 0].mean((0, 2))
    entropy = -(p[:, :, LATE:]*p[:, :, LATE:].clamp_min(1e-30).log()).sum(-1).mean((0, 2))
    points = query_points(length)
    # Received mass from queries at least MIN_LAG away, relative to uniform attention over each prefix.
    far = torch.ones(length, length, dtype=torch.bool).tril(-MIN_LAG)
    queries = far.sum(0).clamp_min(1)
    received = (p*far).sum(2)/queries                                          # [batch, head, key]
    uniform_received = (far/torch.arange(1, length+1)[:, None]).sum(0)/queries   # [key]
    ratio = (received/uniform_received)[..., :length-MIN_LAG]
    head_mean = ratio.mean(1)                                                   # [batch, key]
    # Rank only keys with enough far queries; near the end a few sharp queries dominate the ratio.
    ranked = length-MIN_LAG-MIN_QUERIES
    attractors = []
    for score, h in [(head_mean, None)] + [(ratio[:, h], h) for h in range(heads)]:
        top = score[:, 1:ranked].flatten().topk(TOP)
        attractors.append(dict(head=h, key0_ratio=score[:, 0].mean().item(), top=[
            dict(row=int(i)//(ranked-1), position=int(i) % (ranked-1)+1,
                 token=int(tokens[int(i)//(ranked-1), int(i) % (ranked-1)+1]),
                 ratio=s.item()) for s, i in zip(top.values, top.indices)]))
    value = v.view(batch, length, heads, -1).double().square().mean((0, 3)).sqrt()  # [position, head]
    return dict(
        lag_mass=lag.tolist(), lag_uniform=uniform_late, first_late=first_late.tolist(),
        far_late=(1-lag.sum(-1)-first_late).tolist(), first_uniform_late=uniform_late,
        entropy_late=entropy.tolist(), entropy_uniform_late=torch.log(late.double()+1).mean().item(),
        query_points=points, first_mass=p[:, :, points, 0].mean(0).tolist(),
        self_mass=p[:, :, points, points].mean(0).tolist(), uniform_point=[1/(t+1) for t in points],
        received_ratio_mean=head_mean.mean(0).tolist(), attractors=attractors,
        max_logit_late=max_logit.mean((0, 2)).tolist(), logit_gap_late=(max_logit-mean_logit).mean((0, 2)).tolist(),
        value_rms_first=value[0].tolist(), value_rms_rest=value[1:].square().mean(0).sqrt().tolist())


def attention_profile(model, tokens):
    with torch.no_grad(), capture_attention(model) as (seen, layers):
        model(tokens[:, :-1])
    return [attention_layer(seen[j]['q'], seen[j]['k'], seen[j]['v'], tokens[:, :-1]) for j in range(len(layers))]


@torch.no_grad()
def attention_matrices(model, tokens, length=20):
    """A[layer, head, row, query, key] for the first `length` positions; causal, so this equals the full-context map."""
    with capture_attention(model) as (seen, layers):
        model(tokens[:, :length])
    return torch.stack([probabilities(seen[j]['q'], seen[j]['k'].repeat_interleave(
        seen[j]['q'].shape[1]//seen[j]['k'].shape[1], 1)).transpose(0, 1) for j in range(len(layers))])


def position_cosine(x):
    """Mean cosine between different non-BOS positions of the same sequence: 1 means every position looks alike."""
    u = F.normalize(x[:, 1:].detach().double(), dim=-1)
    count = u.shape[1]
    return ((u.sum(1).square().sum(-1)-count)/(count*(count-1))).mean().item()


def position_profile(model, tokens):
    """Per-position RMS (over batch and width) of every depth-stream entry and its L / L_T gradients.

    Also the cosine between positions at each entry, and the RMS of each branch norm's input
    (for post-norm that is the un-normalized sum x + F(x); for pre-norm it repeats the stream)."""
    stream, norm_inputs, handles = {}, {}, []

    def keep(index):
        def hook(module, args):
            x = args[0]
            x.retain_grad()
            stream[index] = x
        return hook

    def norm_input(index):
        def hook(module, args):
            norm_inputs[index] = rms(args[0])
        return hook
    branches = model.branches()
    for j, branch in enumerate(branches):
        handles.append(branch.register_forward_pre_hook(keep(j)))
        if not isinstance(branch.norm, torch.nn.Identity):
            handles.append(branch.norm.register_forward_pre_hook(norm_input(j)))
    handles.append(model.output_norm.register_forward_pre_hook(keep(len(branches))))
    per_position = lambda x: x.detach().double().square().mean((0, 2)).sqrt().tolist()
    result = {}
    try:
        for objective in ('full', 'last'):
            model.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            loss = (F.cross_entropy(logits[:, -1], tokens[:, -1]) if objective == 'last' else
                    F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()))
            loss.backward()
            entries = [stream[j] for j in sorted(stream)]
            if objective == 'full':
                result['activation'] = [per_position(x) for x in entries]
                result['position_cosine'] = [position_cosine(x) for x in entries]
                result['norm_input_rms'] = [norm_inputs.get(j) for j in range(len(branches))]
            result[f'gradient_{objective}'] = [per_position(x.grad) for x in entries]
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def sink_tests(model, tokens):
    """Is position 0 load-bearing, and is the sink tied to the BOS token or to being first?"""
    targets = tokens[:, 2:]
    def token_losses(logits):
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
    normal = token_losses(model(tokens[:, :-1])[:, 1:])
    with block_first_key():
        blocked = token_losses(model(tokens[:, :-1])[:, 1:])
    # Drop BOS and re-index from 0: RoPE scores are relative, so only content can distinguish the new first token.
    with capture_attention(model) as (seen, layers):
        dropped = token_losses(model(tokens[:, 1:-1]))
    first = [probabilities(seen[j]['q'], seen[j]['k'])[:, :, LATE:, 0].mean((0, 2)).tolist() for j in range(len(layers))]
    summary = lambda x: dict(mean=x.mean().item(), last=x[:, -1].mean().item(),
                             by_position=x.mean(0).tolist())
    return dict(normal=summary(normal), block_key0=summary(blocked), drop_bos=summary(dropped),
                drop_bos_first_late=first)


@torch.no_grad()
def output_and_gates(model, tokens):
    """Output-logit scale and, for gated variants, the sigmoid gate values at BOS and elsewhere."""
    gates, handles = {}, []
    for name, module in model.named_modules():
        if name.endswith('.gate') and isinstance(module, torch.nn.Linear):
            def hook(m, args, output, name=name):
                g = output.float().sigmoid()
                gates[name] = dict(bos=g[:, 0].mean().item(), rest=g[:, 1:].mean().item(),
                                   closed_rest=g[:, 1:].lt(.1).float().mean().item())
            handles.append(module.register_forward_hook(hook))
    try:
        logits = model(tokens[:, :-1]).float()
    finally:
        for handle in handles:
            handle.remove()
    entropy = -(logits.log_softmax(-1).exp()*logits.log_softmax(-1)).sum(-1)
    return dict(logit_rms=rms(logits), logit_rms_by_position=logits.square().mean((0, 2)).sqrt()[[0, 1, 10, 100, -1]].tolist(),
                output_entropy=entropy.mean().item(), gates=gates)


def measure(model, tokens):
    model.eval()
    result = dict(depth=profile(model, tokens), positions=position_profile(model, tokens))
    result['attention'] = attention_profile(model, tokens)
    result['sink'] = sink_tests(model, tokens)
    result['output'] = output_and_gates(model, tokens)
    result['parameters'] = sum(p.numel() for p in model.parameters())
    return result


def measure_variant(name, tokens, *, stages=('early', 'final'), force=False):
    """Init once per variant, then every downloaded LR's checkpoints."""
    from .sweeps import learning_rates, run_name
    jobs = [(f'a4-{name}-init', lambda: initial_model(name))]
    for lr in learning_rates(name):
        run = run_name(name, lr)
        for stage in stages:
            # Only downloaded runs are measured; the rest are compared through their logged losses.
            if (CKPT_ROOT/run/('early' if stage == 'early' else '')/'model.pt').exists():
                jobs.append((f'{run}-{stage}', lambda run=run, stage=stage: trained_model(run, stage)))
    for label, load in jobs:
        path = OUTPUT/f'{label}.json'
        if path.exists() and not force:
            continue
        stats = measure(load(), tokens)
        stats['tokens'] = dict(rows=tokens.shape[0], first_token=tokens[:, 0].tolist())
        save(path, stats)
        print(path, f"L={stats['depth']['full']['loss']:.4f} L_T={stats['depth']['last']['loss']:.4f}")


VARIANTS = ('d8', 'd8-no-residual', 'd8-no-prenorm', 'd8-no-qknorm', 'd8-no-rope', 'd8-no-final-norm',
            'd8-postnorm', 'd8-gated-residual', 'd8-gated-attention', 'd8-gated-norm')


@torch.no_grad()
def gated_updates(model, tokens):
    """Per sublayer of a gated-residual model: ||g * F(N(x))|| / ||x|| (what is actually added), ||F(N(x))|| / ||x||,
    the histogram of the sigmoid gate values over (token, channel), split into position 0 (BOS) and the rest,
    and the channel-mean gate of every token ([sequence][position])."""
    rows, handles = [], []
    for branch in model.branches():
        assert isinstance(branch, GatedResidual)
        # Sublayers run one at a time, so rows[-1] is always the one in progress.
        def before(module, args):
            rows.append(dict(x=args[0]))
        def raw(module, args, output):
            rows[-1]['F'] = output
        def gate(module, args, output):
            rows[-1]['g'] = output.sigmoid()
        def after(module, args, output):
            rows[-1]['update'] = output - rows[-1]['x']
        handles += [branch.register_forward_pre_hook(before), branch.operation.register_forward_hook(raw),
                    branch.register_forward_hook(after), branch.gate.register_forward_hook(gate)]
    try:
        model(tokens)
    finally:
        for h in handles:
            h.remove()
    def histogram(g):
        return torch.histc(g.double(), bins=GATE_BINS, min=0, max=1).div(g.numel()).tolist()
    return dict(gated_update_ratio=[rms(r['update'])/rms(r['x']) for r in rows],
                raw_update_ratio=[rms(r['F'])/rms(r['x']) for r in rows],
                gate_bins=GATE_BINS, gate_hist_rest=[histogram(r['g'][:, 1:]) for r in rows],
                gate_hist_bos=[histogram(r['g'][:, 0]) for r in rows],
                gate_token_mean=[r['g'].double().mean(-1).tolist() for r in rows])


def measure_gated_updates(tokens, name='d8-gated-residual'):
    from .sweeps import learning_rates, run_name
    jobs = [(f'a4-{name}-init', lambda: initial_model(name))]
    for lr in learning_rates(name):
        run = run_name(name, lr)
        for stage in ('early', 'final'):
            if (CKPT_ROOT/run/('early' if stage == 'early' else '')/'model.pt').exists():
                jobs.append((f'{run}-{stage}', lambda run=run, stage=stage: trained_model(run, stage)))
    for label, load in jobs:
        path = OUTPUT/f'{label}-gated_updates.json'
        if not path.exists():
            save(path, gated_updates(load(), tokens))
            print(path)


if __name__ == '__main__':
    torch.set_num_threads(8)
    tokens = held_out()
    for variant in VARIANTS:
        measure_variant(variant, tokens)
    measure_gated_updates(tokens)
