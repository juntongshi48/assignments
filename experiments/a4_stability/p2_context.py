"""Problem 2 (e, f): representation similarity, removal tests and context use, run locally on CPU.

    uv run python -m experiments.a4_stability.p2_context

For each model in RUNS this writes results/a4_stability/p2/context-<run>.json with
  similarity: per depth-stream entry (branch inputs, then the output-norm input), on P1's held-out batch:
              RMS by position; adjacent-layer cosine by position (raw and mean-centred); cosine between
              positions at several lags; a same-position, different-sequence control; cosine to BOS
  skip:       full-sequence loss increase when each residual branch is replaced by the identity ('skip') or by
              the identity plus the branch's mean update ('mean')
  delete:     per deleted input token, the loss change on every later target, plus that token's cosine
              with its predecessor at each depth entry (positions of the remaining tokens are preserved)
  context:    last-token loss on CONTEXT_ROWS validation rows with only the most recent `keep` input
              tokens (RNN state reset, transformer positions preserved), with and without the BOS token
  reach:      RMS over rows and width of dL_T/d(embedding output) at each position
"""
from pathlib import Path

import torch
from torch.nn import functional as F

from .diagnostics import context_loss, save
from .p1_probe import held_out, trained_model

OUTPUT = Path('results/a4_stability/p2')
RUNS = ('a4-rnn-lr0.002', 'a4-d8-lr0.003', 'a4-rnn4-lr0.001')
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DELETE = range(16, 1023, 16)
KEEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768)
CONTEXT_ROWS, REACH_ROWS, CHUNK = 512, 64, 16
EOS = 2


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def capture(model, tokens, position_ids=None):
    """Logits and every depth-stream entry: each branch input, then the output-norm input."""
    entries, handles = {}, []
    branches = model.branches()
    def keep(j):
        def hook(module, args):
            entries[j] = args[0]
        return hook
    for j, branch in enumerate(branches):
        handles.append(branch.register_forward_pre_hook(keep(j)))
    handles.append(model.output_norm.register_forward_pre_hook(keep(len(branches))))
    try:
        logits = model(tokens, position_ids=position_ids)
    finally:
        for handle in handles:
            handle.remove()
    return logits, [entries[j] for j in sorted(entries)]


def similarity(entries):
    """BOS (position 0) is excluded from the lag and control statistics and reported on its own."""
    rows = []
    for j, x in enumerate(entries):
        x = x.double()
        body = x[:, 1:]
        centred = body - body.mean((0, 1))
        row = dict(rms=x.square().mean((0, 2)).sqrt().tolist(),
                   bos_cosine=cosine(body, x[:, :1]).mean().item(),
                   lag_cosine={str(lag): cosine(body[:, lag:], body[:, :-lag]).mean().item() for lag in LAGS},
                   lag_cosine_centred={str(lag): cosine(centred[:, lag:], centred[:, :-lag]).mean().item() for lag in LAGS},
                   other_sequence=cosine(body, body.roll(1, 0)).mean().item(),
                   other_sequence_centred=cosine(centred, centred.roll(1, 0)).mean().item())
        if j + 1 < len(entries):
            nxt = entries[j + 1].double()
            mean, nxt_mean = x[:, 1:].mean((0, 1)), nxt[:, 1:].mean((0, 1))
            row['adjacent_cosine'] = cosine(x, nxt).mean(0).tolist()
            row['adjacent_cosine_centred'] = cosine(x - mean, nxt - nxt_mean).mean(0).tolist()
        rows.append(row)
    return rows


def token_losses(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction='none').view(targets.shape)


def skip_branches(model, tokens, base, entries):
    """Full-sequence loss increase when one residual branch x + F(N(x)) is replaced by x ('skip'), or by
    x plus its update averaged over the batch's non-BOS tokens ('mean'): the second keeps the branch's
    constant offset and removes only its token-dependent part."""
    increases = dict(skip=[], mean=[])
    for j, branch in enumerate(model.branches()):
        mean = (entries[j + 1] - entries[j])[:, 1:].mean((0, 1))
        for kind, replace in (('skip', lambda module, args, output: args[0]),
                              ('mean', lambda module, args, output, mean=mean: args[0] + mean)):
            handle = branch.register_forward_hook(replace)
            try:
                increases[kind].append((token_losses(model(tokens[:, :-1]), tokens[:, 1:]).mean() - base.mean()).item())
            finally:
                handle.remove()
    return increases


def delete_tokens(model, tokens, base, entries):
    """Delete input token s; later targets j > s keep their tokens, the one that predicted s is dropped."""
    length = tokens.shape[1] - 1
    result = []
    for s in DELETE:
        kept = torch.cat((torch.arange(s), torch.arange(s + 1, length)))
        logits = model(tokens[:, kept], position_ids=kept[None])
        targets = torch.cat((kept[1:], torch.tensor([length])))
        after = slice(s - 1, None)  # logits from token s-1 onward now predict targets s+1, s+2, ...
        delta = token_losses(logits[:, after], tokens[:, targets[after]]) - base[:, s:]
        result.append(dict(position=s, delta=delta.tolist(),
                           cosine_previous=[cosine(x[:, s], x[:, s - 1]).tolist() for x in entries]))
    return result


def context_use(model, rows):
    length = rows.shape[1] - 1
    out = dict(full=[], keep={})
    for start in range(0, len(rows), CHUNK):
        batch = rows[start:start + CHUNK]
        out['full'] += context_loss(model, batch, keep=length).tolist()
        for keep in KEEPS:
            for first in (False, True):
                out['keep'].setdefault(f'{keep}_first_{first}', []).extend(context_loss(model, batch, keep, first).tolist())
    inner = rows[:, 1:-1] == EOS
    positions = torch.arange(1, length)
    out['last_eos'] = torch.where(inner, positions, torch.full_like(positions, -1)).max(1).values.tolist()
    return out


def gradient_reach(model, rows):
    total = 0
    branch = model.branches()[0]
    for start in range(0, len(rows), CHUNK):
        batch, seen = rows[start:start + CHUNK], {}
        def hook(module, args):
            args[0].retain_grad()
            seen['x'] = args[0]
        handle = branch.register_forward_pre_hook(hook)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(batch[:, :-1])
            F.cross_entropy(logits[:, -1].float(), batch[:, -1], reduction='sum').backward()
        finally:
            handle.remove()
        total = total + seen['x'].grad.double().square().sum((0, 2))
    model.zero_grad(set_to_none=True)
    return (total / (len(rows) * seen['x'].shape[-1])).sqrt().tolist()


def measure(run, tokens, rows):
    model = trained_model(run, 'final')
    with torch.no_grad():
        logits, entries = capture(model, tokens[:, :-1])
        base = token_losses(logits, tokens[:, 1:])
        result = dict(run=run, loss=base.mean().item(), similarity=similarity(entries),
                      skip=skip_branches(model, tokens, base, entries), delete=delete_tokens(model, tokens, base, entries),
                      context=context_use(model, rows))
    result['reach'] = gradient_reach(model, rows[:REACH_ROWS])
    return result


def validation_rows(count):
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    return torch.tensor(validation[:count]['input_ids'].astype('int64'))


if __name__ == '__main__':
    torch.set_num_threads(8)
    tokens, rows = held_out(), validation_rows(CONTEXT_ROWS)
    for run in RUNS:
        path = OUTPUT / f'context-{run}.json'
        if path.exists():
            continue
        stats = measure(run, tokens, rows)
        save(path, stats)
        print(path, f"L={stats['loss']:.4f}")
