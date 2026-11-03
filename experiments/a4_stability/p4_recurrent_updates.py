"""Problem 4: edit the temporal update, keeping the modern RNN's depth stack.

    uv run python -m experiments.a4_stability.p4_recurrent_updates

z is the fraction of NEW information, following the handout. Reset is applied
BEFORE the candidate's recurrent projection; torch.nn.GRU uses a different form.
"""
import torch
from torch import nn
from .p2_rnn_depth import TanhCell, ModernRNN


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x)

    def forward(self, projected, h):
        return (projected + self.hidden(h)).sigmoid()


class GRUCell(TanhCell):
    def __init__(self, config):
        super().__init__(config)
        self.reset = self.make_reset(config)
        self.update = self.make_update(config)

    def make_reset(self, config):
        return Gate(config)

    def make_update(self, config):
        return Gate(config)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(r*h))
        return self.combine(h, candidate, z)

    def combine(self, h, candidate, z):
        return (1-z)*h + z*candidate


class GRU(ModernRNN):
    cell_type = GRUCell


# (a) ablations: each removes one GRU component and keeps the rest.
class InputGate(Gate):
    """sigma(W x + b): the gate cannot see the old state."""
    def __init__(self, config):
        nn.Module.__init__(self)
        self.input = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, projected, h):
        return projected.sigmoid()


class NoResetCell(GRUCell):
    """r = 1: the candidate's recurrent map sees the whole old state."""
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        return self.combine(h, torch.tanh(candidate_input + self.hidden(h)), z)


class NoUpdateCell(GRUCell):
    """z = 1: h' = tanh(Wx + U(r*h)), a tanh RNN with a reset gate and no carry."""
    def make_update(self, config):
        return None

    def project(self, x):
        return self.input(x), self.reset.project(x)

    def step(self, projected, h):
        candidate_input, reset_input = projected
        r = self.reset(reset_input, h)
        return torch.tanh(candidate_input + self.hidden(r*h))


class FixedUpdateCell(NoUpdateCell):
    """z = .5: a constant leaky carry, so the update is no longer data-dependent."""
    def step(self, projected, h):
        return self.combine(h, super().step(projected, h), .5)


class InputGatesCell(GRUCell):
    """r and z depend only on x_t; only the candidate reads h_{t-1}."""
    def make_reset(self, config):
        return InputGate(config)

    def make_update(self, config):
        return InputGate(config)


class NoResetGRU(ModernRNN):
    cell_type = NoResetCell


class NoUpdateGRU(ModernRNN):
    cell_type = NoUpdateCell


class FixedUpdateGRU(ModernRNN):
    cell_type = FixedUpdateCell


class InputGatesGRU(ModernRNN):
    cell_type = InputGatesCell


# (b) proposal: pre-norm across time. Every recurrent map reads N(h_{t-1}), as each
# depth branch reads N(x); the carry (1-z)*h stays raw, like the depth identity path.
class PreNormCell(GRUCell):
    """r, z from N(h); candidate = tanh(Wx + U(r*N(h))); h' = (1-z)*h + z*candidate."""
    def __init__(self, config):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def candidate(self, candidate_input, recurrent):
        return torch.tanh(candidate_input + recurrent)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        n = self.norm(h)
        r, z = self.reset(reset_input, n), self.update(update_input, n)
        return self.combine(h, self.candidate(candidate_input, self.hidden(r*n)), z)


class PreNormLinearCell(PreNormCell):
    """No candidate tanh: N(h) bounds the recurrent read, so nothing can saturate."""
    def candidate(self, candidate_input, recurrent):
        return candidate_input + recurrent


class PreNormGRU(ModernRNN):
    cell_type = PreNormCell


class PreNormLinearGRU(ModernRNN):
    cell_type = PreNormLinearCell


MODELS = dict(gru=GRU, tanh=ModernRNN, no_reset=NoResetGRU, no_update=NoUpdateGRU,
              fixed_update=FixedUpdateGRU, input_gates=InputGatesGRU,
              prenorm=PreNormGRU, prenorm_linear=PreNormLinearGRU)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('gru')  # Edit the variant and LR here.
