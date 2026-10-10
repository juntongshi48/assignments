"""Noisy quadratic model (Problem 3.1): f(w) = 1/2 sum_i h_i w_i^2 with minibatch gradients
g_t = H w_t + eps_t, eps_t ~ N(0, sigma^2 I / B), trained on a fixed budget of N examples (N / B updates).

simulate() runs many independent initializations and noise draws for a whole LR grid at once. Every LR
sees the same initializations and noise (common random numbers), so LR comparisons within a sweep are
not confounded by sampling noise.
"""

import numpy as np

N_EXAMPLES = 8192
H_2D = np.array([1.0, 10.0])
INIT_VAR_2D = np.array([1.0, 0.1])  # w_{0,1} ~ N(0, 1), w_{0,2} ~ N(0, .1); second argument is a variance


def simulate(optimizer, lrs, batch, *, h=H_2D, init_var=INIT_VAR_2D, noise_var=1.0, n_samples=4096,
             n_examples=N_EXAMPLES, beta1=0.9, beta2=0.95, eps=1e-8, momentum=0.0, seed=0,
             record_every=None):
    """Mean final loss per LR (inf where any sample diverged) and its standard error.

    optimizer: 'sgd' (with optional heavy-ball `momentum`), 'rmsprop' (Adam with beta1 = 0), or 'adam'.
    With record_every=k, also returns the mean loss every k updates as an array [n_records, n_lrs].
    """
    rng = np.random.default_rng(seed)
    lrs = np.asarray(lrs, dtype=float)[:, None, None]
    steps = n_examples // batch
    w = np.broadcast_to(rng.normal(size=(n_samples, h.size)) * np.sqrt(init_var),
                        (lrs.shape[0], n_samples, h.size)).copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1 = 0.0 if optimizer == "rmsprop" else beta1
    curve = []
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, steps + 1):
            g = h * w + rng.normal(size=(n_samples, h.size)) * np.sqrt(noise_var / batch)
            if optimizer == "sgd":
                m = momentum * m + g
                w = w - lrs * m
            elif optimizer in ("adam", "rmsprop"):
                m = b1 * m + (1 - b1) * g
                v = beta2 * v + (1 - beta2) * g * g
                m_hat = m / (1 - b1**t) if b1 > 0 else m
                w = w - lrs * m_hat / (np.sqrt(v / (1 - beta2**t)) + eps)
            else:
                raise ValueError(f"unknown optimizer {optimizer!r}")
            if record_every and t % record_every == 0:
                curve.append(_mean_loss(w, h)[0])
    loss, se = _mean_loss(w, h)
    return (loss, se, np.array(curve)) if record_every else (loss, se)


def _mean_loss(w, h):
    per_sample = 0.5 * np.sum(h * w * w, axis=-1)
    diverged = ~np.isfinite(per_sample).all(axis=1)
    loss = np.where(diverged, np.inf, per_sample.mean(axis=1))
    se = np.where(diverged, np.inf, per_sample.std(axis=1) / np.sqrt(per_sample.shape[1]))
    return loss, se


def sgd_exact(lr, batch, *, h=H_2D, init_var=INIT_VAR_2D, noise_var=1.0, n_examples=N_EXAMPLES):
    """Closed-form expected final loss of plain SGD: m_{t+1} = (1 - lr h)^2 m_t + lr^2 sigma^2 / B."""
    a = (1 - lr * h) ** 2
    steps = n_examples // batch
    with np.errstate(over="ignore", invalid="ignore"):  # diverging LRs overflow to inf, which is the answer
        geometric = np.where(np.isclose(a, 1.0), steps,
                             (1 - a**steps) / np.where(np.isclose(a, 1.0), 1.0, 1 - a))
        second_moment = a**steps * init_var + geometric * lr**2 * noise_var / batch
    return 0.5 * float(np.sum(h * second_moment))


def tune(optimizer, batch, coarse_lrs, *, max_lr=None, zoom=1.5, n_fine=31, extra_lrs=(), **kwargs):
    """Two-stage LR sweep: a coarse grid, then a dense grid (~3% spacing) within x/÷ `zoom` of the coarse
    argmin, both on the same noise draws. Returns (lrs, loss, se, lr*, best loss, at_edge)."""
    lrs = np.asarray(coarse_lrs, dtype=float)
    loss, _ = simulate(optimizer, lrs, batch, **kwargs)
    best = lrs[int(np.argmin(loss))]
    hi = best * zoom if max_lr is None else min(best * zoom, max_lr)
    lrs = np.unique(np.concatenate([lrs, np.geomspace(best / zoom, hi, n_fine), extra_lrs]))
    loss, se = simulate(optimizer, lrs, batch, **kwargs)
    return (lrs, loss, se, *optimum(lrs, loss))


def optimum(lrs, losses, half_width=1):
    """Optimal LR from a local quadratic fit in log(lr) around the grid argmin (falls back to argmin).

    Keep the window narrow and the grid dense: loss curves are asymmetric in log(lr) (steep on the
    low-LR side), so a wide window biases the vertex (about +10% with 7 points at 14% spacing).
    Returns (lr*, loss at the best grid point, at_edge) where at_edge flags an argmin on the grid boundary.
    """
    lrs, losses = np.asarray(lrs), np.asarray(losses)
    i = int(np.argmin(losses))
    at_edge = i == 0 or i == len(lrs) - 1 or not np.isfinite(losses[min(i + 1, len(lrs) - 1)])
    window = [j for j in range(i - half_width, i + half_width + 1) if 0 <= j < len(lrs) and np.isfinite(losses[j])]
    if len(window) >= 3 and not at_edge:
        c2, c1, _ = np.polyfit(np.log(lrs[window]), losses[window], 2)
        if c2 > 0:
            x = -c1 / (2 * c2)
            if np.log(lrs[window[0]]) <= x <= np.log(lrs[window[-1]]):
                return float(np.exp(x)), float(losses[i]), at_edge
    return float(lrs[i]), float(losses[i]), at_edge


def power_law(batches, values):
    """Fit values ~ c * B^p in log-log space; returns (c, p, predict)."""
    p, log_c = np.polyfit(np.log(batches), np.log(values), 1)
    c = float(np.exp(log_c))
    return c, float(p), lambda b: c * np.asarray(b, dtype=float) ** p
