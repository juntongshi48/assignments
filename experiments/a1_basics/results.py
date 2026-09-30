"""W&B access for the assignment's analysis scripts, with an on-disk cache.

Finished runs never change, so their metadata and histories are cached under
experiments/a1_basics/cache/ and later plots do not refetch thousands of rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
CONFIG_KEYS = (
    "learning_rate", "batch_size", "warmup_percent", "weight_decay", "lr_schedule", "grad_norm",
    "dropout", "qk_norm", "beta1", "beta2", "model_seed", "data_seed", "deterministic",
    "num_train_sequences", "num_epochs", "num_micro_batches", "perturb_one_token", "total_steps",
    "parameter_count", "model_name", "run_name_suffix",
)


@dataclass
class Run:
    id: str
    name: str
    state: str
    tags: tuple[str, ...]
    created_at: str
    config: dict
    summary: dict = field(default_factory=dict)

    @property
    def val_loss(self):
        value = self.summary.get("val_loss")
        return float(value) if value is not None and math.isfinite(float(value)) else None

    @property
    def depth(self):
        return int(str(self.config.get("model_name", "d8")).lstrip("d"))

    def has(self, tag):
        return tag in self.tags


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_runs(tags, *, names=None):
    """Runs carrying any of `tags` (or with a name in `names`), newest per run name.

    Skips runs whose logged steps exceed their configured total: a crashed run left
    open in a reused container can absorb the next job's logs.
    """
    filters = {"$or": [{"tags": {"$in": list(tags)}}]}
    if names:
        filters["$or"].append({"display_name": {"$in": list(names)}})
    newest = {}
    for raw in wandb.Api().runs(PROJECT_PATH, filters=filters):
        last_step = _finite(raw.summary.get("optimizer_step"))
        total_steps = _finite(raw.config.get("total_steps"))
        if last_step is not None and total_steps is not None and last_step >= total_steps:
            print(f"  skipping {raw.name} ({raw.id}): logged step {last_step:g} >= total_steps {total_steps:g}")
            continue
        created_at = str(getattr(raw, "created_at", "") or "")
        if raw.name in newest and newest[raw.name].created_at >= created_at:
            continue
        newest[raw.name] = Run(
            id=raw.id,
            name=raw.name,
            state=raw.state,
            tags=tuple(raw.tags or ()),
            created_at=created_at,
            config={key: raw.config.get(key) for key in CONFIG_KEYS},
            summary={key: raw.summary.get(key) for key in ("val_loss", "train_loss", "optimizer_step", "_runtime")},
        )
    return list(newest.values())


def history(run, keys, *, step_key="optimizer_step"):
    """Return {key: np.ndarray} aligned on `step_key` for rows that have all `keys`.

    Cached for finished runs.
    """
    keys = tuple(keys)
    digest = hashlib.sha1(json.dumps([run.id, step_key, keys]).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{run.id}_{digest}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    raw = wandb.Api().run(f"{PROJECT_PATH}/{run.id}")
    rows = {}
    for row in raw.scan_history(keys=[step_key, *keys], page_size=10_000):
        values = [_finite(row.get(key)) for key in (step_key, *keys)]
        if any(value is None for value in values):
            continue
        rows[int(values[0])] = values[1:]
    steps = np.array(sorted(rows))
    result = {step_key: steps}
    for index, key in enumerate(keys):
        result[key] = np.array([rows[step][index] for step in steps])
    if run.state == "finished":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(result))
    return result


def body_params(depth):
    """Non-embedding parameter count from the handout: N ~= 14.5 * depth * width^2."""
    return 14.5 * depth * (64 * depth) ** 2


def body_compute(depth, tokens):
    """Handout's 6ND body-compute proxy."""
    return 6 * body_params(depth) * tokens


def fit_power_law(x, y, floors=None):
    """Fit L = a * x^-alpha + eps by grid search over eps and log-log least squares.

    Returns dict(a, alpha, eps, rmse, predict). With few points eps is weakly identified,
    so callers should treat it (and extrapolations far out) with suspicion.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    floors = np.linspace(0.0, y.min() - 1e-3, 400) if floors is None else floors
    best = None
    for eps in floors:
        slope, intercept = np.polyfit(np.log(x), np.log(y - eps), 1)
        pred = eps + np.exp(intercept) * x ** slope
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = dict(a=float(np.exp(intercept)), alpha=float(-slope), eps=float(eps), rmse=rmse)
    a, alpha, eps = best["a"], best["alpha"], best["eps"]
    best["predict"] = lambda z: eps + a * np.asarray(z, dtype=float) ** -alpha
    return best


def savefig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path.resolve())
    return path
