"""Validation-loss curves of the supplied source runs, fetched from the public W&B project once and
cached under experiments/a2/cache/ (the supplied runs are finished and never change)."""

import json
from pathlib import Path

from experiments.a2.provided_sweeps import load

CACHE = Path(__file__).resolve().parent / "cache" / "source_val_curves.json"


def _fetch(run_url):
    import wandb
    run = wandb.Api().run(run_url.split("wandb.ai/")[1].replace("/runs/", "/"))
    rows = [r for r in run.history(samples=100_000, pandas=False) if r.get("val_loss") is not None]
    return dict(total_steps=run.config["total_steps"],
                steps=[r["optimizer_step"] for r in rows], val_loss=[r["val_loss"] for r in rows])


def val_curves(parts=("P1a", "P1b", "P2a")):
    """run_id -> dict(tokens, learning_rate, weight_decay, final_val_loss, total_steps, steps, val_loss)."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = {r["run_id"]: r for part in parts for r in load(part)}
    missing = [rid for rid in rows if rid not in cache]
    for i, rid in enumerate(missing, 1):
        print(f"fetching {i}/{len(missing)}: {rid}")
        cache[rid] = _fetch(rows[rid]["run_url"])
    if missing:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    keys = ("tokens", "learning_rate", "weight_decay", "final_val_loss")
    return {rid: {**{k: rows[rid][k] for k in keys}, **cache[rid]} for rid in rows}
