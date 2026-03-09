"""Save and load JAX pytrees (model params, optimizer state)."""

from __future__ import annotations
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp


def save_checkpoint(state, path: str | Path, step: int | None = None):
    '''serialize a JAX pytree to disk via pickle'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # move arrays to cpu for serialization
    cpu_state = jax.device_get(state)
    payload = {"state": cpu_state}
    if step is not None:
        payload["step"] = step

    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[checkpoint] saved to {path}")


def load_checkpoint(path: str | Path):
    '''deserialize a JAX pytree from disk'''
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"[checkpoint] loaded from {path}")
    return payload["state"], payload.get("step", 0)
