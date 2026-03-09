"""On-policy rollout buffer with GAE computation via reverse lax.scan."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.struct


@flax.struct.dataclass
class RolloutData:
    '''single-step transition for on-policy collection'''
    obs: jax.Array # (obs_dim,)
    action: jax.Array # (action_dim,)
    reward: jax.Array # scalar
    done: jax.Array # scalar bool
    log_prob: jax.Array # scalar
    value: jax.Array # scalar


@flax.struct.dataclass
class RolloutBatch:
    '''batched trajectory data after gae computation'''
    obs: jax.Array # (T, B, obs_dim)
    action: jax.Array # (T, B, action_dim)
    log_prob: jax.Array # (T, B)
    advantage: jax.Array # (T, B)
    returns: jax.Array # (T, B)
    value: jax.Array # (T, B)


def compute_gae(
    rewards: jax.Array, # (T, B)
    values: jax.Array, # (T, B)
    dones: jax.Array, # (T, B)
    next_value: jax.Array,  # (B,)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    '''compute GAE advantages + returns via reverse jax.lax.scan.

    returns:
        advantages: (T, B)
        returns:    (T, B)
    '''
    def _gae_step(carry, transition):
        next_val, gae = carry
        reward, value, done = transition
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return (value, gae), gae

    # scan in reverse time order
    _, advantages = jax.lax.scan(
        _gae_step,
        (next_value, jnp.zeros_like(next_value)),
        (rewards[::-1], values[::-1], dones[::-1]),  # reversed
    )
    advantages = advantages[::-1]  # un-reverse
    returns = advantages + values
    return advantages, returns


def flatten_batch(batch: RolloutBatch) -> RolloutBatch:
    '''reshape (T, B, ...) → (T*B, ...) for minibatch SGD'''
    def _flatten(x):
        return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
    return jax.tree.map(_flatten, batch)


def make_minibatches(
    batch: RolloutBatch,
    rng: jax.Array,
    num_minibatches: int,
) -> RolloutBatch:
    '''
    Shuffle flat batch and split into minibatches
    returns: RolloutBatch with leading dim = num_minibatches
    '''
    flat = flatten_batch(batch)
    total = flat.obs.shape[0]
    perm = jax.random.permutation(rng, total)

    def _shuffle_and_split(x):
        shuffled = x[perm]
        return shuffled.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, flat)
