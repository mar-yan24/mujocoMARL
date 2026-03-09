"""Off-policy GPU-resident fixed-size replay buffer.

Pre-allocated arrays for JAX static shape requirements.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.struct


@flax.struct.dataclass
class ReplayBufferState:
    '''pre-allocated fixed-size buffer living on GPU'''
    obs: jax.Array # (max_size, obs_dim)
    action: jax.Array # (max_size, action_dim)
    reward: jax.Array # (max_size,)
    next_obs: jax.Array # (max_size, obs_dim)
    done: jax.Array # (max_size,)
    size: jax.Array # scalar int: current number of stored transitions
    ptr: jax.Array # scalar int: circular write pointer


def create_replay_buffer(
    max_size: int,
    obs_dim: int,
    action_dim: int,
) -> ReplayBufferState:
    '''allocate an empty replay buffer on device'''
    return ReplayBufferState(
        obs=jnp.zeros((max_size, obs_dim)),
        action=jnp.zeros((max_size, action_dim)),
        reward=jnp.zeros((max_size,)),
        next_obs=jnp.zeros((max_size, obs_dim)),
        done=jnp.zeros((max_size,)),
        size=jnp.array(0, dtype=jnp.int32),
        ptr=jnp.array(0, dtype=jnp.int32),
    )


@jax.jit
def add_transition(
    buf: ReplayBufferState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
) -> ReplayBufferState:
    '''insert a single transition, or batched via vmap'''
    max_size = buf.obs.shape[0]
    idx = buf.ptr % max_size
    return buf.replace(
        obs=buf.obs.at[idx].set(obs),
        action=buf.action.at[idx].set(action),
        reward=buf.reward.at[idx].set(reward),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        done=buf.done.at[idx].set(done),
        size=jnp.minimum(buf.size + 1, max_size),
        ptr=buf.ptr + 1,
    )


@jax.jit
def add_batch(
    buf: ReplayBufferState,
    obs: jax.Array, # (B, obs_dim)
    action: jax.Array, # (B, action_dim)
    reward: jax.Array, # (B,)
    next_obs: jax.Array, # (B, obs_dim)
    done: jax.Array, # (B,)
) -> ReplayBufferState:
    '''Insert a batch of transitions using dynamic_update_slice'''
    max_size = buf.obs.shape[0]
    batch_size = obs.shape[0]
    start = buf.ptr % max_size

    # handle wraparound with lax.dynamic_update_slice
    indices = (start + jnp.arange(batch_size)) % max_size
    buf = buf.replace(
        obs=buf.obs.at[indices].set(obs),
        action=buf.action.at[indices].set(action),
        reward=buf.reward.at[indices].set(reward),
        next_obs=buf.next_obs.at[indices].set(next_obs),
        done=buf.done.at[indices].set(done),
        size=jnp.minimum(buf.size + batch_size, max_size),
        ptr=buf.ptr + batch_size,
    )
    return buf


@jax.jit
def sample_batch(
    buf: ReplayBufferState,
    rng: jax.Array,
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    '''uniform random sample from buffer'''
    indices = jax.random.randint(rng, (batch_size,), 0, buf.size)
    return (
        buf.obs[indices],
        buf.action[indices],
        buf.reward[indices],
        buf.next_obs[indices],
        buf.done[indices],
    )

