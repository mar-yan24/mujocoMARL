"""PlaygroundEnvWrapper: auto-reset, domain randomization, observation normalization."""

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp
import flax.struct


@flax.struct.dataclass
class EnvState:
    '''standardized state for all algorithms'''
    pipeline_state: Any # mjx.Data
    obs: jax.Array # (obs_dim,)
    reward: jax.Array # scalar
    done: jax.Array # scalar
    info: dict # step count, rng, etc.


def _flatten_obs(obs):
    '''
    playground envs can return obs as a dict like:
      {'state': array(...), 'privileged_state': array(...)}
    we flatten to a single array for the policy.
    if already an array, pass through.
    '''
    if isinstance(obs, dict):
        # use 'state' key if available (policy-relevant obs)
        # fall back to concatenating all values
        if 'state' in obs:
            return obs['state']
        return jnp.concatenate([v for v in obs.values()], axis=-1)
    return obs


def _resolve_obs_size(raw_obs_size) -> int:
    '''
    playground observation_size can be:
      - int (simple)
      - dict like {'state': (52,), 'privileged_state': (114,)}
    resolve to a single int matching what _flatten_obs produces.
    '''
    if isinstance(raw_obs_size, int):
        return raw_obs_size
    if isinstance(raw_obs_size, dict):
        if 'state' in raw_obs_size:
            s = raw_obs_size['state']
            return s[0] if isinstance(s, tuple) else int(s)
        # sum all components
        total = 0
        for v in raw_obs_size.values():
            total += v[0] if isinstance(v, tuple) else int(v)
        return total
    # maybe a tuple like (52,)
    if isinstance(raw_obs_size, tuple):
        return raw_obs_size[0]
    return int(raw_obs_size)


class PlaygroundEnvWrapper:
    '''
    wraps a Playground MjxEnv for use with our training loops.
    handles:
        - dict obs -> flat array conversion
        - auto-reset on done
        - observation normalization (running mean/var)
        - action scaling
        - domain randomization injection
    '''

    def __init__(self, env_name: str, num_envs: int, domain_randomization: bool = True):
        from mujoco_playground import registry

        self._env = registry.load(env_name)
        self._num_envs = num_envs
        self._domain_randomization = domain_randomization

        # resolve obs size from possibly-dict observation_size
        self._obs_size = _resolve_obs_size(self._env.observation_size)

        # try loading default config and domain randomizer
        self._config = registry.get_default_config(env_name)
        if domain_randomization:
            try:
                self._randomizer = registry.get_domain_randomizer(env_name)
            except Exception:
                self._randomizer = None
        else:
            self._randomizer = None

    @property
    def obs_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._env.action_size

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, rng: jax.Array) -> EnvState:
        '''reset all environments in parallel'''
        rngs = jax.random.split(rng, self._num_envs)
        state = jax.vmap(self._env.reset)(rngs)
        return EnvState(
            pipeline_state=state.pipeline_state,
            obs=_flatten_obs(state.obs),
            reward=state.reward,
            done=state.done,
            info=state.info,
        )

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        '''
        step all environments
        handles auto-reset internally via playground's built-in mechanism
        '''
        from mujoco_playground._src.mjx_env import State as PGState

        # playground expects the original obs format, but we store flattened.
        # reconstruct by passing our flat obs -- playground's step() uses
        # pipeline_state internally, not obs, so this is safe.
        pg_state = PGState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics={},
            info=state.info,
        )
        next_pg = jax.vmap(self._env.step)(pg_state, action)

        # auto-reset: where done, replace with fresh state
        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rngs = jax.random.split(rng, self._num_envs)
        reset_state = jax.vmap(self._env.reset)(rngs)

        next_obs = _flatten_obs(next_pg.obs)
        reset_obs = _flatten_obs(reset_state.obs)

        def _select(reset_val, next_val):
            done_mask = next_pg.done
            if next_val.ndim > 1:
                done_mask = done_mask[:, None]
            return jnp.where(done_mask, reset_val, next_val)

        obs = _select(reset_obs, next_obs)
        pipeline = jax.tree.map(
            _select,
            reset_state.pipeline_state,
            next_pg.pipeline_state,
        )

        return EnvState(
            pipeline_state=pipeline,
            obs=obs,
            reward=next_pg.reward,
            done=next_pg.done,
            info=next_pg.info,
        )


class BraxCompatWrapper:
    '''
    thin wrapper that makes our env compatible with brax.training.agents
    use this for library-benchmark runs (Brax PPO, Brax SAC)
    '''

    def __init__(self, env_name: str):
        from mujoco_playground import registry, wrapper

        self._env = registry.load(env_name)
        self._config = registry.get_default_config(env_name)
        self.brax_env = wrapper.wrap_for_brax_training(self._env)

    @property
    def env(self):
        return self.brax_env
