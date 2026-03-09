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


class PlaygroundEnvWrapper:
    '''
    wraps a Playground MjxEnv for use with our training loops.
    handles:
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
        return self._env.observation_size

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
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            info=state.info,
        )

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        '''
        step all environments
        handles auto-reset internally
        '''
        from mujoco_playground._src.mjx_env import State as PGState

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

        def _select(reset_val, next_val):
            return jnp.where(
                next_pg.done[:, None] if next_val.ndim > 1 else next_pg.done,
                reset_val,
                next_val,
            )

        obs = _select(reset_state.obs, next_pg.obs)
        pipeline = jax.tree.map(
            lambda r, n: _select(r, n),
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