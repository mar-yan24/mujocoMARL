"""Shared base for off-policy algorithms (TD3, SAC, DDPG).

Handles network creation, target networks, soft (Polyak) updates,
and the sample-from-buffer > compute-loss > gradient-step loop.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.struct

from src.algorithms.base import BaseAlgorithm
from src.networks.jax.factory import create_actor, create_critic
from src.buffers.replay_buffer import ReplayBufferState, sample_batch


@flax.struct.dataclass
class OffPolicyState:
    '''immutable training state for off-policy algorithms'''
    actor_params: Any
    critic_params: Any
    target_actor_params: Any
    target_critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    rng: jax.Array
    step: jnp.int32


class OffPolicyBase(BaseAlgorithm):
    '''
    base class for off-policy algorithms.
    subclasses implement _actor_loss and _critic_loss.
    this class handles network init with target copies,
    soft target updates, and the gradient step logic.
    '''

    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__(obs_dim, action_dim, config)
        self._algo_name = config.algorithm.value
        self.actor_model = None
        self.critic_model = None
        self.actor_optimizer = optax.adam(config.lr)
        self.critic_optimizer = optax.adam(config.lr)

    def _get_algo_config(self):
        return getattr(self.config, self.config.algorithm.value)

    def init(self, rng, obs_size: int = 0, action_size: int = 0):
        '''create models, init params + target copies + optimizer states'''
        algo_cfg = self._get_algo_config()
        rng, actor_rng, critic_rng, state_rng = jax.random.split(rng, 4)

        distributional = getattr(algo_cfg, 'distributional', False)

        self.actor_model, actor_params = create_actor(
            self._algo_name, self.obs_dim, self.action_dim,
            self.config.network, actor_rng,
        )
        self.critic_model, critic_params = create_critic(
            self._algo_name, self.obs_dim, self.action_dim,
            self.config.network, critic_rng,
            distributional=distributional,
        )

        # target networks are copies of the online networks
        target_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)
        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)

        return OffPolicyState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng=state_rng,
            step=jnp.int32(0),
        )

    def act(self, params, obs, rng, deterministic: bool = False):
        '''deterministic actor + optional exploration noise'''
        action = self.actor_model.apply(params, obs)
        if not deterministic:
            algo_cfg = self._get_algo_config()
            noise_scale = getattr(algo_cfg, 'exploration_noise', 0.1)
            noise = jax.random.normal(rng, action.shape) * noise_scale
            action = jnp.clip(action + noise, -1.0, 1.0)
        return action

    def soft_update(self, online_params, target_params, tau: float):
        '''polyak averaging: target = tau * online + (1 - tau) * target'''
        return optax.incremental_update(online_params, target_params, tau)

    def update(self, state: OffPolicyState, replay_buf: ReplayBufferState, batch=None):
        '''
        single gradient step:
          1. sample batch from replay buffer (if batch not provided)
          2. update critic
          3. optionally update actor (e.g. delayed for TD3)
          4. soft update targets
        returns (new_state, metrics_dict)
        '''
        algo_cfg = self._get_algo_config()
        tau = algo_cfg.tau
        batch_size = algo_cfg.batch_size

        rng = state.rng
        rng, sample_rng, update_rng = jax.random.split(rng, 3)

        if batch is None:
            batch = sample_batch(replay_buf, sample_rng, batch_size)
        obs, action, reward, next_obs, done = batch

        # critic update
        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(
            state.critic_params, state.target_actor_params,
            state.target_critic_params, obs, action, reward, next_obs, done, update_rng,
        )
        critic_updates, critic_opt = self.critic_optimizer.update(
            critic_grads, state.critic_opt_state, state.critic_params,
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        # actor update (subclass may condition on step count via _should_update_actor)
        rng, actor_rng = jax.random.split(rng)

        def _do_actor_update(actor_params, actor_opt):
            (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
                self._actor_loss, has_aux=True
            )(actor_params, critic_params, obs, actor_rng)
            actor_updates, new_actor_opt = self.actor_optimizer.update(
                actor_grads, actor_opt, actor_params,
            )
            new_actor_params = optax.apply_updates(actor_params, actor_updates)
            return new_actor_params, new_actor_opt, actor_metrics

        def _skip_actor_update(actor_params, actor_opt):
            dummy_metrics = {"actor_loss": jnp.float32(0.0)}
            return actor_params, actor_opt, dummy_metrics

        should_update = self._should_update_actor(state.step)
        actor_params, actor_opt, actor_metrics = jax.lax.cond(
            should_update,
            _do_actor_update,
            _skip_actor_update,
            state.actor_params, state.actor_opt_state,
        )

        # soft target update
        target_actor = self.soft_update(actor_params, state.target_actor_params, tau)
        target_critic = self.soft_update(critic_params, state.target_critic_params, tau)

        metrics = {**critic_metrics, **actor_metrics}

        new_state = OffPolicyState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor,
            target_critic_params=target_critic,
            actor_opt_state=actor_opt,
            critic_opt_state=critic_opt,
            rng=rng,
            step=state.step + 1,
        )
        return new_state, metrics

    def _should_update_actor(self, step) -> bool:
        '''override in TD3 for delayed policy updates'''
        return jnp.bool_(True)

    @abstractmethod
    def _actor_loss(self, actor_params, critic_params, obs, rng):
        '''returns (loss, metrics_dict)'''

    @abstractmethod
    def _critic_loss(
        self, critic_params, target_actor_params, target_critic_params,
        obs, action, reward, next_obs, done, rng,
    ):
        '''returns (loss, metrics_dict)'''
