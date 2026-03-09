"""Shared base for on-policy algorithms (PPO, A2C, TRPO).

Handles network creation, rollout action/value forward passes,
and the multi-epoch minibatch update loop.
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
from src.buffers.rollout_buffer import RolloutBatch, make_minibatches


@flax.struct.dataclass
class OnPolicyState:
    '''immutable training state that flows through lax.scan'''
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    rng: jax.Array
    step: jnp.int32


class OnPolicyBase(BaseAlgorithm):
    '''
    base class for on-policy algorithms.
    subclasses implement _actor_loss and _critic_loss.
    this class handles network init, act/value forward passes,
    and the multi-epoch minibatch update loop.
    '''

    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__(obs_dim, action_dim, config)
        algo_name = config.algorithm.value
        self.actor_model = None
        self.critic_model = None
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self._get_max_grad_norm()),
            optax.adam(config.lr),
        )
        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self._get_max_grad_norm()),
            optax.adam(config.lr),
        )
        self._algo_name = algo_name

    def _get_max_grad_norm(self) -> float:
        algo_cfg = self._get_algo_config()
        return getattr(algo_cfg, 'max_grad_norm', 2.0)

    def _get_algo_config(self):
        '''grab the algorithm-specific sub-config'''
        return getattr(self.config, self.config.algorithm.value)

    def init(self, rng, obs_size: int = 0, action_size: int = 0):
        '''create models, init params and optimizer states'''
        rng, actor_rng, critic_rng, state_rng = jax.random.split(rng, 4)

        self.actor_model, actor_params = create_actor(
            self._algo_name, self.obs_dim, self.action_dim,
            self.config.network, actor_rng,
        )
        self.critic_model, critic_params = create_critic(
            self._algo_name, self.obs_dim, self.action_dim,
            self.config.network, critic_rng,
        )

        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)

        return OnPolicyState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng=state_rng,
            step=jnp.int32(0),
        )

    def act(self, params, obs, rng, deterministic: bool = False):
        '''forward pass through stochastic actor -> (action, log_prob)'''
        action, log_prob, _, _ = self.actor_model.apply(
            params, obs, deterministic=deterministic, rngs={"sample": rng},
        )
        return action, log_prob

    def value(self, params, obs):
        '''forward pass through critic -> scalar value'''
        return self.critic_model.apply(params, obs)

    def update(self, state: OnPolicyState, batch: RolloutBatch):
        '''
        multi-epoch minibatch update.
        normalizes advantages, then for each epoch:
          - shuffle + split into minibatches
          - for each minibatch: compute grads, apply
        returns (new_state, metrics_dict)
        '''
        algo_cfg = self._get_algo_config()
        num_epochs = getattr(algo_cfg, 'update_epochs', 1)
        num_minibatches = getattr(algo_cfg, 'num_minibatches', 32)

        # normalize advantages
        adv = batch.advantage
        if getattr(algo_cfg, 'normalize_advantages', True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = batch.replace(advantage=adv)

        def _epoch(carry, _):
            actor_params, critic_params, actor_opt, critic_opt, rng = carry
            rng, shuffle_rng = jax.random.split(rng)
            minibatches = make_minibatches(batch, shuffle_rng, num_minibatches)

            def _minibatch_step(carry, mb):
                actor_params, critic_params, actor_opt, critic_opt, rng = carry
                rng, fwd_rng = jax.random.split(rng)

                # actor gradient
                (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
                    self._actor_loss, has_aux=True
                )(actor_params, mb, fwd_rng)
                actor_updates, actor_opt = self.actor_optimizer.update(
                    actor_grads, actor_opt, actor_params
                )
                actor_params = optax.apply_updates(actor_params, actor_updates)

                # critic gradient
                (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
                    self._critic_loss, has_aux=True
                )(critic_params, mb)
                critic_updates, critic_opt = self.critic_optimizer.update(
                    critic_grads, critic_opt, critic_params
                )
                critic_params = optax.apply_updates(critic_params, critic_updates)

                metrics = {**actor_metrics, **critic_metrics}
                return (actor_params, critic_params, actor_opt, critic_opt, rng), metrics

            carry = (actor_params, critic_params, actor_opt, critic_opt, rng)
            carry, epoch_metrics = jax.lax.scan(_minibatch_step, carry, minibatches)
            actor_params, critic_params, actor_opt, critic_opt, rng = carry
            # average metrics over minibatches
            avg_metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)
            return (actor_params, critic_params, actor_opt, critic_opt, rng), avg_metrics

        carry = (
            state.actor_params, state.critic_params,
            state.actor_opt_state, state.critic_opt_state, state.rng,
        )
        carry, all_metrics = jax.lax.scan(_epoch, carry, None, length=num_epochs)
        actor_params, critic_params, actor_opt, critic_opt, rng = carry

        # average metrics over epochs
        metrics = jax.tree.map(lambda x: x.mean(), all_metrics)

        new_state = OnPolicyState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=actor_opt,
            critic_opt_state=critic_opt,
            rng=rng,
            step=state.step + 1,
        )
        return new_state, metrics

    @abstractmethod
    def _actor_loss(self, actor_params, batch: RolloutBatch, rng):
        '''compute policy loss. returns (loss, metrics_dict)'''

    @abstractmethod
    def _critic_loss(self, critic_params, batch: RolloutBatch):
        '''compute value loss. returns (loss, metrics_dict)'''
