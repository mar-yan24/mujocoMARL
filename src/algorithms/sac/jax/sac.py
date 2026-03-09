"""From-scratch JAX SAC implementation.

Entropy-regularized actor-critic with automatic temperature tuning.
Uses TanhStochasticActor for squashed gaussian policy.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.struct

from src.algorithms.off_policy_base import OffPolicyBase, OffPolicyState
from src.buffers.replay_buffer import ReplayBufferState, sample_batch


@flax.struct.dataclass
class SACState(OffPolicyState):
    '''extends OffPolicyState with learnable temperature alpha'''
    log_alpha: jax.Array
    alpha_opt_state: Any


class SAC(OffPolicyBase):
    '''
    soft actor-critic with auto entropy tuning.
    uses TanhStochasticActor (squashed gaussian) for bounded actions.
    '''

    def __init__(self, obs_dim, action_dim, config):
        super().__init__(obs_dim, action_dim, config)
        sac_cfg = config.sac
        # alpha optimizer
        self.alpha_optimizer = optax.adam(config.lr)
        # target entropy = -dim(A)
        self._target_entropy = (
            sac_cfg.target_entropy if sac_cfg.target_entropy is not None
            else -float(action_dim)
        )

    def init(self, rng, obs_size=0, action_size=0):
        '''init networks + alpha parameter'''
        base_state = super().init(rng, obs_size, action_size)

        sac_cfg = self.config.sac
        log_alpha = jnp.log(jnp.float32(sac_cfg.init_alpha))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)

        return SACState(
            actor_params=base_state.actor_params,
            critic_params=base_state.critic_params,
            target_actor_params=base_state.target_actor_params,
            target_critic_params=base_state.target_critic_params,
            actor_opt_state=base_state.actor_opt_state,
            critic_opt_state=base_state.critic_opt_state,
            rng=base_state.rng,
            step=base_state.step,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
        )

    def act(self, params, obs, rng, deterministic=False):
        '''sample from squashed gaussian policy'''
        action, log_prob = self.actor_model.apply(
            params, obs, deterministic=deterministic, rngs={"sample": rng},
        )
        return action

    def update(self, state: SACState, replay_buf: ReplayBufferState, batch=None):
        '''
        SAC update:
          1. sample batch
          2. update critic (min of twin Q with entropy-augmented target)
          3. update actor (max entropy RL objective)
          4. update alpha (dual variable for target entropy)
          5. soft update targets
        '''
        sac_cfg = self.config.sac
        gamma = self.config.gamma
        tau = sac_cfg.tau

        rng = state.rng
        rng, sample_rng, critic_rng, actor_rng, alpha_rng = jax.random.split(rng, 5)

        if batch is None:
            batch = sample_batch(replay_buf, sample_rng, sac_cfg.batch_size)
        obs, action, reward, next_obs, done = batch

        alpha = jnp.exp(state.log_alpha)

        # -- critic update --
        # sample next action from current policy
        next_action, next_log_prob = self.actor_model.apply(
            state.actor_params, next_obs, rngs={"sample": critic_rng},
        )
        # min twin target Q
        target_q1, target_q2 = self.critic_model.apply(
            state.target_critic_params, next_obs, next_action,
        )
        target_q = jnp.minimum(target_q1, target_q2) - alpha * next_log_prob
        target = reward + gamma * (1.0 - done) * target_q

        def critic_loss_fn(critic_params):
            q1, q2 = self.critic_model.apply(critic_params, obs, action)
            loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
            return loss, {"critic_loss": loss, "q1_mean": q1.mean()}

        (c_loss, c_metrics), c_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)
        c_updates, critic_opt = self.critic_optimizer.update(
            c_grads, state.critic_opt_state, state.critic_params,
        )
        critic_params = optax.apply_updates(state.critic_params, c_updates)

        # -- actor update --
        def actor_loss_fn(actor_params):
            new_action, new_log_prob = self.actor_model.apply(
                actor_params, obs, rngs={"sample": actor_rng},
            )
            q1, q2 = self.critic_model.apply(critic_params, obs, new_action)
            q_min = jnp.minimum(q1, q2)
            loss = (alpha * new_log_prob - q_min).mean()
            return loss, {"actor_loss": loss, "log_prob_mean": new_log_prob.mean()}

        (a_loss, a_metrics), a_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        a_updates, actor_opt = self.actor_optimizer.update(
            a_grads, state.actor_opt_state, state.actor_params,
        )
        actor_params = optax.apply_updates(state.actor_params, a_updates)

        # -- alpha update (if auto tuning) --
        def alpha_loss_fn(log_alpha):
            # re-sample for current policy
            _, lp = self.actor_model.apply(
                actor_params, obs, rngs={"sample": alpha_rng},
            )
            return -(log_alpha * (lp + self._target_entropy)).mean()

        if sac_cfg.auto_alpha:
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)
            alpha_updates, alpha_opt = self.alpha_optimizer.update(
                alpha_grads, state.alpha_opt_state,
            )
            log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)
        else:
            log_alpha = state.log_alpha
            alpha_opt = state.alpha_opt_state

        # soft target update
        target_critic = self.soft_update(critic_params, state.target_critic_params, tau)
        target_actor = self.soft_update(actor_params, state.target_actor_params, tau)

        metrics = {
            **c_metrics, **a_metrics,
            "alpha": jnp.exp(log_alpha),
        }

        new_state = SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor,
            target_critic_params=target_critic,
            actor_opt_state=actor_opt,
            critic_opt_state=critic_opt,
            rng=rng,
            step=state.step + 1,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt,
        )
        return new_state, metrics

    # these are not used directly -- SAC overrides full update()
    def _actor_loss(self, actor_params, critic_params, obs, rng):
        raise NotImplementedError("SAC overrides update() directly")

    def _critic_loss(self, critic_params, target_actor_params, target_critic_params,
                     obs, action, reward, next_obs, done, rng):
        raise NotImplementedError("SAC overrides update() directly")
