"""From-scratch JAX PPO implementation.

Clipped surrogate objective with GAE, value clipping, and entropy bonus.
Uses nested jax.lax.scan for fully JIT-compiled training.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from src.algorithms.on_policy_base import OnPolicyBase
from src.buffers.rollout_buffer import RolloutBatch


class PPO(OnPolicyBase):
    '''
    proximal policy optimization with clipped surrogate.
    inherits multi-epoch minibatch loop from OnPolicyBase.
    '''

    def _actor_loss(self, actor_params, batch: RolloutBatch, rng):
        '''
        clipped surrogate objective:
          L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        plus entropy bonus
        '''
        ppo_cfg = self.config.ppo

        # forward pass through actor
        new_action, new_log_prob, mean, log_std = self.actor_model.apply(
            actor_params, batch.obs, rngs={"sample": rng},
        )

        # importance sampling ratio
        ratio = jnp.exp(new_log_prob - batch.log_prob)

        # clipped surrogate
        adv = batch.advantage
        loss_unclipped = ratio * adv
        loss_clipped = jnp.clip(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * adv
        policy_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()

        # entropy bonus (gaussian entropy = 0.5 * log(2*pi*e*sigma^2))
        entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1).mean()

        total_loss = policy_loss - ppo_cfg.entropy_coeff * entropy

        # approx KL for monitoring
        approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > ppo_cfg.clip_eps).mean()

        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return total_loss, metrics

    def _critic_loss(self, critic_params, batch: RolloutBatch):
        '''
        value loss with optional clipping:
          L = max((V - R)^2, (clip(V, V_old +/- eps) - R)^2)
        '''
        ppo_cfg = self.config.ppo

        value = self.critic_model.apply(critic_params, batch.obs)

        # clipped value loss
        value_clipped = batch.value + jnp.clip(
            value - batch.value, -ppo_cfg.value_clip_eps, ppo_cfg.value_clip_eps
        )
        loss_unclipped = (value - batch.returns) ** 2
        loss_clipped = (value_clipped - batch.returns) ** 2
        value_loss = 0.5 * jnp.maximum(loss_unclipped, loss_clipped).mean()

        total_loss = ppo_cfg.value_coeff * value_loss

        metrics = {
            "value_loss": value_loss,
            "value_mean": value.mean(),
        }
        return total_loss, metrics
