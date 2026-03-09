"""From-scratch JAX A2C implementation.

Vanilla policy gradient with GAE (PPO without clipping).
Single gradient step per rollout, no minibatch epochs.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from src.algorithms.on_policy_base import OnPolicyBase
from src.buffers.rollout_buffer import RolloutBatch


class A2C(OnPolicyBase):
    '''
    advantage actor-critic.
    vanilla policy gradient: L = -log_prob * advantage
    same update loop as PPO but update_epochs=1, no clipping.
    '''

    def _actor_loss(self, actor_params, batch: RolloutBatch, rng):
        '''vanilla policy gradient + entropy bonus'''
        a2c_cfg = self.config.a2c

        _, new_log_prob, _, log_std = self.actor_model.apply(
            actor_params, batch.obs, rngs={"sample": rng},
        )

        # vanilla PG loss: -E[log_prob * advantage]
        policy_loss = -(new_log_prob * batch.advantage).mean()

        # entropy
        entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1).mean()

        total_loss = policy_loss - a2c_cfg.entropy_coeff * entropy

        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
        }
        return total_loss, metrics

    def _critic_loss(self, critic_params, batch: RolloutBatch):
        '''simple MSE value loss'''
        a2c_cfg = self.config.a2c

        value = self.critic_model.apply(critic_params, batch.obs)
        value_loss = 0.5 * ((value - batch.returns) ** 2).mean()

        total_loss = a2c_cfg.value_coeff * value_loss

        metrics = {
            "value_loss": value_loss,
            "value_mean": value.mean(),
        }
        return total_loss, metrics
