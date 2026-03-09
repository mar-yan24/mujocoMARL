"""From-scratch JAX TD3 implementation with FastTD3 scaling.

Twin delayed DDPG with distributional critic and large batch support.
Delayed policy updates via jax.lax.cond (no Python branching).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from src.algorithms.off_policy_base import OffPolicyBase


class TD3(OffPolicyBase):
    '''
    twin delayed deep deterministic policy gradient.
    - twin critics: min(Q1, Q2) for target
    - delayed actor updates (every policy_delay steps)
    - target policy smoothing (noise on target actions)
    '''

    def _should_update_actor(self, step):
        td3_cfg = self.config.td3
        return (step % td3_cfg.policy_delay) == 0

    def _actor_loss(self, actor_params, critic_params, obs, rng):
        '''maximize Q(s, actor(s)) using first critic only'''
        action = self.actor_model.apply(actor_params, obs)
        q1, _ = self.critic_model.apply(critic_params, obs, action)
        actor_loss = -q1.mean()

        metrics = {"actor_loss": actor_loss}
        return actor_loss, metrics

    def _critic_loss(
        self, critic_params, target_actor_params, target_critic_params,
        obs, action, reward, next_obs, done, rng,
    ):
        '''twin critic MSE with target policy smoothing'''
        td3_cfg = self.config.td3
        gamma = self.config.gamma

        # target action with smoothing noise
        target_action = self.actor_model.apply(target_actor_params, next_obs)
        noise = jax.random.normal(rng, target_action.shape) * td3_cfg.target_noise
        noise = jnp.clip(noise, -td3_cfg.target_noise_clip, td3_cfg.target_noise_clip)
        target_action = jnp.clip(target_action + noise, -1.0, 1.0)

        # min of twin target Q
        target_q1, target_q2 = self.critic_model.apply(
            target_critic_params, next_obs, target_action,
        )
        target_q = jnp.minimum(target_q1, target_q2)
        target = reward + gamma * (1.0 - done) * target_q

        # current Q estimates
        q1, q2 = self.critic_model.apply(critic_params, obs, action)

        critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()

        metrics = {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
        }
        return critic_loss, metrics
