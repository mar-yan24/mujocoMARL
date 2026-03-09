"""From-scratch JAX AMP (Adversarial Motion Priors) implementation.

Discriminator-augmented PPO for motion imitation from MoCap data.
The discriminator learns to distinguish agent transitions from reference
motion transitions, providing a style reward that shapes the policy.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.struct

from src.algorithms.on_policy_base import OnPolicyBase, OnPolicyState
from src.buffers.rollout_buffer import RolloutBatch
from src.networks.jax.factory import create_discriminator


@flax.struct.dataclass
class AMPState(OnPolicyState):
    '''extends OnPolicyState with discriminator params'''
    disc_params: Any
    disc_opt_state: Any


class AMP(OnPolicyBase):
    '''
    adversarial motion priors.
    PPO-based policy + learned discriminator reward.
    reward = task_weight * task_reward + style_weight * disc_reward
    '''

    def __init__(self, obs_dim, action_dim, config):
        super().__init__(obs_dim, action_dim, config)
        self.disc_model = None
        amp_cfg = config.amp
        self.disc_optimizer = optax.chain(
            optax.clip_by_global_norm(2.0),
            optax.adamw(amp_cfg.disc_lr, weight_decay=amp_cfg.disc_weight_decay),
        )

    def init(self, rng, obs_size=0, action_size=0):
        '''init PPO networks + discriminator'''
        base_state = super().init(rng, obs_size, action_size)
        amp_cfg = self.config.amp

        rng, disc_rng = jax.random.split(base_state.rng)

        self.disc_model, disc_params = create_discriminator(
            self.obs_dim, self.config.network, amp_cfg.disc_hidden, disc_rng,
        )
        disc_opt_state = self.disc_optimizer.init(disc_params)

        return AMPState(
            actor_params=base_state.actor_params,
            critic_params=base_state.critic_params,
            actor_opt_state=base_state.actor_opt_state,
            critic_opt_state=base_state.critic_opt_state,
            rng=rng,
            step=base_state.step,
            disc_params=disc_params,
            disc_opt_state=disc_opt_state,
        )

    def compute_disc_reward(self, disc_params, obs_t, obs_t1):
        '''style reward from discriminator: clamp(1 - 0.25 * (D-1)^2, 0, 1)'''
        logit = self.disc_model.apply(disc_params, obs_t, obs_t1)
        # least-squares GAN style reward
        reward = 1.0 - 0.25 * (logit - 1.0) ** 2
        return jnp.clip(reward, 0.0, 1.0)

    def disc_loss(self, disc_params, agent_obs_t, agent_obs_t1,
                  expert_obs_t, expert_obs_t1):
        '''
        least-squares discriminator loss + gradient penalty.
        D(expert) -> 1, D(agent) -> 0
        '''
        amp_cfg = self.config.amp

        expert_logit = self.disc_model.apply(disc_params, expert_obs_t, expert_obs_t1)
        agent_logit = self.disc_model.apply(disc_params, agent_obs_t, agent_obs_t1)

        # LSGAN: (D(expert) - 1)^2 + D(agent)^2
        loss = ((expert_logit - 1.0) ** 2).mean() + (agent_logit ** 2).mean()

        # gradient penalty on expert data
        def _disc_apply(obs_t, obs_t1):
            return self.disc_model.apply(disc_params, obs_t, obs_t1).sum()

        grad_obs = jax.grad(_disc_apply)(expert_obs_t, expert_obs_t1)
        grad_norm = jnp.sqrt(
            jnp.sum(grad_obs[0] ** 2, axis=-1) + jnp.sum(grad_obs[1] ** 2, axis=-1)
        )
        gp = amp_cfg.disc_gradient_penalty * ((grad_norm - 1.0) ** 2).mean()

        total = loss + gp
        return total, {"disc_loss": loss, "grad_penalty": gp}

    def _actor_loss(self, actor_params, batch: RolloutBatch, rng):
        '''same as PPO actor loss (AMP modifies rewards, not the policy loss)'''
        ppo_cfg = self.config.ppo

        _, new_log_prob, _, log_std = self.actor_model.apply(
            actor_params, batch.obs, rngs={"sample": rng},
        )

        ratio = jnp.exp(new_log_prob - batch.log_prob)
        adv = batch.advantage
        loss_unclipped = ratio * adv
        loss_clipped = jnp.clip(
            ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps
        ) * adv
        policy_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()

        entropy = (log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1).mean()
        total_loss = policy_loss - ppo_cfg.entropy_coeff * entropy

        return total_loss, {"policy_loss": policy_loss, "entropy": entropy}

    def _critic_loss(self, critic_params, batch: RolloutBatch):
        '''same as PPO critic loss'''
        ppo_cfg = self.config.ppo
        value = self.critic_model.apply(critic_params, batch.obs)
        value_loss = 0.5 * ((value - batch.returns) ** 2).mean()
        total = ppo_cfg.value_coeff * value_loss
        return total, {"value_loss": value_loss}
