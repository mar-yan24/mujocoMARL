"""From-scratch JAX DDPG implementation.

Deterministic policy gradient with target networks.
Simplified TD3: single critic, no delayed updates, no target noise.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from src.algorithms.off_policy_base import OffPolicyBase
from src.networks.jax.factory import create_actor, create_critic
from src.networks.jax.mlp import QCritic


class DDPG(OffPolicyBase):
    '''
    deep deterministic policy gradient.
    like TD3 but: single Q, no policy delay, no target smoothing noise.
    '''

    def init(self, rng, obs_size=0, action_size=0):
        '''override init to use single QCritic instead of DoubleQ'''
        rng, actor_rng, critic_rng, state_rng = jax.random.split(rng, 4)

        self.actor_model, actor_params = create_actor(
            "ddpg", self.obs_dim, self.action_dim,
            self.config.network, actor_rng,
        )

        # single Q critic (not twin)
        self.critic_model = QCritic(
            hidden_dims=self.config.network.critic_hidden,
            activation=self.config.network.activation,
        )
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.action_dim))
        critic_params = self.critic_model.init(critic_rng, dummy_obs, dummy_act)

        target_actor_params = jax.tree.map(lambda x: x.copy(), actor_params)
        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        from src.algorithms.off_policy_base import OffPolicyState
        return OffPolicyState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            actor_opt_state=self.actor_optimizer.init(actor_params),
            critic_opt_state=self.critic_optimizer.init(critic_params),
            rng=state_rng,
            step=jnp.int32(0),
        )

    def _actor_loss(self, actor_params, critic_params, obs, rng):
        '''maximize Q(s, mu(s))'''
        action = self.actor_model.apply(actor_params, obs)
        q = self.critic_model.apply(critic_params, obs, action)
        actor_loss = -q.mean()
        return actor_loss, {"actor_loss": actor_loss}

    def _critic_loss(
        self, critic_params, target_actor_params, target_critic_params,
        obs, action, reward, next_obs, done, rng,
    ):
        '''standard Bellman MSE with target network'''
        gamma = self.config.gamma

        # target
        target_action = self.actor_model.apply(target_actor_params, next_obs)
        target_q = self.critic_model.apply(target_critic_params, next_obs, target_action)
        target = reward + gamma * (1.0 - done) * target_q

        # current
        q = self.critic_model.apply(critic_params, obs, action)
        critic_loss = ((q - target) ** 2).mean()

        return critic_loss, {"critic_loss": critic_loss, "q_mean": q.mean()}
