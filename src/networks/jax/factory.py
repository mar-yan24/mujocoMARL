"""Network factory: create_actor(), create_critic() from config."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.networks.jax.mlp import (
    DeterministicActor,
    StochasticActor,
    TanhStochasticActor,
    Critic,
    QCritic,
    DoubleQCritic,
    DistributionalQCritic,
    AMPDiscriminator,
)
from src.utils.config import NetworkConfig


def create_actor(
    algo: str,
    obs_dim: int,
    action_dim: int,
    cfg: NetworkConfig,
    rng: jax.Array,
):
    '''instantiate + initialize the appropriate actor for the algorithm'''
    if algo in ("td3", "ddpg"):
        model = DeterministicActor(
            hidden_dims=cfg.actor_hidden,
            action_dim=action_dim,
            activation=cfg.activation,
            init_scale=cfg.init_scale,
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        params = model.init(rng, dummy_obs)
    elif algo == "sac":
        model = TanhStochasticActor(
            hidden_dims=cfg.actor_hidden,
            action_dim=action_dim,
            activation=cfg.activation,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )
        # stochastic actors need "sample" rng for self.make_rng("sample")
        dummy_obs = jnp.zeros((1, obs_dim))
        rng_params, rng_sample = jax.random.split(rng)
        params = model.init({"params": rng_params, "sample": rng_sample}, dummy_obs)
    else:  # ppo, a2c, trpo, amp
        model = StochasticActor(
            hidden_dims=cfg.actor_hidden,
            action_dim=action_dim,
            activation=cfg.activation,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            state_dependent_std=False,
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        rng_params, rng_sample = jax.random.split(rng)
        params = model.init({"params": rng_params, "sample": rng_sample}, dummy_obs)

    return model, params


def create_critic(
    algo: str,
    obs_dim: int,
    action_dim: int,
    cfg: NetworkConfig,
    rng: jax.Array,
    distributional: bool = False,
):
    '''instantiate + initialize appropriate critic'''
    if algo in ("ppo", "a2c", "trpo", "amp"):
        model = Critic(hidden_dims=cfg.critic_hidden, activation=cfg.activation)
        dummy = jnp.zeros((1, obs_dim))
        params = model.init(rng, dummy)
    elif distributional:
        model = DistributionalQCritic(
            hidden_dims=cfg.critic_hidden, activation=cfg.activation
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, action_dim))
        params = model.init(rng, dummy_obs, dummy_act)
    else:
        model = DoubleQCritic(
            hidden_dims=cfg.critic_hidden, activation=cfg.activation
        )
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, action_dim))
        params = model.init(rng, dummy_obs, dummy_act)

    return model, params


def create_discriminator(
    obs_dim: int,
    cfg: NetworkConfig,
    disc_hidden: list[int],
    rng: jax.Array,
):
    '''instantiate + initialize AMP discriminator'''
    model = AMPDiscriminator(hidden_dims=disc_hidden, activation=cfg.activation)
    dummy = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy, dummy)
    return model, params
