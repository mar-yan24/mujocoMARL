"""Network factory: create_actor(), create_critic() from config."""


import jax
import jax.numpy as jnp

# from src.networks.jax.mlp import *
# from src.utils.config import NetworkConfig, AlgorithmType


def create_actor(
    algo: "AlgorithmType",
    obs_dim: int,
    action_dim: int,
    cfg: "NetworkConfig",
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
    elif algo == "sac":
        model = TanhStochasticActor(
            hidden_dims=cfg.actor_hidden,
            action_dim=action_dim,
            activation=cfg.activation,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )
    else:  # ppo, a2c, trpo
        model = StochasticActor(
            hidden_dims=cfg.actor_hidden,
            action_dim=action_dim,
            activation=cfg.activation,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            state_dependent_std=(algo == "sac"),
        )

    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    return model, params


def create_critic(
    algo: "AlgorithmType",
    obs_dim: int,
    action_dim: int,
    cfg: "NetworkConfig",
    rng: jax.Array,
    distributional: bool = False,
):
    '''instantiate + initialize appropriate critic'''
    if algo in ("ppo", "a2c", "trpo"):
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

