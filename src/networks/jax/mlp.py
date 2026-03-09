"""Flax neural network modules: MLP, Actor, Critic, Discriminator."""
from __future__ import annotations
from typing import Sequence, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

def get_activation(name: str) -> Callable:
    return {"elu": nn.elu, "relu": nn.relu, "tanh": nn.tanh}[name]


class MLP(nn.Module):
    '''Generic MLP block'''
    hidden_dims: Sequence[int]
    activation: str = "elu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = get_activation(self.activation)
        for dim in self.hidden_dims:
            x = act(nn.Dense(dim)(x))
        return x
    

class DeterministicActor(nn.Module):
    '''Deterministic policy: obs -> action specificlly for ddpg and td3'''
    hidden_dims: Sequence[int]
    action_dim: int
    activation: str = "elu"
    init_scale: float = 0.01

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = MLP(self.hidden_dims, self.activation)(obs)
        action = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.uniform(scale=self.init_scale),
        )(x)
        return jnp.tanh(action)
    

class StochasticActor(nn.Module):
    '''
    we want gaussian policy: obs -> (mean, log_std) -> sample
    specifically used for ppo, sac, a2c, trpo
    '''
    action_dim: int
    activation: str = "elu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    state_dependent_std: bool = True # true for sac, false for ppo

    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool = False):
        x = MLP(self.hidden_dims, self.activation)(obs)

        mean = nn.Dense(self.action_dim)(x)

        if self.state_dependent_std:
            log_std = nn.Dense(self.action_dim)(x)
        else:
            log_std = self.param(
                "log_std",
                nn.initializers.zeros,
                (self.action_dim,),
            )

        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)

        dist = distrax.Normal(mean, std)
        if deterministic:
            action = mean
            log_prob = dist.log_prob(action).sum(-1)
        else:
            action = dist.sample(seed=self.make_rng("sample"))
            log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, mean, log_std
    

class TanhStochasticActor(nn.Module):
    '''compressed gaussian for SAC'''
    hidden_dims: Sequence[int]
    action_dim: int
    activation: str = "elu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool = False):
        x = MLP(self.hidden_dims, self.activation)(obs)
        mean = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(
            nn.Dense(self.action_dim)(x), self.log_std_min, self.log_std_max
        )

        dist = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )

        if deterministic:
            action = jnp.tanh(mean)
            log_prob = dist.log_prob(action)
        else:
            action, log_prob = dist.sample_and_log_prob(seed=self.make_rng("sample"))

        return action, log_prob


class Critic(nn.Module):
    '''state-value V(s) for on-policy methods'''
    hidden_dims: Sequence[int]
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = MLP(self.hidden_dims, self.activation)(obs)
        return nn.Dense(1)(x).squeeze(-1)


class QCritic(nn.Module):
    '''action-value Q(s,a) for off-policy methods'''
    hidden_dims: Sequence[int]
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(self.hidden_dims, self.activation)(x)
        return nn.Dense(1)(x).squeeze(-1)


class DoubleQCritic(nn.Module):
    '''twin Q-networks for TD3/SAC, min of two critics'''
    hidden_dims: Sequence[int]
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        q1 = QCritic(self.hidden_dims, self.activation)(obs, action)
        q2 = QCritic(self.hidden_dims, self.activation)(obs, action)
        return q1, q2


class DistributionalQCritic(nn.Module):
    '''C51-style distributional critic for FastTD3'''
    hidden_dims: Sequence[int]
    num_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(self.hidden_dims, self.activation)(x)
        logits = nn.Dense(self.num_atoms)(x)

        support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        probs = nn.softmax(logits, axis=-1)
        q_value = jnp.sum(probs * support, axis=-1)
        return q_value, logits, support


class AMPDiscriminator(nn.Module):
    '''AMP discriminator: (s_t, s_{t+1}) → real/fake logit'''
    hidden_dims: Sequence[int]
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs_t: jnp.ndarray, obs_t1: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs_t, obs_t1], axis=-1)
        x = MLP(self.hidden_dims, self.activation)(x)
        return nn.Dense(1)(x).squeeze(-1)