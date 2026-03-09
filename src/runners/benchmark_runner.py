"""Benchmark runner integrating Brax and Stable-Baselines3 baselines.

Provides a unified interface to run library implementations alongside
our from-scratch ones for wall-clock and performance comparison.
"""

from __future__ import annotations
from dataclasses import dataclass
import time

import jax

from src.envs.wrappers import BraxCompatWrapper
from src.utils.config import TrainConfig, make_env_name


@dataclass
class BenchmarkResult:
    algorithm: str
    framework: str
    robot: str
    seed: int
    mean_reward: float
    train_time_sec: float
    fps: float


def run_brax_ppo(config: TrainConfig) -> BenchmarkResult:
    '''run brax's built-in PPO on a playground env'''
    from brax.training.agents.ppo import train as brax_ppo_train

    env_name = make_env_name(config.env)
    brax_env = BraxCompatWrapper(env_name).env

    t0 = time.time()
    make_inference_fn, params, metrics = brax_ppo_train(
        environment=brax_env,
        num_timesteps=config.total_timesteps,
        episode_length=config.env.episode_length,
        num_envs=config.env.num_envs,
        learning_rate=config.lr,
        seed=config.seed,
    )
    train_time = time.time() - t0

    reward = float(metrics.get("eval/episode_reward", 0.0))
    fps = config.total_timesteps / max(train_time, 1e-6)

    return BenchmarkResult(
        algorithm="brax_ppo",
        framework="jax",
        robot=config.env.robot,
        seed=config.seed,
        mean_reward=reward,
        train_time_sec=train_time,
        fps=fps,
    )


def run_brax_sac(config: TrainConfig) -> BenchmarkResult:
    '''run brax's built-in SAC on a playground env'''
    from brax.training.agents.sac import train as brax_sac_train

    env_name = make_env_name(config.env)
    brax_env = BraxCompatWrapper(env_name).env

    t0 = time.time()
    make_inference_fn, params, metrics = brax_sac_train(
        environment=brax_env,
        num_timesteps=config.total_timesteps,
        episode_length=config.env.episode_length,
        num_envs=min(config.env.num_envs, 256),  # SAC typically uses fewer envs
        learning_rate=config.lr,
        seed=config.seed,
    )
    train_time = time.time() - t0

    reward = float(metrics.get("eval/episode_reward", 0.0))
    fps = config.total_timesteps / max(train_time, 1e-6)

    return BenchmarkResult(
        algorithm="brax_sac",
        framework="jax",
        robot=config.env.robot,
        seed=config.seed,
        mean_reward=reward,
        train_time_sec=train_time,
        fps=fps,
    )


def run_sb3_baseline(algo_name: str, config: TrainConfig) -> BenchmarkResult:
    '''run a stable-baselines3 algorithm on a playground env (via gymnasium)'''
    import gymnasium as gym

    # SB3 algorithms
    if algo_name == "sb3_ppo":
        from stable_baselines3 import PPO as SB3PPO
        model_cls = SB3PPO
    elif algo_name == "sb3_td3":
        from stable_baselines3 import TD3 as SB3TD3
        model_cls = SB3TD3
    elif algo_name == "sb3_sac":
        from stable_baselines3 import SAC as SB3SAC
        model_cls = SB3SAC
    elif algo_name == "sb3_ddpg":
        from stable_baselines3 import DDPG as SB3DDPG
        model_cls = SB3DDPG
    elif algo_name == "sb3_a2c":
        from stable_baselines3 import A2C as SB3A2C
        model_cls = SB3A2C
    elif algo_name == "sb3_trpo":
        from sb3_contrib import TRPO as SB3TRPO
        model_cls = SB3TRPO
    else:
        raise ValueError(f"unknown sb3 algo: {algo_name}")

    env_name = make_env_name(config.env)
    # NOTE: requires a gym-compatible wrapper for playground envs
    # this is a placeholder -- actual gym registration would be needed
    print(f"[benchmark] SB3 baselines require gym-registered envs (not yet wired for {env_name})")
    return BenchmarkResult(
        algorithm=algo_name,
        framework="torch",
        robot=config.env.robot,
        seed=config.seed,
        mean_reward=0.0,
        train_time_sec=0.0,
        fps=0.0,
    )
