"""Evaluate a trained policy: rollout + render + metrics.

Usage:
    python -m scripts.evaluate checkpoints/final.pkl --algorithm ppo --robot BerkeleyHumanoid
    python -m scripts.evaluate checkpoints/final.pkl --algorithm ppo --robot BerkeleyHumanoid --render
"""

from __future__ import annotations
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.config import (
    TrainConfig,
    AlgorithmType,
    Framework,
    make_env_name,
)
from src.utils.checkpoint import load_checkpoint
from src.envs.wrappers import PlaygroundEnvWrapper


def parse_eval_args():
    parser = argparse.ArgumentParser(description="evaluate a trained policy")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint .pkl")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--framework", type=str, default="jax")
    parser.add_argument("--robot", type=str, default="BerkeleyHumanoid")
    parser.add_argument("--task", type=str, default="Joystick")
    parser.add_argument("--terrain", type=str, default="FlatTerrain")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_eval_args()

    config = TrainConfig()
    config.algorithm = AlgorithmType(args.algorithm)
    config.framework = Framework(args.framework)
    config.env.robot = args.robot
    config.env.task = args.task
    config.env.terrain = args.terrain
    config.seed = args.seed

    # load checkpoint
    train_state, step = load_checkpoint(args.checkpoint)
    print(f"[eval] loaded checkpoint from step {step}")

    # create env with single instance for eval
    env_name = make_env_name(config.env)
    env = PlaygroundEnvWrapper(env_name, num_envs=1, domain_randomization=False)

    # create algorithm (for act method)
    from scripts.train import ALGO_MAP, _import_class
    key = (config.algorithm, config.framework)
    algo_cls = _import_class(ALGO_MAP[key])
    algo = algo_cls(obs_dim=env.obs_size, action_dim=env.action_size, config=config)

    # init models so we have the model objects (we'll use loaded params)
    rng = jax.random.PRNGKey(config.seed)
    _ = algo.init(rng)

    # run evaluation episodes
    rewards_all = []
    lengths_all = []

    for ep in range(args.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        env_state = env.reset(reset_rng)
        obs = env_state.obs
        total_reward = 0.0
        length = 0

        for t in range(config.env.episode_length):
            rng, act_rng = jax.random.split(rng)
            result = algo.act(train_state.actor_params, obs, act_rng, deterministic=True)
            # on-policy returns (action, log_prob), off-policy returns just action
            action = result[0] if isinstance(result, tuple) else result
            env_state = env.step(env_state, action)
            obs = env_state.obs
            total_reward += float(env_state.reward.squeeze())
            length += 1

            if float(env_state.done.squeeze()) > 0.5:
                break

        rewards_all.append(total_reward)
        lengths_all.append(length)
        print(f"  episode {ep+1}/{args.num_episodes}: reward={total_reward:.2f} length={length}")

    print(f"\n[eval] {args.num_episodes} episodes:")
    print(f"  mean reward: {np.mean(rewards_all):.2f} +/- {np.std(rewards_all):.2f}")
    print(f"  mean length: {np.mean(lengths_all):.1f}")

    if args.render:
        print("[eval] rendering not yet implemented (needs mujoco viewer integration)")


if __name__ == "__main__":
    main()
