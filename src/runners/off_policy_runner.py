"""Off-policy training runner: step > buffer insert > update loop.

Separates env stepping from parameter updates to avoid
jax.lax.cond pytree structure mismatches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.algorithms.off_policy_base import OffPolicyBase, OffPolicyState
from src.envs.wrappers import PlaygroundEnvWrapper, EnvState
from src.buffers.replay_buffer import (
    ReplayBufferState,
    create_replay_buffer,
    add_batch,
)
from src.utils.config import TrainConfig
from src.utils.logger import Logger
from src.utils.checkpoint import save_checkpoint


def run_training(
    algo: OffPolicyBase,
    env: PlaygroundEnvWrapper,
    config: TrainConfig,
    logger: Logger | None = None,
):
    '''
    off-policy training loop.
    uses a Python for-loop with two JIT'd functions:
      1. _env_step: act, step env, insert into buffer
      2. _update: sample batch, compute grads, apply
    warmup check is done in Python to avoid cond pytree issues.
    '''
    algo_cfg = algo._get_algo_config()
    num_envs = config.env.num_envs
    total_steps = config.total_timesteps // num_envs
    learning_starts = algo_cfg.learning_starts // num_envs  # convert to step count

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    train_state = algo.init(init_rng)
    env_state = env.reset(env_rng)
    replay_buf = create_replay_buffer(algo_cfg.buffer_size, env.obs_size, env.action_size)
    obs = env_state.obs

    # JIT: env step + buffer insert
    @jax.jit
    def _env_step(train_state, env_state, obs, replay_buf, rng):
        rng, act_rng = jax.random.split(rng)
        action = algo.act(train_state.actor_params, obs, act_rng)
        next_env_state = env.step(env_state, action)
        replay_buf = add_batch(
            replay_buf, obs, action,
            next_env_state.reward, next_env_state.obs, next_env_state.done,
        )
        return next_env_state, next_env_state.obs, replay_buf, rng

    # JIT: sample + update
    @jax.jit
    def _update(train_state, replay_buf, rng):
        return algo.update(train_state, replay_buf)

    print(f"[runner] starting off-policy training "
          f"({total_steps} steps, warmup={learning_starts})")

    for step in range(total_steps):
        rng, step_rng = jax.random.split(rng)

        # env step
        env_state, obs, replay_buf, _ = _env_step(
            train_state, env_state, obs, replay_buf, step_rng,
        )

        # update after warmup (Python-level check avoids cond pytree issues)
        if step >= learning_starts:
            rng, update_rng = jax.random.split(rng)
            train_state, metrics = _update(train_state, replay_buf, update_rng)

            global_step = (step + 1) * num_envs

            # logging
            if logger is not None and step % config.log_interval == 0:
                metrics_float = jax.tree.map(float, metrics)
                logger.log(metrics_float, step=global_step)
                logger.print_metrics(metrics_float, global_step, config.total_timesteps)

        # checkpoint
        global_step = (step + 1) * num_envs
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(
                train_state,
                f"{config.checkpoint_dir}/checkpoint_{global_step}.pkl",
                step=global_step,
            )

    save_checkpoint(train_state, f"{config.checkpoint_dir}/final.pkl")
    print("[runner] off-policy training done")
    return train_state
