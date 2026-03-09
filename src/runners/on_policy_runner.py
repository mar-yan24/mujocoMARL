"""On-policy training runner using nested jax.lax.scan.

Compiles a chunk of N update steps into a single XLA program,
then runs chunks in a Python loop for periodic logging/checkpointing.
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import flax.struct

from src.algorithms.on_policy_base import OnPolicyBase, OnPolicyState
from src.envs.wrappers import PlaygroundEnvWrapper, EnvState
from src.buffers.rollout_buffer import (
    RolloutData,
    RolloutBatch,
    compute_gae,
)
from src.utils.config import TrainConfig
from src.utils.logger import Logger
from src.utils.checkpoint import save_checkpoint


@flax.struct.dataclass
class RunnerState:
    '''full state carried through the training scan'''
    train_state: OnPolicyState
    env_state: EnvState
    last_obs: jax.Array # (num_envs, obs_dim)
    rng: jax.Array


def make_train_step(
    algo: OnPolicyBase,
    env: PlaygroundEnvWrapper,
    config: TrainConfig,
    steps_per_call: int,
):
    '''
    builds a JIT-compiled function that runs `steps_per_call` update iterations.
    each iteration: collect rollout (inner scan) -> GAE -> algo.update.
    returns: (RunnerState, metrics_dict)
    '''
    algo_cfg = algo._get_algo_config()
    rollout_steps = config.rollout_steps
    gamma = config.gamma
    gae_lambda = getattr(algo_cfg, 'gae_lambda', 0.95)

    def _train_chunk(runner_state: RunnerState):

        def _update_step(runner_state: RunnerState, _):
            # -- collect rollout --
            def _env_step(carry, _):
                train_state, env_state, obs, rng = carry
                rng, act_rng = jax.random.split(rng)

                action, log_prob = algo.act(
                    train_state.actor_params, obs, act_rng,
                )
                value = algo.value(train_state.critic_params, obs)

                next_env_state = env.step(env_state, action)

                transition = RolloutData(
                    obs=obs,
                    action=action,
                    reward=next_env_state.reward,
                    done=next_env_state.done,
                    log_prob=log_prob,
                    value=value,
                )
                return (train_state, next_env_state, next_env_state.obs, rng), transition

            carry = (
                runner_state.train_state,
                runner_state.env_state,
                runner_state.last_obs,
                runner_state.rng,
            )
            carry, transitions = jax.lax.scan(
                _env_step, carry, None, length=rollout_steps,
            )
            train_state, env_state, last_obs, rng = carry

            # -- compute GAE --
            last_value = algo.value(train_state.critic_params, last_obs)
            advantages, returns = compute_gae(
                transitions.reward,
                transitions.value,
                transitions.done,
                last_value,
                gamma=gamma,
                lam=gae_lambda,
            )

            batch = RolloutBatch(
                obs=transitions.obs,
                action=transitions.action,
                log_prob=transitions.log_prob,
                advantage=advantages,
                returns=returns,
                value=transitions.value,
            )

            # -- update --
            train_state, metrics = algo.update(train_state, batch)

            # track episode reward from env info if available
            new_runner_state = RunnerState(
                train_state=train_state,
                env_state=env_state,
                last_obs=last_obs,
                rng=rng,
            )
            return new_runner_state, metrics

        runner_state, chunk_metrics = jax.lax.scan(
            _update_step, runner_state, None, length=steps_per_call,
        )
        # average metrics over the chunk
        avg_metrics = jax.tree.map(lambda x: x.mean(), chunk_metrics)
        return runner_state, avg_metrics

    return jax.jit(_train_chunk)


def run_training(
    algo: OnPolicyBase,
    env: PlaygroundEnvWrapper,
    config: TrainConfig,
    logger: Logger | None = None,
):
    '''
    training loop with periodic logging and checkpointing.
    compiles a chunk of update steps, then calls it in a Python loop.
    '''
    num_envs = config.env.num_envs
    rollout_steps = config.rollout_steps
    steps_per_update = num_envs * rollout_steps
    num_updates = config.total_timesteps // steps_per_update

    # how many updates per JIT call (trade off: fewer = more logging, more = faster)
    steps_per_call = min(config.log_interval, num_updates)
    num_calls = num_updates // steps_per_call
    # handle remainder
    if num_updates % steps_per_call != 0:
        num_calls += 1

    print(f"[runner] {num_updates} total updates, {steps_per_call} per JIT call, {num_calls} calls")
    print(f"[runner] compiling...")

    train_step = make_train_step(algo, env, config, steps_per_call)

    # -- init --
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    train_state = algo.init(init_rng)
    env_state = env.reset(env_rng)

    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_state,
        last_obs=env_state.obs,
        rng=rng,
    )

    # -- training loop --
    updates_done = 0
    for call_idx in range(num_calls):
        # handle last chunk possibly being shorter
        remaining = num_updates - updates_done
        if remaining < steps_per_call:
            # recompile for the smaller chunk
            train_step = make_train_step(algo, env, config, remaining)

        runner_state, metrics = train_step(runner_state)
        jax.block_until_ready(metrics)

        updates_done += min(steps_per_call, remaining)
        global_step = updates_done * steps_per_update

        # log
        metrics_float = jax.tree.map(float, metrics)
        if logger is not None:
            logger.log(metrics_float, step=global_step)
            logger.print_metrics(metrics_float, global_step, config.total_timesteps)
        else:
            parts = [f"step {global_step}/{config.total_timesteps}"]
            for k, v in metrics_float.items():
                parts.append(f"{k}={v:.4f}")
            print(" | ".join(parts))

        # checkpoint
        if (call_idx + 1) % config.save_interval == 0:
            save_checkpoint(
                runner_state.train_state,
                f"{config.checkpoint_dir}/checkpoint_{global_step}.pkl",
                step=global_step,
            )

    # final checkpoint
    save_checkpoint(
        runner_state.train_state,
        f"{config.checkpoint_dir}/final.pkl",
    )

    print("[runner] training done!")
    return runner_state, metrics
