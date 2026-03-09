"""Main training entry point: config > runner > algorithm.

Usage:
    python -m scripts.train --algorithm ppo --robot BerkeleyHumanoid --num-envs 4096
    python -m scripts.train --config configs/experiment/berkeley_ppo_jax.yaml
"""

from __future__ import annotations
import os

from src.utils.config import (
    TrainConfig,
    AlgorithmType,
    Framework,
    parse_args,
    make_env_name,
    get_algo_config,
)
from src.utils.logger import Logger
from src.envs.wrappers import PlaygroundEnvWrapper

# algorithm registry: maps (AlgorithmType, Framework) -> class
ALGO_MAP = {
    (AlgorithmType.PPO, Framework.JAX): "src.algorithms.ppo.jax.ppo.PPO",
    (AlgorithmType.A2C, Framework.JAX): "src.algorithms.a2c.jax.a2c.A2C",
    (AlgorithmType.TRPO, Framework.JAX): "src.algorithms.trpo.jax.trpo.TRPO",
    (AlgorithmType.TD3, Framework.JAX): "src.algorithms.td3.jax.td3.TD3",
    (AlgorithmType.SAC, Framework.JAX): "src.algorithms.sac.jax.sac.SAC",
    (AlgorithmType.DDPG, Framework.JAX): "src.algorithms.ddpg.jax.ddpg.DDPG",
    (AlgorithmType.AMP, Framework.JAX): "src.algorithms.amp.jax.amp.AMP",
}

# on-policy vs off-policy for runner selection
ON_POLICY = {AlgorithmType.PPO, AlgorithmType.A2C, AlgorithmType.TRPO, AlgorithmType.AMP}
OFF_POLICY = {AlgorithmType.TD3, AlgorithmType.SAC, AlgorithmType.DDPG}


def _import_class(dotted_path: str):
    '''dynamically import a class from a dotted module path'''
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    config = parse_args()

    # set JAX precision for ampere GPUs
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

    print(f"[train] algorithm={config.algorithm.value} "
          f"framework={config.framework.value} "
          f"robot={config.env.robot} "
          f"num_envs={config.env.num_envs} "
          f"total_timesteps={config.total_timesteps}")

    # create env
    env_name = make_env_name(config.env)
    print(f"[train] env_name={env_name}")
    env = PlaygroundEnvWrapper(
        env_name,
        num_envs=config.env.num_envs,
        domain_randomization=config.env.domain_randomization,
    )

    # create algorithm
    key = (config.algorithm, config.framework)
    if key not in ALGO_MAP:
        raise ValueError(
            f"algorithm {config.algorithm.value} not implemented for {config.framework.value}. "
            f"available: {[f'{a.value}/{f.value}' for a, f in ALGO_MAP.keys()]}"
        )

    algo_cls = _import_class(ALGO_MAP[key])
    algo = algo_cls(
        obs_dim=env.obs_size,
        action_dim=env.action_size,
        config=config,
    )

    # create logger
    import dataclasses
    run_name = f"{config.algorithm.value}_{config.env.robot}_s{config.seed}"
    logger = Logger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=run_name,
        log_dir=f"logs/{run_name}",
        use_wandb=config.wandb_entity is not None,
        use_tb=True,
        config=dataclasses.asdict(config),
    )

    # run training
    if config.algorithm in ON_POLICY:
        from src.runners.on_policy_runner import run_training
        run_training(algo, env, config, logger=logger)
    else:
        from src.runners.off_policy_runner import run_training
        run_training(algo, env, config, logger=logger)

    logger.close()
    print("[train] done!")


if __name__ == "__main__":
    main()
