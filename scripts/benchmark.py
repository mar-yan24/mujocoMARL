"""Cross-algorithm x cross-robot x multi-seed benchmark runner.

Usage:
    python -m scripts.benchmark --algorithms ppo,td3,sac --robots BerkeleyHumanoid,G1,H1 --seeds 3
    python -m scripts.benchmark --algorithms ppo --robots BerkeleyHumanoid --seeds 1 --total-timesteps 10000000
"""

from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
from dataclasses import asdict

from src.utils.config import (
    TrainConfig,
    AlgorithmType,
    Framework,
    make_env_name,
)
from src.utils.logger import Logger
from src.envs.wrappers import PlaygroundEnvWrapper
from src.runners.benchmark_runner import BenchmarkResult


def parse_benchmark_args():
    parser = argparse.ArgumentParser(description="run benchmark suite")
    parser.add_argument(
        "--algorithms", type=str, default="ppo",
        help="comma-separated list of algorithms (e.g. ppo,td3,sac,brax_ppo)",
    )
    parser.add_argument(
        "--robots", type=str, default="BerkeleyHumanoid",
        help="comma-separated list of robots",
    )
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds to run")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    return parser.parse_args()


def run_single(algo_name: str, robot: str, seed: int, args) -> BenchmarkResult:
    '''run a single algorithm/robot/seed combination'''
    config = TrainConfig()
    config.seed = seed
    config.total_timesteps = args.total_timesteps
    config.env.num_envs = args.num_envs
    config.env.robot = robot

    # handle library baselines
    if algo_name.startswith("brax_"):
        from src.runners.benchmark_runner import run_brax_ppo, run_brax_sac
        if algo_name == "brax_ppo":
            return run_brax_ppo(config)
        elif algo_name == "brax_sac":
            return run_brax_sac(config)
    elif algo_name.startswith("sb3_"):
        from src.runners.benchmark_runner import run_sb3_baseline
        return run_sb3_baseline(algo_name, config)

    # our from-scratch implementations
    config.algorithm = AlgorithmType(algo_name)
    config.framework = Framework.JAX

    env_name = make_env_name(config.env)
    env = PlaygroundEnvWrapper(
        env_name, num_envs=config.env.num_envs,
        domain_randomization=config.env.domain_randomization,
    )

    from scripts.train import ALGO_MAP, ON_POLICY, _import_class
    key = (config.algorithm, config.framework)
    algo_cls = _import_class(ALGO_MAP[key])
    algo = algo_cls(obs_dim=env.obs_size, action_dim=env.action_size, config=config)

    t0 = time.time()
    if config.algorithm in ON_POLICY:
        from src.runners.on_policy_runner import run_training
        runner_state, metrics = run_training(algo, env, config)
    else:
        from src.runners.off_policy_runner import run_training
        run_training(algo, env, config)
        metrics = {}
    train_time = time.time() - t0

    fps = config.total_timesteps / max(train_time, 1e-6)

    return BenchmarkResult(
        algorithm=algo_name,
        framework="jax",
        robot=robot,
        seed=seed,
        mean_reward=0.0,  # TODO: extract from eval run
        train_time_sec=train_time,
        fps=fps,
    )


def main():
    args = parse_benchmark_args()
    algorithms = args.algorithms.split(",")
    robots = args.robots.split(",")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    total_runs = len(algorithms) * len(robots) * args.seeds
    run_idx = 0

    for algo_name in algorithms:
        for robot in robots:
            for seed in range(args.seeds):
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"[benchmark] run {run_idx}/{total_runs}: "
                      f"{algo_name} / {robot} / seed={seed}")
                print(f"{'='*60}")

                try:
                    result = run_single(algo_name, robot, seed, args)
                    results.append(result)
                    print(f"  -> reward={result.mean_reward:.2f} "
                          f"time={result.train_time_sec:.1f}s "
                          f"fps={result.fps:.0f}")
                except Exception as e:
                    print(f"  -> FAILED: {e}")

    # save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[benchmark] results saved to {results_path}")

    # print summary table
    print(f"\n{'='*80}")
    print(f"{'Algorithm':<15} {'Robot':<20} {'Reward':>10} {'Time (s)':>10} {'FPS':>10}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r.algorithm:<15} {r.robot:<20} {r.mean_reward:>10.2f} "
              f"{r.train_time_sec:>10.1f} {r.fps:>10.0f}")


if __name__ == "__main__":
    main()
