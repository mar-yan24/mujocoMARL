<p align="center">
  <img src="assets/banner.png" alt="humanoid_rl banner" width="100%"/>
</p>

<h1 align="center">humanoid_rl</h1>

<p align="center">
  <b>Multi-algorithm reinforcement learning for humanoid locomotion on MuJoCo Playground</b>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#algorithms">Algorithms</a> •
  <a href="#robots">Robots</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/JAX-0.4.30%2B-green" alt="JAX"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MuJoCo-3.2%2B-red" alt="MuJoCo"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License"/>
</p>

---

## What is this?

**humanoid_rl** is a unified codebase for training humanoid locomotion policies using multiple RL algorithms, built on top of [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Every algorithm ships in both a **from-scratch JAX implementation** (GPU-accelerated, fully JIT-compiled) and a **PyTorch implementation** for cross-framework benchmarking, with additional baselines from [Brax](https://github.com/google/brax) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

The goal is threefold:

1. **Learn deeply**: each algorithm is written from scratch with clear, readable code and no hidden abstractions.
2. **Benchmark rigorously**: compare algorithms, frameworks, and robots on identical environments with reproducible configs.
3. **Contribute upstream**: the codebase is architected to extract individual algorithms as PRs to MuJoCo Playground, which currently only ships PPO.

### Highlights

- **7 algorithms**: PPO, TD3 (with FastTD3 scaling), SAC, DDPG, A2C, TRPO, AMP
- **5 humanoid robots**: Unitree G1, Unitree H1, Berkeley Humanoid, Robotis OP3, Booster T1
- **Dual framework**: JAX-primary (native MJX) with PyTorch comparison
- **GPU-accelerated everything**: 4,096+ parallel environments, GPU-resident replay buffers, fully JIT-compiled training loops
- **Robot-agnostic**: swap robots via a single config flag; architecture supports quadrupeds and beyond
- **Adversarial Motion Priors**: AMP implementation for natural-looking gaits from motion capture data

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA 12.x + compatible NVIDIA GPU (recommended: RTX 3090/4090 or better)
- MuJoCo 3.2+

### Installation

```bash
git clone https://github.com/your-username/humanoid_rl.git
cd humanoid_rl
pip install -e .
```

> **GPU note:** For Ampere GPUs (RTX 30/40 series), set `JAX_DEFAULT_MATMUL_PRECISION=highest` to avoid TF32-induced training instability:
> ```bash
> export JAX_DEFAULT_MATMUL_PRECISION=highest
> ```

### Train your first policy

The fastest path to a working policy: PPO on the Berkeley Humanoid (simplest model, ~15 min on a single 4090):

```bash
python -m scripts.train \
    --algorithm ppo \
    --robot BerkeleyHumanoid \
    --terrain FlatTerrain \
    --num-envs 4096 \
    --total-timesteps 100000000 \
    --seed 0
```

### Evaluate a trained policy

```bash
python -m scripts.evaluate \
    checkpoints/checkpoint_100000000.pkl \
    --algorithm ppo \
    --robot BerkeleyHumanoid \
    --render
```

### Run a full benchmark

```bash
python -m scripts.benchmark \
    --algorithms ppo,td3,sac \
    --robots BerkeleyHumanoid,G1,H1 \
    --seeds 3 \
    --total-timesteps 50000000
```

---

## Algorithms

Every algorithm is implemented from scratch in JAX under `src/algorithms/{name}/jax/` and in PyTorch under `src/algorithms/{name}/torch/`. Library baselines (Brax, SB3) are available for comparison through the benchmark runner.

### On-Policy

| Algorithm | Description | JAX | PyTorch | Library Baseline |
|-----------|-------------|:---:|:-------:|:----------------:|
| **PPO** | Clipped surrogate with GAE, value clipping, entropy bonus | ✅ | ✅ | Brax PPO |
| **A2C** | Vanilla policy gradient with GAE (PPO without clipping) | ✅ | ✅ | SB3 A2C |
| **TRPO** | Natural gradient with KL constraint via conjugate gradient | ✅ | ✅ | SB3-contrib TRPO |

### Off-Policy

| Algorithm | Description | JAX | PyTorch | Library Baseline |
|-----------|-------------|:---:|:-------:|:----------------:|
| **TD3** | Twin delayed DDPG with FastTD3 scaling (distributional critic, 32K batches) | ✅ | ✅ | SB3 TD3 |
| **SAC** | Entropy-regularized with automatic temperature tuning | ✅ | ✅ | Brax SAC |
| **DDPG** | Deterministic policy gradient with target networks | ✅ | ✅ | SB3 DDPG |

### Imitation Learning

| Algorithm | Description | JAX | PyTorch | Library Baseline |
|-----------|-------------|:---:|:-------:|:----------------:|
| **AMP** | Adversarial Motion Priors: discriminator-augmented PPO for motion imitation | ✅ | 🚧 | LocoMuJoCo |

### Key references

- **PPO**: [PureJaxRL](https://github.com/luchris429/purejaxrl) nested `lax.scan` pattern
- **FastTD3**: [Seo et al. 2025](https://arxiv.org/abs/2505.22642), TD3 matching PPO wall-clock time with proper scaling
- **SAC**: [Brax SAC](https://github.com/google/brax) + [Stoix](https://github.com/EdanToledo/Stoix) GPU-resident buffer patterns
- **TRPO**: Fisher-vector products via `jax.jvp`/`jax.vjp` with CG solver
- **AMP**: [Peng et al. SIGGRAPH 2021](https://xbpeng.github.io/projects/AMP/) + [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) JAX implementation
- **Replay buffers**: [Flashbax](https://github.com/instadeepai/flashbax) for JAX-native GPU-resident buffers

---

## Robots

All robots are sourced from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) with MJX-compatible scene files. Environments use [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)'s `MjxEnv` base class with joystick velocity tracking tasks.

| Robot | Actuated DOF | Approx. Obs Dim | Sim-to-Real | Train Time (PPO, flat) |
|-------|:---:|:---:|:---:|:---:|
| **Berkeley Humanoid** | 12 | ~49 | ✅ | ~15 min |
| **Unitree H1** | 19 | ~70 | | ~20 min |
| **Robotis OP3** | 20 | ~73 | | ~20 min |
| **Booster T1** | 23 | ~82 | ✅ | ~25 min |
| **Unitree G1** | 29 | ~100 | ✅ | ~30 min |

> Train times measured on dual RTX 4090, 4,096 parallel envs, 100M timesteps.

### Adding a new robot

1. Ensure the robot has an MJX-compatible model in Menagerie (with `scene_mjx.xml`)
2. Create `configs/robot/{name}.yaml` with observation/action dimensions and PD gains
3. Register the Playground environment name (`{Robot}{Task}{Terrain}`)
4. Run training with no code changes needed:
   ```bash
   python -m scripts.train --algorithm ppo --robot YourNewRobot
   ```

---

## Architecture

```
humanoid_rl/
├── configs/
│   ├── algorithm/              # Per-algorithm hyperparameters (ppo.yaml, td3.yaml, ...)
│   ├── robot/                  # Per-robot model config (g1.yaml, h1.yaml, ...)
│   └── experiment/             # Full experiment configs (composes algo + robot)
│
├── src/
│   ├── algorithms/
│   │   ├── base.py             # Abstract BaseAlgorithm interface
│   │   ├── on_policy_base.py   # Shared: rollout collection, GAE, minibatch loop
│   │   ├── off_policy_base.py  # Shared: replay buffer, target networks, soft update
│   │   ├── ppo/
│   │   │   ├── jax/ppo.py      # From-scratch JAX PPO
│   │   │   └── torch/ppo.py    # From-scratch PyTorch PPO
│   │   ├── td3/                # (same jax/ + torch/ structure)
│   │   ├── sac/
│   │   ├── ddpg/
│   │   ├── a2c/
│   │   ├── trpo/
│   │   └── amp/
│   │
│   ├── networks/
│   │   ├── jax/mlp.py          # Flax: MLP, actors, critics, discriminator
│   │   ├── jax/factory.py      # create_actor(), create_critic() from config
│   │   └── torch/mlp.py        # nn.Module equivalents
│   │
│   ├── buffers/
│   │   ├── rollout_buffer.py   # On-policy: GAE via reverse lax.scan
│   │   └── replay_buffer.py    # Off-policy: GPU-resident fixed-size buffer
│   │
│   ├── envs/
│   │   └── wrappers.py         # PlaygroundEnvWrapper (auto-reset, domain rand)
│   │
│   ├── runners/
│   │   ├── on_policy_runner.py  # Nested lax.scan training loop
│   │   ├── off_policy_runner.py # Step > buffer > update loop
│   │   └── benchmark_runner.py  # Brax + SB3 baselines
│   │
│   └── utils/
│       ├── config.py            # Typed dataclass configs
│       ├── logger.py            # WandB + TensorBoard
│       └── checkpoint.py        # Save/load JAX pytrees
│
├── scripts/
│   ├── train.py                 # Main entry: config > runner > algorithm
│   ├── evaluate.py              # Rollout + render + metrics
│   └── benchmark.py             # Cross-algo x cross-robot x multi-seed
│
├── benchmarks/                  # Results, plots, comparison tables
├── notebooks/                   # Getting started, algorithm walkthroughs
└── tests/                       # Unit + integration tests
```

### Design principles

- **Algorithm / Runner separation**: Algorithms implement `init`, `act`, `update`. Runners handle environment interaction and data collection. Any algorithm plugs into any runner.
- **Framework-parallel**: Every algorithm has `jax/` and `torch/` directories with identical interfaces, enabling direct wall-clock and performance comparison.
- **Robot-agnostic**: The environment wrapper abstracts Playground's registry. Swap robots with a config change, no algorithm modifications needed.
- **Fully JIT-compiled** (JAX path): On-policy training compiles the entire collect > GAE > update loop into a single XLA program via nested `jax.lax.scan`. Off-policy training JIT-compiles step + buffer insert + update.
- **Reproducible configs**: Typed Python dataclasses with YAML overrides. Every experiment is fully specified by a single config file.

### Key JAX patterns used

| Pattern | Where | Why |
|---------|-------|-----|
| Nested `jax.lax.scan` | On-policy runner | Zero Python overhead; entire training loop compiles to XLA |
| `jax.vmap` | Environment stepping | Vectorize across 4,096+ parallel envs |
| Reverse `lax.scan` for GAE | Rollout buffer | Compile-time efficient advantage estimation |
| `optax.incremental_update` | Off-policy base | Polyak averaging for target networks |
| `jax.lax.cond` | TD3 update | Delayed policy updates without Python branching |
| `flax.struct.dataclass` | Buffers, state | Immutable JAX-compatible state containers |
| Pre-allocated fixed arrays | Replay buffer | JAX requires static shapes at compile time |
| `distrax.Transformed` | SAC actor | Tanh-squashed Gaussian with correct log-prob |

---

## Configuration

### YAML config structure

Configs compose three layers: algorithm, robot, and experiment.

```yaml
# configs/experiment/g1_td3_jax.yaml
algorithm: td3
framework: jax
seed: 0
total_timesteps: 100_000_000
lr: 3e-4
gamma: 0.99

env:
  robot: G1
  task: Joystick
  terrain: FlatTerrain
  num_envs: 4096
  domain_randomization: true

network:
  actor_hidden: [512, 256, 128]
  critic_hidden: [1024, 512, 256]
  activation: elu

td3:
  tau: 0.005
  policy_delay: 2
  batch_size: 32768
  distributional: true       # FastTD3 C51 critic
  num_atoms: 101
```

### CLI overrides

Any config value can be overridden from the command line:

```bash
python -m scripts.train \
    --algorithm td3 \
    --robot G1 \
    --num-envs 8192 \
    --total-timesteps 200000000 \
    --seed 42
```

---

## Benchmarks

### Algorithm comparison (Berkeley Humanoid, Flat Terrain)

> 📊 Results populated after first full benchmark run. Run with:
> ```bash
> python -m scripts.benchmark --algorithms ppo,td3,sac,ddpg,a2c --robots BerkeleyHumanoid --seeds 5
> ```

| Algorithm | Mean Reward | Train Time | FPS (env steps/sec) |
|-----------|:-----------:|:----------:|:-------------------:|
| PPO (JAX) | | | |
| TD3 (JAX, FastTD3) | | | |
| SAC (JAX) | | | |
| DDPG (JAX) | | | |
| A2C (JAX) | | | |
| PPO (Brax baseline) | | | |

### Cross-robot comparison (PPO, Flat Terrain)

| Robot | Mean Reward | Train Time | Notes |
|-------|:-----------:|:----------:|:-----:|
| Berkeley Humanoid | | | Fastest convergence |
| Unitree H1 | | | |
| Robotis OP3 | | | |
| Booster T1 | | | |
| Unitree G1 | | | Most complex |

### JAX vs PyTorch wall-clock comparison

| Algorithm | JAX (4096 envs) | PyTorch (4096 envs) | Speedup |
|-----------|:---:|:---:|:---:|
| PPO | | | |
| TD3 | | | |
| SAC | | | |

---

## Roadmap

### Phase 1: Foundation ✅
- [x] Project structure and config system
- [x] Flax network modules (actors, critics, discriminator)
- [x] Rollout buffer with GAE via reverse `lax.scan`
- [x] GPU-resident replay buffer
- [x] Playground environment wrapper
- [x] On-policy and off-policy runners

### Phase 2: Core Algorithms (In Progress)
- [ ] PPO (JAX): end-to-end training on BerkeleyHumanoid
- [ ] TD3 (JAX): with FastTD3 distributional critic + large batch scaling
- [ ] SAC (JAX): with auto-alpha tuning
- [ ] DDPG (JAX): simplified TD3 baseline

### Phase 3: Extended Algorithms
- [ ] A2C (JAX): PPO without clipping
- [ ] TRPO (JAX): CG solver + Fisher-vector products
- [ ] PyTorch implementations for all above

### Phase 4: AMP & Imitation Learning
- [ ] AMP discriminator training loop
- [ ] Motion data loading (CMU MoCap / AMASS)
- [ ] Reference motion retargeting to humanoid morphologies
- [ ] GAIL baseline

### Phase 5: Polish & Benchmarks
- [ ] Full benchmark matrix across all algorithms x robots x terrains
- [ ] Rough terrain training with domain randomization
- [ ] Quadruped support (Go1, Anymal) to validate robot-agnostic architecture
- [ ] WandB dashboard and comparison plots
- [ ] Comprehensive documentation and algorithm walkthroughs

### Stretch Goals
- [ ] Upstream contributions to MuJoCo Playground (TD3/SAC/AMP training scripts)
- [ ] Sim-to-real deployment scripts for supported robots
- [ ] Multi-skill AMP with diverse motion datasets

---

## Development

### Running tests

```bash
pytest tests/ -v
```

### Code style

```bash
ruff check src/ scripts/
ruff format src/ scripts/
```

### Adding a new algorithm

1. Create `src/algorithms/{name}/jax/{name}.py` (and optionally `torch/`)
2. Inherit from `OnPolicyBase` or `OffPolicyBase`
3. Implement `_policy_loss`, `_critic_loss` (on-policy) or `_actor_loss`, `_critic_loss` (off-policy)
4. Add config dataclass to `src/utils/config.py`
5. Register in `ALGO_MAP` in `scripts/train.py`
6. Add YAML config to `configs/algorithm/{name}.yaml`
7. Write tests in `tests/test_{name}.py`

---

## Acknowledgments

This project builds on top of exceptional open-source work:

- [MuJoCo](https://github.com/google-deepmind/mujoco) (physics engine)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (GPU-accelerated RL environments)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (robot models)
- [Brax](https://github.com/google/brax) (JAX RL algorithms and MJX integration)
- [PureJaxRL](https://github.com/luchris429/purejaxrl) (fully JIT-compiled RL patterns)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) (single-file reference implementations)
- [Flashbax](https://github.com/instadeepai/flashbax) (JAX-native replay buffers)
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) (locomotion benchmarks with AMP/GAIL)
- [FastTD3](https://arxiv.org/abs/2505.22642) (off-policy scaling for humanoid control)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PyTorch RL baselines)

---

## License

MIT

---

<p align="center">
  Built for learning, benchmarking, and contributing to the MuJoCo ecosystem.
</p>