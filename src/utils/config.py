"""Typed dataclass configs with YAML loading and CLI override support."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml

class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"

class AlgorithmType(Enum):
    PPO = "ppo"
    TD3 = "td3"
    SAC = "sac"
    DDPG = "ddpg"
    A2C = "a2c"
    TRPO = "trpo"
    AMP = "amp"


@dataclass
class NetworkConfig:
    actor_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [1024, 512, 256])
    activation: str = "elu" # elu|relu|tanh
    init_scale: float = 0.01 # final layer init scale
    log_std_min: float = -5.0 # stochastic policy
    log_std_max: float = 2.0


'''
A lot of these RL training dataclass have sorta arbitrary values
edit as you like
'''


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2 
    entropy_coeff: float = 0.01 
    value_coeff: float = 0.5
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 5
    max_grad_norm: float = 2.0
    normalize_advantages: bool = True
    targe_kl: Optional[float] = None # early stop


@dataclass
class TD3Config:
    tau: float = 0.005 #target update rate
    policy_delay: int = 2 # delayed policy updates
    target_noise: float = 0.2 # target policy smoothing noise
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1
    batch_size: int = 32768 #large batch -- test
    buffer_size: int = 1_000_000
    learning_starts: int = 25_000
    # fasttd3 distributional critic (experimental, not yet wired into loss)
    distributional: bool = False
    num_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0


@dataclass
class SACConfig:
    tau: float = 0.005
    init_alpha: float = 1.0
    auto_alpha: bool = True # auto entropy tuning
    target_entropy: Optional[float] = None # None = -dim(A)
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000


@dataclass
class DDPGConfig:
    tau: float = 0.005
    exploration_noise: float = 0.1
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000


@dataclass
class A2CConfig:
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


@dataclass
class TRPOConfig:
    gae_lambda: float = 0.95
    max_kl: float = 0.01 # kl constraint
    cg_iters: int = 10 # cg_iters = conjugate gradient iterations
    cg_damping: float = 0.1
    line_search_steps: int = 10
    value_coeff: float = 0.5


@dataclass
class AMPConfig:
    disc_hidden: list[int] = field(default_factory=lambda: [1024, 512])
    disc_lr: float = 1e-5
    disc_gradient_penalty: float = 5.0
    disc_weight_decay: float = 0.0001
    task_reward_weight: float = 0.5
    style_reward_weight: float = 0.5
    replay_buf_size: int = 1_000_000
    motion_data_path: str = "" #path to reference .npz


@dataclass
class EnvConfig:
    robot: str = "G1" # choose between playground models: g1|h1|berkeleyhumanoid|op3|boostert1
    task: str = "Joystick"
    terrain: str = "FlatTerrain" #for now just gonna do flatterrain, can change later
    num_envs: int = 4096 # too much?
    episode_length: int = 1000
    action_repeat: int = 1
    domain_randomization: bool = True


@dataclass
class TrainConfig:
    algorithm: AlgorithmType = AlgorithmType.PPO
    framework: Framework = Framework.JAX
    seed: int = 0
    total_timesteps: int = 100_000_000
    rollout_steps: int = 10 # env steps per update (on-policy)
    lr: float = 3e-4
    gamma: float = 0.99
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "humanoid-rl"
    wandb_entity: Optional[str] = None

    #subconfigs
    network: NetworkConfig = field(default_factory=NetworkConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    sac: SACConfig = field(default_factory=SACConfig)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    trpo: TRPOConfig = field(default_factory=TRPOConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)


def _merge_into_dataclass(dc, overrides: dict):
    '''recursively merge a dict into a dataclass instance'''
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(current, Enum):
            setattr(dc, key, type(current)(value))
        elif hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | Path) -> TrainConfig:
    '''load a yaml config and merge it into a TrainConfig'''
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = TrainConfig()
    if raw:
        _merge_into_dataclass(cfg, raw)
    return cfg


def make_env_name(cfg: EnvConfig) -> str:
    '''construct playground registry name: {Robot}{Task}{Terrain}'''
    return f"{cfg.robot}{cfg.task}{cfg.terrain}"


def get_algo_config(cfg: TrainConfig):
    '''return the algorithm-specific sub-config'''
    mapping = {
        AlgorithmType.PPO: cfg.ppo,
        AlgorithmType.TD3: cfg.td3,
        AlgorithmType.SAC: cfg.sac,
        AlgorithmType.DDPG: cfg.ddpg,
        AlgorithmType.A2C: cfg.a2c,
        AlgorithmType.TRPO: cfg.trpo,
        AlgorithmType.AMP: cfg.amp,
    }
    return mapping[cfg.algorithm]


def parse_args() -> TrainConfig:
    '''parse CLI args into a TrainConfig'''
    import argparse
    parser = argparse.ArgumentParser(description="humanoid_rl training")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--framework", type=str, default="jax")
    parser.add_argument("--robot", type=str, default="BerkeleyHumanoid")
    parser.add_argument("--task", type=str, default="Joystick")
    parser.add_argument("--terrain", type=str, default="FlatTerrain")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--wandb-project", type=str, default="humanoid-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # start from yaml or defaults
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = TrainConfig()

    # CLI overrides
    cfg.algorithm = AlgorithmType(args.algorithm)
    cfg.framework = Framework(args.framework)
    cfg.seed = args.seed
    cfg.total_timesteps = args.total_timesteps
    cfg.rollout_steps = args.rollout_steps
    cfg.lr = args.lr
    cfg.gamma = args.gamma
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.env.robot = args.robot
    cfg.env.task = args.task
    cfg.env.terrain = args.terrain
    cfg.env.num_envs = args.num_envs

    return cfg
