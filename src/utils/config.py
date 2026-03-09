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
    # fasttd3 distro critic?
    distributional: bool = True
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
    speed: int = 0
    total_timesteps: int = 100_000_000
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
    a2c: A2CConfig = field(default_factory=A2CConfig)
    trpo: TRPOConfig = field(default_factory=TRPOConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)


def load_config(path: str | Path) -> TrainConfig:
    # load a yaml config and load it into a terrainconfig
    with open(path) as f:
        raw = yaml.safe_load(f)
    #  TODO: recursive merge into trainconfig dataclass
    raise NotImplementedError


def make_env_name(cfg: EnvConfig) -> str:
    # construct playground registry name: robot,task,terrain
    return f"{cfg.robot}{cfg.task}{cfg.terrain}"
