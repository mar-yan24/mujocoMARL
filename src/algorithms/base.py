"""Abstract base class for all RL algorithms."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import jax

class BaseAlgorithm(ABC):
    """Base interface that all algorithms (JAX and PyTorch) must implement."""

    def __init__(self, obs_dim: int, action_dim: int, config: Any):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

    @abstractmethod
    def init(self, rng, obs_size: int, action_size: int):
        """Initialize parameters and optimizer state."""

    @abstractmethod
    def act(self, params, obs, rng, deterministic: bool = False):
        """Select an action given observation."""

    @abstractmethod
    def update(self, state, batch):
        """Perform a single gradient update and return updated state + metrics."""
