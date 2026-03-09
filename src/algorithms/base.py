"""Abstract base class for all RL algorithms."""

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base interface that all algorithms (JAX and PyTorch) must implement."""

    @abstractmethod
    def init(self, rng, obs_size: int, action_size: int):
        """Initialize parameters and optimizer state."""

    @abstractmethod
    def act(self, params, obs, rng, deterministic: bool = False):
        """Select an action given observation."""

    @abstractmethod
    def update(self, state, batch):
        """Perform a single gradient update and return updated state + metrics."""
