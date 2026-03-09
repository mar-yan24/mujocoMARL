"""Shared base for on-policy algorithms (PPO, A2C, TRPO).

Handles rollout collection, GAE computation, and minibatch update loop.
"""

from src.algorithms.base import BaseAlgorithm


class OnPolicyBase(BaseAlgorithm):
    """Base class for on-policy algorithms."""

    pass
