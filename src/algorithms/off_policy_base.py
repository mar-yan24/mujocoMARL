"""Shared base for off-policy algorithms (TD3, SAC, DDPG).

Handles replay buffer interaction, target networks, and soft (Polyak) updates.
"""

from src.algorithms.base import BaseAlgorithm


class OffPolicyBase(BaseAlgorithm):
    """Base class for off-policy algorithms."""

    pass
