"""Baselines de comparaison pour le dispatching minier (Section 4.7)."""

from .base import BaselinePolicy, encode_action
from .fifo_policy import FIFOPolicy
from .fixed_policy import FixedAssignmentPolicy
from .nearest_policy import NearestShovelPolicy
from .shortest_path_policy import ShortestPathPolicy

__all__ = [
    "BaselinePolicy",
    "FIFOPolicy",
    "FixedAssignmentPolicy",
    "NearestShovelPolicy",
    "ShortestPathPolicy",
    "encode_action",
]