"""Baselines de comparaison pour le dispatching minier (Section 4.7)."""

from .fifo_policy import FIFOPolicy
from .fixed_policy import FixedAssignmentPolicy
from .nearest_policy import NearestShovelPolicy
from .random_policy import RandomPolicy
from .shortest_path_policy import ShortestPathPolicy

__all__ = [
    "FIFOPolicy",
    "FixedAssignmentPolicy",
    "NearestShovelPolicy",
    "RandomPolicy",
    "ShortestPathPolicy",
]
