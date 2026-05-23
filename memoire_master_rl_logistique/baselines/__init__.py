"""Baselines de comparaison pour le dispatching minier (Section 4.5)."""

from .fixed_policy import FixedAssignmentPolicy
from .nearest_policy import NearestShovelPolicy
from .queue_aware_policy import QueueAwarePolicy

__all__ = [
    "FixedAssignmentPolicy",
    "NearestShovelPolicy",
    "QueueAwarePolicy",
]
