"""Politique d'assignation fixe (Fixed Assignment).

Baseline de niveau zéro (Section 4.5 du mémoire) :
chaque camion est dédié à une seule pelle de manière cyclique.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FixedAssignmentPolicy:
    """Chaque camion est assigné à une pelle fixe (round-robin)."""

    def __init__(self, num_shovels: int) -> None:
        self.num_shovels = num_shovels

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne l'action (index de pelle) basée sur l'index du camion."""
        truck_idx = 0
        if info is not None:
            truck_idx = info.get("current_truck_idx", 0)
        return truck_idx % self.num_shovels
