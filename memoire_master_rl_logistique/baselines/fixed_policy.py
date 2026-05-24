"""Politique d'assignation fixe (Fixed Assignment).

Baseline de niveau zéro (Section 4.5 du mémoire) :
chaque camion est dédié à une seule pelle de manière cyclique.
Le dump est également assigné de manière cyclique.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FixedAssignmentPolicy:
    """Chaque camion est assigné à une paire (pelle, dump) fixe."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne l'action encodant la paire (pelle, dump) round-robin."""
        truck_idx = 0
        if info is not None:
            truck_idx = info.get("current_truck_idx", 0)
        shovel_idx = truck_idx % self.num_shovels
        dump_idx = truck_idx % self.num_dumps
        return shovel_idx * self.num_dumps + dump_idx
