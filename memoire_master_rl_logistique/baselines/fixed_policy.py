"""Politique d'assignation fixe (Fixed Assignment).

Baseline de niveau zéro (Section 4.7.1 du mémoire) :
chaque camion est dédié à une seule pelle de manière cyclique.
Le dump est également assigné de manière cyclique.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FixedAssignmentPolicy:
    """Assignation VRAIMENT statique : basée sur l'ID physique du camion."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps
        # Mapping fixe par ID physique (T1→P1, T2→P2, T3→P3, T4→P1, ...)
        # Ce mapping est calculé une seule fois et ne change JAMAIS
        self._static_map = {}

    def _get_assignment(self, truck_id: int) -> int:
        """Retourne l'action fixe pour un ID de camion physique."""
        if truck_id not in self._static_map:
            shovel_idx = truck_id % self.num_shovels
            dump_idx = truck_id % self.num_dumps
            self._static_map[truck_id] = shovel_idx * self.num_dumps + dump_idx
        return self._static_map[truck_id]

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne l'action encodant la paire (pelle, dump) fixe pour ce camion."""
        # IMPORTANT : on utilise l'ID physique du camion (0 à N-1)
        # pas current_truck_idx qui change selon la disponibilité
        truck_idx = 0
        if info is not None:
            truck_idx = info.get("current_truck_idx", 0)
        return self._get_assignment(truck_idx)
