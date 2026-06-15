"""Politique d'assignation fixe (Fixed Assignment).

Baseline de niveau zéro (Section 4.7.1 du mémoire) :
chaque camion est dédié à une pelle selon un schéma round-robin statique
(affectation par modulo du nombre de pelles). Le dump est assigné de la
même manière, par modulo du nombre de dumps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action


class FixedAssignmentPolicy:
    """Assignation statique camion → (pelle, dump) par round-robin sur l'ID physique."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Retourne l'action encodant la paire (pelle, dump) fixe pour ce camion."""
        truck_idx = 0 if info is None else info.get("current_truck_idx", 0)
        shovel_idx = truck_idx % self.num_shovels
        dump_idx = truck_idx % self.num_dumps
        return encode_action(shovel_idx, dump_idx, self.num_dumps)
