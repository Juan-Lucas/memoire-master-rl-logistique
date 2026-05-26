"""Politique FIFO (First In, First Out).

Baseline classique (Section 4.7.1 du mémoire) :
les camions sont affectés aux pelles dans l'ordre de leur arrivée
en zone d'attente. Cette règle est simple et équitable mais ne tient
compte ni de la distance ni de la charge des pelles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.entities import Shovel


class FIFOPolicy:
    """Affecte les camions aux pelles dans l'ordre d'arrivée (FIFO)."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps
        self._counter = 0

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne l'action encodant la paire (pelle, dump) selon FIFO."""
        # Sélection cyclique des pelles (round-robin)
        shovel_idx = self._counter % self.num_shovels
        dump_idx = self._counter % self.num_dumps
        self._counter += 1
        return shovel_idx * self.num_dumps + dump_idx
