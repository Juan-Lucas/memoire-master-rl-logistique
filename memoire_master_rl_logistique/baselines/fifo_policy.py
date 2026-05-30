"""Politique FIFO (First In, First Out).

Baseline classique (Section 4.7.1 du mémoire) :
les camions sont affectés à la pelle dont la file d'attente est la plus
courte, c'est-à-dire la pelle disponible le plus tôt (idle depuis le plus
longtemps). Cette règle est simple et équitable mais ne tient compte ni
de la distance ni de la charge de travail globale.

Principe FIFO appliqué au dispatching :
  parmi toutes les pelles, on sert en premier celle qui attend des camions
  depuis le plus longtemps (available_at_min la plus petite = file la plus
  courte). Le dump le plus tôt disponible est choisi indépendamment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FIFOPolicy:
    """Baseline FIFO : affecter à la pelle la plus sous-utilisée (idle la plus longue)."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne la paire (pelle la plus disponible, dump le plus disponible).

        La pelle dont available_at_min est la plus petite est celle qui a
        attendu le plus longtemps sans camion → principe FIFO de service.
        """
        shovel_times = [] if info is None else info.get("shovel_available_at", [])
        dump_times = [] if info is None else info.get("dump_available_at", [])

        if not shovel_times or not dump_times:
            return 0

        # Pelle la plus sous-utilisée : available_at_min minimale
        best_shovel = int(np.argmin(shovel_times))

        # Dump le plus tôt disponible (indépendant)
        best_dump = int(np.argmin(dump_times))

        return best_shovel * self.num_dumps + best_dump
