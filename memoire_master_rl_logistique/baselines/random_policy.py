"""Politique aléatoire simple.

Baseline de comparaison (Section 4.7.1 du mémoire) :
choisit une action aléatoire uniformément parmi toutes les actions valides.
Cette politique ne nécessite aucune information sur l'état de l'environnement.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomPolicy:
    """Choisit une action aléatoire uniformément."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps
        self.num_actions = num_shovels * num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Retourne une action aléatoire."""
        return int(np.random.randint(0, self.num_actions))
