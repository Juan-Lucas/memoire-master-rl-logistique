"""Règle « pelle libre le plus tôt », appelée FIFO dans le projet et dans l'article.

Règle exacte (T22) : affecter le camion à la pelle qui sera libre le plus tôt
(``shovel_available_at`` minimal) et, indépendamment, à la verse qui sera libre le
plus tôt (``dump_available_at`` minimal). En cas d'égalité, l'indice le plus petit.

Ce que la règle ignore :
- **les temps de trajet** : une pelle libre dans 1 min mais à 10 min de trajet passe
  avant une pelle libre dans 2 min où se trouve déjà le camion. Les camions changent
  donc souvent de pelle, ce qui allonge leurs cycles (≈ 29,8 min, contre ≈ 24,1 min
  pour l'Affectation fixe en conditions nominales) ;
- la position du camion et l'ordre d'arrivée des camions.

Ce n'est donc **pas** un « premier arrivé, premier servi » au sens des files d'attente :
le nom FIFO est conservé pour la continuité avec la v1 (noms des politiques dans les CSV).
La règle qui tient compte à la fois des files et des trajets est MTWT (T23).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action


class FIFOPolicy:
    """Pelle libre le plus tôt et verse libre le plus tôt, sans tenir compte des trajets."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Action (pelle libre le plus tôt, verse libre le plus tôt), lue dans ``info``.

        Sans ``info`` (disponibilités inconnues), renvoie l'action 0 (pelle 1, verse 1).
        """
        shovel_times = [] if info is None else info.get("shovel_available_at", [])
        dump_times = [] if info is None else info.get("dump_available_at", [])

        if not shovel_times or not dump_times:
            return 0

        # Pelle libre le plus tôt (premier indice en cas d'égalité)
        best_shovel = int(np.argmin(shovel_times))

        # Verse libre le plus tôt, choisie indépendamment de la pelle
        best_dump = int(np.argmin(dump_times))

        return encode_action(best_shovel, best_dump, self.num_dumps)
