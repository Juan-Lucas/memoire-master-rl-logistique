"""Règle FIFO de dispatching : servir d'abord la pelle qui attend un camion depuis le plus
longtemps.

Règle exacte (T22) : affecter le camion à la pelle de plus petite heure de libération
(``shovel_available_at``) et, indépendamment, à la verse de plus petite heure de
libération (``dump_available_at``). En cas d'égalité, l'indice le plus petit.

Pour une pelle déjà libre, cette heure est l'instant, passé, où elle s'est libérée
(``MineEnv.forecast_free_at`` ne la ramène pas à l'heure courante). D'où :
- si des pelles attendent un camion, la règle choisit celle qui attend **depuis le plus
  longtemps** : c'est la règle FIFO (« premier arrivé, premier servi ») du dispatching
  minier, appliquée aux pelles qui attendent ;
- si toutes les pelles sont occupées, elle choisit celle qui se libérera le plus tôt
  (camions en route compris).

Ce que la règle ignore :
- **les temps de trajet** : une pelle libre dans 1 min mais à 10 min de trajet passe
  avant une pelle libre dans 2 min où se trouve déjà le camion. Les camions changent
  donc souvent de pelle, ce qui allonge leurs cycles (≈ 29,8 min, contre ≈ 24,1 min
  pour l'Affectation fixe en conditions nominales) ;
- la position du camion.

Correction du 22 octobre 2026 : cette docstring disait que la règle n'était « pas un
premier arrivé, premier servi », en pensant à l'ordre de service des camions dans une
file (que le simulateur respecte pour toutes les politiques). Au sens du dispatching,
défini dans l'article (§2), c'est bien la règle FIFO. La règle qui tient compte à la
fois des files et des trajets est MTWT (T23).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action


class FIFOPolicy:
    """Pelle (et verse) qui attend depuis le plus longtemps, ou libre le plus tôt ; sans les trajets."""

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
