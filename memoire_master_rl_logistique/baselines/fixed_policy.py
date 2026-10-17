"""Affectation fixe (Fixed Assignment).

Règle exacte (T22) : le camion d'indice i (0, 1, 2, …) va toujours à la pelle
i mod S et à la verse i mod D (S = 3 pelles, D = 2 verses), pendant tout le poste,
**sans aucune réaffectation** : ni selon les files, ni après une panne. L'état
(observation, files, position) n'est jamais lu, seul l'indice du camion compte.

Conséquences :
- la flotte est répartie à parts égales : 4 camions par pelle avec 12 camions,
  6 avec 18 (avec 13 camions, la pelle 1 en a 5 : scénario « nominal_asymétrique ») ;
- après son premier cycle, un camion revient toujours à sa propre pelle : il ne fait
  jamais de trajet entre pelles, d'où des cycles courts.

Une réaffectation de secours (par exemple quand une pelle tombe en panne) n'a de sens
qu'avec des pannes de pelles (T16, reporté) ; elle n'est pas implémentée.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action


class FixedAssignmentPolicy:
    """Camion i → pelle i mod S, verse i mod D, pendant tout le poste, sans réaffectation."""

    def __init__(self, num_shovels: int, num_dumps: int = 1) -> None:
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Action (pelle, verse) fixe du camion courant (``info["current_truck_idx"]``).

        Sans ``info``, le camion d'indice 0 est supposé : action 0 (pelle 1, verse 1).
        """
        truck_idx = 0 if info is None else info.get("current_truck_idx", 0)
        shovel_idx = truck_idx % self.num_shovels
        dump_idx = truck_idx % self.num_dumps
        return encode_action(shovel_idx, dump_idx, self.num_dumps)
