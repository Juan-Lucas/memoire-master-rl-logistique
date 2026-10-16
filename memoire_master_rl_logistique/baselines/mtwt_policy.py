"""Règle du temps d'attente minimal des camions (MTWT, *Minimum Truck Waiting Time*), T23.

Règle exacte, à chaque décision (heure t, camion à la position p) :
1. **pelle** : pour chaque pelle j, heure d'arrivée prévue a_j = t + temps de trajet
   moyen à vide de p à j (``RoadGraph.mean_route_time_minutes``), puis attente prévue
   w_j = max(0, disponibilité de j − a_j). On choisit la pelle de plus petite attente,
   puis, en cas d'égalité, celle où l'on arrive le plus tôt ;
2. **verse** : même logique depuis la pelle choisie, avec le trajet en charge et une
   heure de départ estimée = max(a_j, disponibilité de j) + durée moyenne de chargement.

Différence avec la règle appelée FIFO (T22) : FIFO choisit la pelle libre le plus tôt
sans regarder le trajet ; MTWT compare **l'attente qu'aura le camion à son arrivée**,
donc combine files et temps de trajet, comme les systèmes commerciaux de gestion de
flotte (Zeng et al., 2022).

Disponibilité lue dans ``info`` : la libération **prévue** de chaque pelle et verse,
camions déjà en route compris (``MineEnv.forecast_free_at``, T15). Les camions qui
seront affectés après cette décision ne sont pas anticipés.

Depuis T16 bis, le camion décide à la verse où il vient de déverser (au parc pour son
premier cycle) : les trajets vers toutes les pelles sont comparables. Avant, il décidait
à la pelle qu'il venait de servir, et le trajet nul vers celle-ci la favorisait.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph

# Durée moyenne de chargement d'une pelle (min), comme ``Shovel.load_time_mean_min``.
DEFAULT_LOAD_TIME_MIN = 2.0


def _least_wait(arrivals: list[float], available_at: list[float]) -> int:
    """Indice de plus petite attente prévue, puis de plus petite heure d'arrivée."""
    waits = [max(0.0, free - arrival) for arrival, free in zip(arrivals, available_at)]
    return min(range(len(waits)), key=lambda k: (waits[k], arrivals[k]))


class MTWTPolicy:
    """Pelle puis verse de plus petite attente prévue à l'arrivée du camion."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str],
        load_time_min: float = DEFAULT_LOAD_TIME_MIN,
    ) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids
        self.dump_node_ids = dump_node_ids
        self.load_time_min = load_time_min

    def choose(
        self,
        truck_location: str,
        now: float,
        shovel_available_at: list[float],
        dump_available_at: list[float],
    ) -> tuple[int, int]:
        """(pelle, verse) choisies, en indices."""
        shovel_arrivals = [
            now + self.graph.mean_route_time_minutes(truck_location, node, loaded=False)
            for node in self.shovel_node_ids
        ]
        shovel = _least_wait(shovel_arrivals, shovel_available_at)

        departure = (max(shovel_arrivals[shovel], shovel_available_at[shovel])
                     + self.load_time_min)
        dump_arrivals = [
            departure + self.graph.mean_route_time_minutes(
                self.shovel_node_ids[shovel], node, loaded=True)
            for node in self.dump_node_ids
        ]
        return shovel, _least_wait(dump_arrivals, dump_available_at)

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Action (pelle, verse) de plus petite attente prévue ; action 0 sans ``info``."""
        if info is None or truck_location is None:
            return 0
        shovel, dump = self.choose(
            truck_location, info["current_time_min"],
            info["shovel_available_at"], info["dump_available_at"],
        )
        return encode_action(shovel, dump, len(self.dump_node_ids))
