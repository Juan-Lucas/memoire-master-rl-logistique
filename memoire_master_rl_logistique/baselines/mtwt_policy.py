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
from memoire_master_rl_logistique.simulation.constants import LOAD_TIME_MEAN_MIN
from memoire_master_rl_logistique.simulation.forecast import forecast_route
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph

# Durée moyenne de chargement d'une pelle (min), celle du simulateur.
DEFAULT_LOAD_TIME_MIN = LOAD_TIME_MEAN_MIN


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
        """(pelle, verse) choisies, en indices : la pelle de plus petite attente prévue
        (puis d'arrivée la plus tôt), puis, depuis elle, la verse de plus petite attente
        prévue (puis d'arrivée la plus tôt). Calcul commun avec l'observation des agents
        (``simulation.forecast.forecast_route``)."""
        routes = {
            (j, k): forecast_route(self.graph, truck_location, now, s_node, d_node,
                                   shovel_available_at[j], dump_available_at[k],
                                   self.load_time_min)
            for j, s_node in enumerate(self.shovel_node_ids)
            for k, d_node in enumerate(self.dump_node_ids)
        }
        shovel = min(range(len(self.shovel_node_ids)),
                     key=lambda j: (routes[j, 0].shovel_wait, routes[j, 0].shovel_arrival))
        dump = min(range(len(self.dump_node_ids)),
                   key=lambda k: (routes[shovel, k].dump_wait, routes[shovel, k].dump_arrival))
        return shovel, dump

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
