"""Politique Shortest Path (Chemin le plus court).

Baseline classique (Section 4.7.1 du mémoire) :
le camion emprunte l'itinéraire minimisant le coût de cycle complet
C_ij (Eq. 3.1), en combinant temps de trajet, distance et consommation
de carburant sur les trois segments du cycle :
  camion → pelle (à vide) + pelle → dump (chargé) + dump → pelle (à vide).

Équation 3.1 :
  C_cycle = α · T_cycle + β · D_cycle + γ · E_cycle
  avec α=0.5, β=0.3, γ=0.2 (valeurs de la thèse, Section 3.2)

Différence avec Nearest Shovel :
  Nearest minimise uniquement d(camion→pelle).
  ShortestPath minimise le coût total du cycle, ce qui peut conduire à
  choisir une pelle plus loin si le segment pelle→dump est nettement
  plus court ou moins pentu.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.simulation.fuel_model import estimate_travel_fuel_l
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class ShortestPathPolicy:
    """Minimise le coût de cycle complet C = α·T + β·D + γ·E (Eq. 3.1)."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str],
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids
        self.dump_node_ids = dump_node_ids
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _segment_cost(self, src: str, dst: str, loaded: bool) -> tuple[float, float, float]:
        """Calcule (T, D, E) sur le plus court chemin entre deux nœuds.

        T : temps de trajet moyen (min) — déterministe, sans bruit.
        D : distance totale (km).
        E : consommation de carburant (L).
        """
        path = self.graph.shortest_path(src, dst)
        if len(path) < 2:
            return 0.0, 0.0, 0.0

        total_time = 0.0
        total_dist = 0.0
        total_fuel = 0.0

        for i in range(len(path) - 1):
            edge = self.graph.get_edge(path[i], path[i + 1])

            total_time += self.graph.mean_travel_time_minutes(
                path[i], path[i + 1], loaded,
            )
            total_dist += edge.distance_km

            total_fuel += estimate_travel_fuel_l(
                distance_km=edge.distance_km,
                slope_pct=edge.slope_pct,
                road_state=edge.road_state,
                loaded=loaded,
            )

        return total_time, total_dist, total_fuel

    def _cycle_cost(self, truck_loc: str, shovel_node: str, dump_node: str) -> float:
        """Coût total du cycle camion→pelle→dump→pelle selon Eq. 3.1."""
        t1, d1, e1 = self._segment_cost(truck_loc, shovel_node, loaded=False)
        t2, d2, e2 = self._segment_cost(shovel_node, dump_node, loaded=True)
        t3, d3, e3 = self._segment_cost(dump_node, shovel_node, loaded=False)

        T = t1 + t2 + t3
        D = d1 + d2 + d3
        E = e1 + e2 + e3

        return self.alpha * T + self.beta * D + self.gamma * E

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Retourne la paire (pelle, dump) avec le coût de cycle minimal (Eq. 3.1)."""
        if truck_location is None:
            raise ValueError(
                "truck_location must be provided to ShortestPathPolicy.predict"
            )
        num_dumps = len(self.dump_node_ids)
        best_action = 0
        best_cost = float("inf")

        for i, s_node in enumerate(self.shovel_node_ids):
            for j, d_node in enumerate(self.dump_node_ids):
                cost = self._cycle_cost(truck_location, s_node, d_node)

                if cost < best_cost:
                    best_cost = cost
                    best_action = encode_action(i, j, num_dumps)

        return best_action
