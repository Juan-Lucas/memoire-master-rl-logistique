"""Politique Nearest Shovel (Greedy).

Baseline classique (Section 4.5 du mémoire) :
le camion choisit la pelle la plus proche géographiquement.
C'est l'approche la plus courante en l'absence de système intelligent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class NearestShovelPolicy:
    """Choisit la pelle la plus proche du camion courant."""

    def __init__(self, graph: RoadGraph, shovel_node_ids: list[str]) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str = "yard",
    ) -> int:
        """Retourne l'index de la pelle la plus proche."""
        best_action = 0
        best_distance = float("inf")

        for i, shovel_node in enumerate(self.shovel_node_ids):
            src = truck_location
            if src not in ["yard", "dump_1"]:
                src = "yard"
            if (src, shovel_node) in self.graph.edges:
                edge = self.graph.get_edge(src, shovel_node)
                if edge.distance_km < best_distance:
                    best_distance = edge.distance_km
                    best_action = i

        return best_action
