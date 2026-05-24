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

    def _estimate_distance(self, src: str, dst: str) -> float:
        """Estime la distance entre deux nœuds via les arêtes disponibles."""
        if (src, dst) in self.graph.edges:
            return self.graph.get_edge(src, dst).distance_km
        for mid in self.graph.nodes:
            if (src, mid) in self.graph.edges and (mid, dst) in self.graph.edges:
                return (
                    self.graph.get_edge(src, mid).distance_km
                    + self.graph.get_edge(mid, dst).distance_km
                )
        return float("inf")

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
            dist = self._estimate_distance(truck_location, shovel_node)
            if dist < best_distance:
                best_distance = dist
                best_action = i

        return best_action
