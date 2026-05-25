"""Politique Nearest Shovel (Greedy).

Baseline classique (Section 4.7.1 du mémoire) :
le camion choisit la pelle la plus proche géographiquement,
puis le dump le plus proche de cette pelle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class NearestShovelPolicy:
    """Choisit la paire (pelle, dump) la plus proche."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids
        self.dump_node_ids = dump_node_ids or []

    def _estimate_distance(self, src: str, dst: str) -> float:
        """Distance du plus court chemin via Dijkstra (Section 3.2)."""
        return self.graph.shortest_distance(src, dst)

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str = "yard",
    ) -> int:
        """Retourne l'action encodant (pelle la plus proche, dump le plus proche)."""
        num_dumps = max(len(self.dump_node_ids), 1)

        # Pelle la plus proche du camion
        best_shovel = 0
        best_s_dist = float("inf")
        for i, s_node in enumerate(self.shovel_node_ids):
            dist = self._estimate_distance(truck_location, s_node)
            if dist < best_s_dist:
                best_s_dist = dist
                best_shovel = i

        # Dump le plus proche de la pelle choisie
        best_dump = 0
        if self.dump_node_ids:
            best_d_dist = float("inf")
            shovel_node = self.shovel_node_ids[best_shovel]
            for j, d_node in enumerate(self.dump_node_ids):
                dist = self._estimate_distance(shovel_node, d_node)
                if dist < best_d_dist:
                    best_d_dist = dist
                    best_dump = j

        return best_shovel * num_dumps + best_dump
