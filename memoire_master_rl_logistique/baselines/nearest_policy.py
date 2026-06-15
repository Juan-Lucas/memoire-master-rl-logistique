"""Politique Nearest Shovel (Greedy).

Baseline classique (Section 4.7.1 du mémoire) :
le camion choisit la pelle la plus proche géographiquement,
puis le dump le plus proche de cette pelle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class NearestShovelPolicy:
    """Choisit la pelle géographiquement la plus proche (greedy pur)."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str],
    ) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids
        self.dump_node_ids = dump_node_ids

    def _estimate_distance(self, src: str, dst: str) -> float:
        """Distance du plus court chemin via Dijkstra (Section 3.2)."""
        return self.graph.shortest_distance(src, dst)

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Retourne l'action encodant (pelle la plus proche, dump le plus proche)."""
        if truck_location is None:
            raise ValueError(
                "truck_location must be provided to NearestShovelPolicy.predict"
            )
        num_dumps = len(self.dump_node_ids)

        # Pelle la plus proche géographiquement (distance uniquement)
        best_shovel = 0
        min_distance = float("inf")

        for i, s_node in enumerate(self.shovel_node_ids):
            dist = self._estimate_distance(truck_location, s_node)
            if dist < min_distance:
                min_distance = dist
                best_shovel = i

        # Dump le plus proche de la pelle choisie
        best_dump = 0
        min_dump_distance = float("inf")
        shovel_node = self.shovel_node_ids[best_shovel]
        for j, d_node in enumerate(self.dump_node_ids):
            dist = self._estimate_distance(shovel_node, d_node)
            if dist < min_dump_distance:
                min_dump_distance = dist
                best_dump = j

        return encode_action(best_shovel, best_dump, num_dumps)
