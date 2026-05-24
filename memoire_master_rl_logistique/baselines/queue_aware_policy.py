"""Politique Queue-Aware (sensible aux files d'attente).

Baseline avancée (Section 4.5 du mémoire) :
choisit la pelle dont le temps d'attente estimé est minimal.
Combine distance et disponibilité pour une décision plus informée.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.entities import Shovel
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class QueueAwarePolicy:
    """Choisit la pelle avec le temps d'attente estimé le plus court."""

    def __init__(
        self,
        graph: RoadGraph,
        shovels: list[Shovel],
    ) -> None:
        self.graph = graph
        self.shovels = shovels

    def _estimate_travel_time(self, src: str, dst: str) -> float:
        """Estime le temps de trajet entre deux nœuds."""
        base_speed = 32.0
        if (src, dst) in self.graph.edges:
            edge = self.graph.get_edge(src, dst)
            return (edge.distance_km / base_speed) * 60.0
        for mid in self.graph.nodes:
            if (src, mid) in self.graph.edges and (mid, dst) in self.graph.edges:
                d1 = self.graph.get_edge(src, mid).distance_km
                d2 = self.graph.get_edge(mid, dst).distance_km
                return ((d1 + d2) / base_speed) * 60.0
        return 30.0

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str = "yard",
        current_time_min: float = 0.0,
    ) -> int:
        """Retourne l'index de la pelle avec le temps total minimal."""
        best_action = 0
        best_total_time = float("inf")

        for i, shovel in enumerate(self.shovels):
            travel_time = self._estimate_travel_time(
                truck_location, shovel.node_id,
            )
            arrival_time = current_time_min + travel_time
            wait_time = max(0.0, shovel.available_at_min - arrival_time)
            total_time = travel_time + wait_time

            if total_time < best_total_time:
                best_total_time = total_time
                best_action = i

        return best_action
