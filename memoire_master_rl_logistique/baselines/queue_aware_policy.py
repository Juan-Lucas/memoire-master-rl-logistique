"""Politique Queue-Aware (sensible aux files d'attente).

Baseline avancée (Sprint C, Section plan_codage) :
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

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str = "yard",
        current_time_min: float = 0.0,
    ) -> int:
        """Retourne l'index de la pelle avec le temps d'arrivée + attente minimal."""
        best_action = 0
        best_total_time = float("inf")

        for i, shovel in enumerate(self.shovels):
            # Estimation du temps de trajet
            src = truck_location
            if src not in ["yard", "dump_1"]:
                src = "yard"

            travel_time = 0.0
            if (src, shovel.node_id) in self.graph.edges:
                edge = self.graph.get_edge(src, shovel.node_id)
                base_speed = 32.0  # km/h à vide
                travel_time = (edge.distance_km / base_speed) * 60.0

            # Estimation du temps d'attente à la pelle
            arrival_time = current_time_min + travel_time
            wait_time = max(0.0, shovel.available_at_min - arrival_time)

            total_time = travel_time + wait_time

            if total_time < best_total_time:
                best_total_time = total_time
                best_action = i

        return best_action
