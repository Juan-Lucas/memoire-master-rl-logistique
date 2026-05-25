"""Politique Queue-Aware (sensible aux files d'attente).

Baseline avancée (Section 4.7.1 du mémoire) :
choisit la paire (pelle, dump) dont le temps d'attente total
estimé est minimal. Combine distance et disponibilité pour
une décision plus informée.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.entities import DumpSite, Shovel
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class QueueAwarePolicy:
    """Choisit la paire (pelle, dump) avec le temps total minimal."""

    def __init__(
        self,
        graph: RoadGraph,
        shovels: list[Shovel],
        dumps: list[DumpSite] | None = None,
    ) -> None:
        self.graph = graph
        self.shovels = shovels
        self.dumps = dumps or []

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
        """Retourne l'action encodant la paire (pelle, dump) optimale."""
        num_dumps = max(len(self.dumps), 1)
        best_action = 0
        best_total_time = float("inf")

        dump_list = list(enumerate(self.dumps)) if self.dumps else [(0, None)]

        for i, shovel in enumerate(self.shovels):
            travel_to_shovel = self._estimate_travel_time(
                truck_location, shovel.node_id,
            )
            arrival_shovel = current_time_min + travel_to_shovel
            wait_shovel = max(0.0, shovel.available_at_min - arrival_shovel)

            for j, dump in dump_list:
                if dump is not None:
                    travel_to_dump = self._estimate_travel_time(
                        shovel.node_id, dump.node_id,
                    )
                    arrival_dump = (
                        arrival_shovel + wait_shovel + 2.0 + travel_to_dump
                    )
                    wait_dump = max(0.0, dump.available_at_min - arrival_dump)
                    total_time = (
                        travel_to_shovel + wait_shovel
                        + travel_to_dump + wait_dump
                    )
                else:
                    total_time = travel_to_shovel + wait_shovel

                if total_time < best_total_time:
                    best_total_time = total_time
                    best_action = i * num_dumps + j

        return best_action
