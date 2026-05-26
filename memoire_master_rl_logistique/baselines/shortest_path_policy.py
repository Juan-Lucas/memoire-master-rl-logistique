"""Politique Shortest Path (Chemin le plus court).

Baseline classique (Section 4.7.1 du mémoire) :
le camion emprunte systématiquement l'itinéraire minimisant
le coût de trajet C_ij (Eq. 3.1), indépendamment de l'état courant
des autres camions ou des files d'attente. Cette règle optimise
localement le routage sans considération globale du système.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class ShortestPathPolicy:
    """Choisit la paire (pelle, dump) avec le coût de trajet minimal."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.shovel_node_ids = shovel_node_ids
        self.dump_node_ids = dump_node_ids or []

    def _estimate_cost(self, src: str, dst: str) -> float:
        """Estime le coût de trajet selon Eq. 3.1 du mémoire.
        
        C_ij = α·T_ij + β·D_ij + γ·E_ij
        
        Pour simplifier, nous utilisons la distance pondérée
        comme proxy du coût total.
        """
        return self.graph.shortest_distance(src, dst)

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str = "yard",
    ) -> int:
        """Retourne l'action encodant (pelle, dump) avec coût minimal."""
        num_dumps = max(len(self.dump_node_ids), 1)

        best_action = 0
        best_total_cost = float("inf")

        dump_list = list(enumerate(self.dump_node_ids)) if self.dump_node_ids else [(0, None)]

        for i, s_node in enumerate(self.shovel_node_ids):
            cost_to_shovel = self._estimate_cost(truck_location, s_node)

            for j, d_node in dump_list:
                if d_node is not None:
                    cost_shovel_to_dump = self._estimate_cost(s_node, d_node)
                    total_cost = cost_to_shovel + cost_shovel_to_dump
                else:
                    total_cost = cost_to_shovel

                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_action = i * num_dumps + j

        return best_action
