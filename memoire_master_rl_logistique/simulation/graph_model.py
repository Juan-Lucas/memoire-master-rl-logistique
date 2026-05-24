"""Modèle de graphe routier minier G = (V, E).

Le réseau routier est modélisé comme un graphe orienté pondéré
(Section 3.2 du mémoire). Chaque arête porte la distance, la pente
et l'état de surface, permettant de calculer les temps de trajet
stochastiques (Eq. 3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from random import Random


@dataclass(slots=True)
class RoadNode:
    """Nœud du graphe routier (pelle, dump, intersection, yard)."""

    node_id: str
    node_type: str


@dataclass(slots=True)
class RoadEdge:
    """Arête orientée avec caractéristiques physiques."""

    src: str
    dst: str
    distance_km: float
    slope_pct: float
    road_state: float


class RoadGraph:
    """Graphe orienté pondéré du réseau routier minier."""

    def __init__(self) -> None:
        self.nodes: dict[str, RoadNode] = {}
        self.edges: dict[tuple[str, str], RoadEdge] = {}

    def add_node(self, node_id: str, node_type: str) -> None:
        """Ajoute un nœud (lève ValueError si doublon)."""
        if node_id in self.nodes:
            msg = f"Node {node_id} already exists."
            raise ValueError(msg)
        self.nodes[node_id] = RoadNode(node_id=node_id, node_type=node_type)

    def add_edge(
        self,
        src: str,
        dst: str,
        distance_km: float,
        slope_pct: float,
        road_state: float = 1.0,
    ) -> None:
        """Ajoute une arête orientée entre deux nœuds existants."""
        if src not in self.nodes:
            msg = f"Source node {src} does not exist."
            raise ValueError(msg)
        if dst not in self.nodes:
            msg = f"Destination node {dst} does not exist."
            raise ValueError(msg)
        self.edges[(src, dst)] = RoadEdge(
            src=src,
            dst=dst,
            distance_km=distance_km,
            slope_pct=slope_pct,
            road_state=road_state,
        )

    def get_edge(self, src: str, dst: str) -> RoadEdge:
        """Récupère l'arête ou lève ValueError."""
        if (src, dst) not in self.edges:
            msg = f"Edge from {src} to {dst} does not exist."
            raise ValueError(msg)
        return self.edges[(src, dst)]

    def sample_travel_time_minutes(
        self, src: str, dst: str, loaded: bool, rng: Random,
    ) -> float:
        """Échantillonne un temps de trajet log-normal (Eq. 3.2).

        T_ij(t) = T̄_ij + η_ij(t), implémenté via une distribution
        log-normale tenant compte de la vitesse, pente et état route.
        """
        edge = self.get_edge(src, dst)
        base_speed_kmh = 26.0 if loaded else 32.0

        slope_factor = max(0.55, 1.0 - max(0.0, edge.slope_pct) * 0.03)
        state_factor = max(0.55, min(1.2, edge.road_state))
        mean_speed_kmh = max(5.0, base_speed_kmh * slope_factor * state_factor)
        mean_time_min = (edge.distance_km / mean_speed_kmh) * 60.0

        sigma = 0.12
        mu = log(max(0.2, mean_time_min)) - 0.5 * sigma * sigma
        return max(0.2, rng.lognormvariate(mu, sigma))


def build_mine_graph(
    shovel_count: int = 3,
    dump_count: int = 2,
) -> RoadGraph:
    """Construit un graphe routier minier réaliste et paramétrable.

    Topologie : yard central → intersection → pelles et dumps,
    avec arêtes bidirectionnelles. Les distances et pentes sont
    réalistes (basées sur les données Afrapoli 2019, Mohtasham 2023).
    """
    graph = RoadGraph()

    # Nœuds
    graph.add_node("yard", "yard")
    graph.add_node("junction_1", "junction")

    for i in range(1, shovel_count + 1):
        graph.add_node(f"shovel_{i}", "shovel")

    for i in range(1, dump_count + 1):
        graph.add_node(f"dump_{i}", "dump")

    # Yard ↔ Junction
    graph.add_edge("yard", "junction_1", distance_km=0.8, slope_pct=2.0)
    graph.add_edge("junction_1", "yard", distance_km=0.8, slope_pct=1.0)

    # Junction ↔ Pelles
    shovel_params = [
        (1.5, 4.0),  # (distance_km, slope_pct)
        (1.8, 5.0),
        (2.2, 6.0),
        (2.5, 3.0),
        (1.9, 7.0),
    ]
    for i in range(1, shovel_count + 1):
        idx = (i - 1) % len(shovel_params)
        dist, slope = shovel_params[idx]
        node = f"shovel_{i}"
        graph.add_edge("junction_1", node, distance_km=dist, slope_pct=slope)
        graph.add_edge(node, "junction_1", distance_km=dist, slope_pct=max(1.0, slope - 2.0))

    # Junction ↔ Dumps
    dump_params = [
        (2.8, 6.0),
        (3.2, 8.0),
        (2.5, 5.0),
    ]
    for i in range(1, dump_count + 1):
        idx = (i - 1) % len(dump_params)
        dist, slope = dump_params[idx]
        node = f"dump_{i}"
        graph.add_edge("junction_1", node, distance_km=dist, slope_pct=slope)
        graph.add_edge(node, "junction_1", distance_km=dist, slope_pct=max(1.0, slope - 3.0))

    # Pelles → Dumps (trajets chargés directs)
    for si in range(1, shovel_count + 1):
        for di in range(1, dump_count + 1):
            s_node = f"shovel_{si}"
            d_node = f"dump_{di}"
            s_idx = (si - 1) % len(shovel_params)
            d_idx = (di - 1) % len(dump_params)
            dist = (shovel_params[s_idx][0] + dump_params[d_idx][0]) * 0.8
            slope = (shovel_params[s_idx][1] + dump_params[d_idx][1]) / 2.0
            graph.add_edge(s_node, d_node, distance_km=dist, slope_pct=slope)

    # Dumps → Pelles (retour à vide)
    for di in range(1, dump_count + 1):
        for si in range(1, shovel_count + 1):
            d_node = f"dump_{di}"
            s_node = f"shovel_{si}"
            d_idx = (di - 1) % len(dump_params)
            s_idx = (si - 1) % len(shovel_params)
            dist = (dump_params[d_idx][0] + shovel_params[s_idx][0]) * 0.8
            slope = max(1.0, (dump_params[d_idx][1] + shovel_params[s_idx][1]) / 2.0 - 2.0)
            graph.add_edge(d_node, s_node, distance_km=dist, slope_pct=slope)

    return graph
