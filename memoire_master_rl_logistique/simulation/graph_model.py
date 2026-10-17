"""Graphe routier de la mine G = (V, E), construit avec NetworkX.

Graphe orienté : chaque arête porte la distance, la pente et l'état de la
piste, qui déterminent le temps de trajet (log-normal) et la consommation.
Les plus courts chemins sont calculés par Dijkstra sur la distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from random import Random

import networkx as nx

from memoire_master_rl_logistique.simulation.constants import SLOPE_PENALTY_PER_PCT


@dataclass(slots=True)
class RoadEdge:
    """Arête orientée avec caractéristiques physiques."""

    src: str
    dst: str
    distance_km: float
    slope_pct: float
    road_state: float


class RoadGraph:
    """Graphe orienté pondéré du réseau routier (enveloppe un ``nx.DiGraph``).

    Les arêtes sont aussi gardées dans un cache de ``RoadEdge`` : ce sont ces
    objets, et non les attributs NetworkX, que ``mine_env`` modifie quand une
    piste se dégrade.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._edge_cache: dict[tuple[str, str], RoadEdge] = {}

    @property
    def nodes(self) -> nx.classes.reportviews.NodeView:
        """Vue sur les nœuds (itérable, ``node_id in graph.nodes``)."""
        return self._g.nodes

    @property
    def edges(self) -> nx.classes.reportviews.OutEdgeView:
        """Vue sur les arêtes (itérable, ``(src, dst) in graph.edges``)."""
        return self._g.edges

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Accès direct au DiGraph NetworkX (pour visualisation)."""
        return self._g

    def add_node(self, node_id: str, node_type: str) -> None:
        """Ajoute un nœud (lève ValueError si doublon)."""
        if node_id in self._g:
            msg = f"Node {node_id} already exists."
            raise ValueError(msg)
        self._g.add_node(node_id, node_type=node_type)

    def add_edge(
        self,
        src: str,
        dst: str,
        distance_km: float,
        slope_pct: float,
        road_state: float = 1.0,
    ) -> None:
        """Ajoute une arête orientée entre deux nœuds existants."""
        if src not in self._g:
            msg = f"Source node {src} does not exist."
            raise ValueError(msg)
        if dst not in self._g:
            msg = f"Destination node {dst} does not exist."
            raise ValueError(msg)
        self._g.add_edge(
            src, dst,
            distance_km=distance_km,
            slope_pct=slope_pct,
            road_state=road_state,
        )
        edge = RoadEdge(
            src=src, dst=dst,
            distance_km=distance_km,
            slope_pct=slope_pct,
            road_state=road_state,
        )
        self._edge_cache[(src, dst)] = edge

    def get_edge(self, src: str, dst: str) -> RoadEdge:
        """Récupère l'arête ou lève ValueError."""
        edge = self._edge_cache.get((src, dst))
        if edge is None:
            msg = f"Edge from {src} to {dst} does not exist."
            raise ValueError(msg)
        return edge

    def shortest_path(
        self, src: str, dst: str, weight: str = "distance_km",
    ) -> list[str]:
        """Plus court chemin via Dijkstra.

        Parameters
        ----------
        src : nœud source
        dst : nœud destination
        weight : attribut d'arête à minimiser (défaut : distance_km)

        Returns
        -------
        Liste ordonnée des nœuds du chemin, ou liste vide si aucun chemin.
        """
        try:
            return nx.shortest_path(self._g, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def shortest_distance(
        self, src: str, dst: str, weight: str = "distance_km",
    ) -> float:
        """Distance du plus court chemin via Dijkstra.

        Returns
        -------
        Distance totale, ou ``float('inf')`` si aucun chemin.
        """
        try:
            return nx.shortest_path_length(self._g, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            return float("inf")

    def mean_travel_time_minutes(self, src: str, dst: str, loaded: bool) -> float:
        """Temps de trajet moyen (min), sans aléa.

        Vitesse de base 26 km/h chargé, 32 km/h à vide, réduite par la pente
        et l'état de la piste, avec un plancher de 5 km/h. Sert de moyenne au
        tirage aléatoire et d'estimation de coût pour ShortestPathPolicy.
        """
        edge = self.get_edge(src, dst)
        base_speed_kmh = 26.0 if loaded else 32.0

        slope_factor = max(0.55, 1.0 - max(0.0, edge.slope_pct) * SLOPE_PENALTY_PER_PCT)
        state_factor = max(0.55, min(1.2, edge.road_state))
        mean_speed_kmh = max(5.0, base_speed_kmh * slope_factor * state_factor)
        return (edge.distance_km / mean_speed_kmh) * 60.0

    def mean_route_time_minutes(self, src: str, dst: str, loaded: bool) -> float:
        """Temps moyen (min) d'un trajet complet, le long du plus court chemin en distance,
        celui que suit le simulateur (``MineEnv._find_route``). 0 si ``src == dst``."""
        path = self.shortest_path(src, dst)
        return sum(
            self.mean_travel_time_minutes(a, b, loaded) for a, b in zip(path, path[1:])
        )

    def sample_travel_time_minutes(
        self, src: str, dst: str, loaded: bool, rng: Random,
    ) -> float:
        """Tire un temps de trajet (min) selon une loi log-normale.

        mu est choisi pour que l'espérance soit égale au temps moyen ;
        sigma = 0,12. Minimum 0,2 min.
        """
        mean_time_min = self.mean_travel_time_minutes(src, dst, loaded)
        sigma = 0.12
        mu = log(max(0.2, mean_time_min)) - 0.5 * sigma * sigma
        return max(0.2, rng.lognormvariate(mu, sigma))


def build_mine_graph(
    shovel_count: int = 3,
    dump_count: int = 2,
) -> RoadGraph:
    """Construit le graphe de la mine.

    Topologie en étoile : parc ↔ intersection ↔ pelles et verses, plus des
    arêtes directes pelle → verse (chargé) et verse → pelle (à vide).
    Distances et pentes inspirées d'Afrapoli (2019) et Mohtasham (2023).
    """
    graph = RoadGraph()

    graph.add_node("yard", "yard")
    graph.add_node("junction_1", "junction")

    for i in range(1, shovel_count + 1):
        graph.add_node(f"shovel_{i}", "shovel")

    for i in range(1, dump_count + 1):
        graph.add_node(f"dump_{i}", "dump")

    graph.add_edge("yard", "junction_1", distance_km=0.8, slope_pct=2.0)
    graph.add_edge("junction_1", "yard", distance_km=0.8, slope_pct=1.0)

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
        # Pente de retour réduite (descente partielle), plancher à 1 %.
        graph.add_edge(node, "junction_1", distance_km=dist, slope_pct=max(1.0, slope - 2.0))

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
        # Pente de retour réduite (descente partielle), plancher à 1 %.
        graph.add_edge(node, "junction_1", distance_km=dist, slope_pct=max(1.0, slope - 3.0))

    # Pelle → verse (chargé) : distance = 0,8 × (somme des deux accès).
    for si in range(1, shovel_count + 1):
        for di in range(1, dump_count + 1):
            s_node = f"shovel_{si}"
            d_node = f"dump_{di}"
            s_idx = (si - 1) % len(shovel_params)
            d_idx = (di - 1) % len(dump_params)
            dist = (shovel_params[s_idx][0] + dump_params[d_idx][0]) * 0.8
            slope = (shovel_params[s_idx][1] + dump_params[d_idx][1]) / 2.0
            graph.add_edge(s_node, d_node, distance_km=dist, slope_pct=slope)

    # Verse → pelle (à vide).
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
