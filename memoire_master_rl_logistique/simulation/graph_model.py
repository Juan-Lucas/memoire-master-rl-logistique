from dataclasses import dataclass
from math import log
from random import Random


@dataclass(slots=True)
class RoadNode:
    """Représente un nœud du graphe routier, avec un type (yard, shovel, dump)."""
    node_id: str
    node_type: str


@dataclass(slots=True)
class RoadEdge:
    """Représente une arête orientée entre deux nœuds du graphe routier, avec les caractéristiques de la route."""
    src: str
    dst: str
    distance_km: float
    slope_pct: float
    road_state: float


class RoadGraph:
    """Représentation en graphe orienté du réseau routier minier."""

    def __init__(self) -> None:
        self.nodes: dict[str, RoadNode] = {}
        self.edges: dict[tuple[str, str], RoadEdge] = {}

    def add_node(self, node_id: str, node_type: str) -> None:
        """Ajoute un nœud au graphe, avec un type (yard, shovel, dump). Lève une exception si le nœud existe déjà."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists.")
        self.nodes[node_id] = RoadNode(node_id=node_id, node_type=node_type)

    def add_edge(
        self,
        src: str,
        dst: str,
        distance_km: float,
        slope_pct: float,
        road_state: float,
    ) -> None:
        """Ajoute une arête orientée entre deux nœuds, avec les caractéristiques de la route. Lève une exception si les nœuds n'existent pas."""
        if src not in self.nodes:
            raise ValueError(f"Source node {src} does not exist.")
        if dst not in self.nodes:
            raise ValueError(f"Destination node {dst} does not exist.")
        self.edges[(src, dst)] = RoadEdge(
            src=src,
            dst=dst,
            distance_km=distance_km,
            slope_pct=slope_pct,
            road_state=road_state,
        )

    def get_edge(self, src: str, dst: str) -> RoadEdge:
        """Récupère l'arête entre deux nœuds, ou lève une exception si elle n'existe pas."""
        if (src, dst) not in self.edges:
            raise ValueError(f"Edge from {src} to {dst} does not exist.")
        return self.edges[(src, dst)]

    def sample_travel_time_minutes(self, src: str, dst: str, loaded: bool, rng: Random) -> float:
        """Échantillonne un temps de trajet en minutes entre deux nœuds, en fonction de la distance, de la pente, de l'état de la route et de la charge du camion."""
        edge = self.get_edge(src, dst)
        base_speed_kmh = 26.0 if loaded else 32.0

        # Une pente positive réduit la vitesse, une route dégradée aussi.
        slope_factor = max(0.55, 1.0 - max(0.0, edge.slope_pct) * 0.03)
        state_factor = max(0.55, min(1.2, edge.road_state))
        mean_speed_kmh = max(5.0, base_speed_kmh * slope_factor * state_factor)
        mean_time_min = (edge.distance_km / mean_speed_kmh) * 60.0

        sigma = 0.12
        mu = log(max(0.2, mean_time_min)) - 0.5 * sigma * sigma
        return max(0.2, rng.lognormvariate(mu, sigma))


def build_default_graph() -> RoadGraph:
    """Petit graphe MVP Sprint A: 2 pelles, 1 déchargement, 1 aire d'attente."""

    graph = RoadGraph()
    graph.add_node("yard", "yard")
    graph.add_node("shovel_1", "shovel")
    graph.add_node("shovel_2", "shovel")
    graph.add_node("dump_1", "dump")

    # Trajets vers les pelles
    graph.add_edge("yard", "shovel_1", distance_km=1.5, slope_pct=4.0, road_state=1.0)
    graph.add_edge("yard", "shovel_2", distance_km=1.8, slope_pct=5.0, road_state=1.0)

    # Pelle vers point de déchargement (camion chargé)
    graph.add_edge("shovel_1", "dump_1", distance_km=2.8, slope_pct=6.0, road_state=0.98)
    graph.add_edge("shovel_2", "dump_1", distance_km=3.2, slope_pct=8.0, road_state=0.96)

    # Retour à vide
    graph.add_edge("dump_1", "shovel_1", distance_km=2.8, slope_pct=1.0, road_state=0.98)
    graph.add_edge("dump_1", "shovel_2", distance_km=3.2, slope_pct=1.0, road_state=0.96)

    return graph
