"""MTWT sensible au carburant (M5 du rapport du 17/11/2026).

Référence orientée énergie, pour vérifier si l'avantage de PPO en carburant par tonne
subsiste face à une règle qui tient compte des files **et** du carburant. Pour chaque pelle
j, le score est l'attente prévue à la pelle plus λ fois le carburant prévu du cycle :

    score_j = attente_j (min) + λ · (carburant du trajet à vide vers j
                                     + carburant du trajet chargé de j vers la verse) (L)

La verse est celle que MTWT choisirait depuis j (plus petite attente prévue). λ, en
minutes par litre, règle le compromis : λ = 0 redonne exactement MTWT ; un λ grand envoie
les camions vers les cycles les plus sobres, quelle que soit l'attente.
"""

from __future__ import annotations

from memoire_master_rl_logistique.baselines.mtwt_policy import DEFAULT_LOAD_TIME_MIN, MTWTPolicy
from memoire_master_rl_logistique.simulation.forecast import forecast_route
from memoire_master_rl_logistique.simulation.fuel_model import (
    DEFAULT_FUEL,
    FuelParams,
    estimate_travel_fuel_l,
)
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


class FuelAwareMTWTPolicy(MTWTPolicy):
    """Pelle de plus petit « attente prévue + λ · carburant prévu », puis verse comme MTWT."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: list[str],
        dump_node_ids: list[str],
        fuel_weight_min_per_l: float,
        load_time_min: float = DEFAULT_LOAD_TIME_MIN,
        fuel: FuelParams = DEFAULT_FUEL,
    ) -> None:
        if fuel_weight_min_per_l < 0:
            raise ValueError(f"λ doit être ≥ 0 : {fuel_weight_min_per_l}")
        super().__init__(graph, shovel_node_ids, dump_node_ids, load_time_min)
        self.fuel_weight = fuel_weight_min_per_l
        self.fuel = fuel

    def route_fuel_l(self, src: str, dst: str, loaded: bool) -> float:
        """Carburant prévu (L) sur le plus court chemin, état actuel des pistes."""
        path = self.graph.shortest_path(src, dst)
        total = 0.0
        for a, b in zip(path, path[1:]):
            edge = self.graph.get_edge(a, b)
            total += estimate_travel_fuel_l(edge.distance_km, edge.slope_pct, edge.road_state,
                                            loaded, self.fuel)
        return total

    def choose(self, truck_location, now, shovel_available_at, dump_available_at):  # noqa: ANN001
        routes = {
            (j, k): forecast_route(self.graph, truck_location, now, s_node, d_node,
                                   shovel_available_at[j], dump_available_at[k],
                                   self.load_time_min)
            for j, s_node in enumerate(self.shovel_node_ids)
            for k, d_node in enumerate(self.dump_node_ids)
        }

        def dump_for(j: int) -> int:
            return min(range(len(self.dump_node_ids)),
                       key=lambda k: (routes[j, k].dump_wait, routes[j, k].dump_arrival))

        def score(j: int) -> tuple[float, float]:
            wait = routes[j, 0].shovel_wait
            if self.fuel_weight == 0:
                return wait, routes[j, 0].shovel_arrival  # exactement MTWT
            s_node, d_node = self.shovel_node_ids[j], self.dump_node_ids[dump_for(j)]
            fuel = (self.route_fuel_l(truck_location, s_node, loaded=False)
                    + self.route_fuel_l(s_node, d_node, loaded=True))
            return wait + self.fuel_weight * fuel, routes[j, 0].shovel_arrival

        shovel = min(range(len(self.shovel_node_ids)), key=score)
        return shovel, dump_for(shovel)
