from dataclasses import dataclass
from random import Random

from .entities import DumpSite, Shovel, Truck
from .fuel_model import estimate_idle_fuel_l, estimate_travel_fuel_l
from .graph_model import RoadGraph, build_default_graph
from .kpi import compute_kpis


@dataclass(slots=True)
class EpisodeResult:
    """Résultat d'une simulation d'épisode, avec des KPIs globaux et des détails par camion."""
    kpis: dict[str, float]
    per_truck: list[dict[str, float | str]]


class MineSimulation:
    """Simulation minimale de bout en bout pour le MVP du Sprint A."""

    def __init__(
        self,
        graph: RoadGraph,
        trucks: list[Truck],
        shovels: list[Shovel],
        dumps: list[DumpSite],
        seed: int = 42,
        breakdown_probability: float = 0.02,
    ) -> None:
        self.graph = graph
        self.trucks = trucks
        self.shovels = shovels
        self.dumps = dumps
        self.rng = Random(seed)
        self.breakdown_probability = breakdown_probability

    def _sample_duration(self, mean_min: float, std_min: float) -> float:
        """Échantillonne une durée à partir d'une distribution log-normale pour éviter les valeurs négatives."""
        return max(0.2, self.rng.gauss(mean_min, std_min))

    def _select_shovel(self, time_min: float) -> Shovel:
        """Sélectionne la pelle disponible la plus tôt à partir du temps actuel."""
        return min(self.shovels, key=lambda shovel: max(shovel.available_at_min, time_min))

    def _update_road_state(self, src: str, dst: str) -> None:
        """Simule une légère dégradation de l'état de la route après chaque traversée, avec une faible probabilité."""
        # Faible probabilité de dégradation après chaque traversée.
        if self.rng.random() < 0.05:
            edge = self.graph.get_edge(src, dst)
            edge.road_state = max(0.6, edge.road_state - 0.02)

    def _travel(self, src: str, dst: str, loaded: bool) -> tuple[float, float]:
        """Simule un déplacement entre deux nœuds, en échantillonnant le temps de trajet et en estimant la consommation de carburant."""
        travel_min = self.graph.sample_travel_time_minutes(src=src, dst=dst, loaded=loaded, rng=self.rng)
        edge = self.graph.get_edge(src, dst)
        travel_fuel_l = estimate_travel_fuel_l(
            distance_km=edge.distance_km,
            slope_pct=edge.slope_pct,
            road_state=edge.road_state,
            loaded=loaded,
        )
        self._update_road_state(src, dst)
        return travel_min, travel_fuel_l

    def run_episode(self, episode_minutes: float = 8.0 * 60.0) -> EpisodeResult:
        """Simule un épisode complet de fonctionnement des camions, en suivant une logique simple de chargement, déplacement, déchargement et retour, avec des événements de panne aléatoires."""
        dump = self.dumps[0]

        for truck in self.trucks:
            time_min = truck.available_at_min
            location = "yard"

            while time_min < episode_minutes:
                shovel = self._select_shovel(time_min)

                # Déplacement vers la pelle si le camion n'y est pas déjà.
                if location != shovel.node_id:
                    travel_min, travel_fuel_l = self._travel(location, shovel.node_id, loaded=False)
                    time_min += travel_min
                    truck.total_active_min += travel_min
                    truck.total_fuel_l += travel_fuel_l
                    location = shovel.node_id

                if time_min >= episode_minutes:
                    break

                # Attente de disponibilité de la pelle.
                wait_load_min = max(0.0, shovel.available_at_min - time_min)
                if wait_load_min > 0:
                    time_min += wait_load_min
                    truck.total_wait_min += wait_load_min
                    truck.total_fuel_l += estimate_idle_fuel_l(wait_load_min)

                # Chargement.
                load_min = self._sample_duration(shovel.load_time_mean_min, shovel.load_time_std_min)
                time_min += load_min
                truck.total_active_min += load_min
                truck.total_fuel_l += estimate_idle_fuel_l(load_min)
                shovel.available_at_min = time_min

                if time_min >= episode_minutes:
                    break

                # Trajet chargé vers le déchargement.
                loaded_travel_min, loaded_travel_fuel_l = self._travel(shovel.node_id, dump.node_id, loaded=True)
                time_min += loaded_travel_min
                truck.total_active_min += loaded_travel_min
                truck.total_fuel_l += loaded_travel_fuel_l
                location = dump.node_id

                if time_min >= episode_minutes:
                    break

                # Attente de disponibilité du point de déchargement.
                wait_dump_min = max(0.0, dump.available_at_min - time_min)
                if wait_dump_min > 0:
                    time_min += wait_dump_min
                    truck.total_wait_min += wait_dump_min
                    truck.total_fuel_l += estimate_idle_fuel_l(wait_dump_min)

                # Déchargement.
                unload_min = self._sample_duration(dump.unload_time_mean_min, dump.unload_time_std_min)
                time_min += unload_min
                truck.total_active_min += unload_min
                truck.total_fuel_l += estimate_idle_fuel_l(unload_min)
                dump.available_at_min = time_min

                if time_min >= episode_minutes:
                    break

                # Retour à vide vers la même pelle.
                return_travel_min, return_travel_fuel_l = self._travel(dump.node_id, shovel.node_id, loaded=False)
                time_min += return_travel_min
                truck.total_active_min += return_travel_min
                truck.total_fuel_l += return_travel_fuel_l
                location = shovel.node_id

                # Événement de panne simple en fin de cycle.
                if self.rng.random() < self.breakdown_probability:
                    repair_min = self.rng.uniform(10.0, 30.0)
                    time_min += repair_min
                    truck.total_wait_min += repair_min
                    truck.total_fuel_l += estimate_idle_fuel_l(repair_min)

                truck.cycles_completed += 1
                truck.total_tonnage_t += truck.capacity_tonnes

                truck.history.append(
                    {
                        "cycle": truck.cycles_completed,
                        "time_min": round(time_min, 2),
                        "tonnage_t": truck.total_tonnage_t,
                    }
                )

            truck.available_at_min = time_min

        total_tonnage_t = sum(truck.total_tonnage_t for truck in self.trucks)
        total_wait_min = sum(truck.total_wait_min for truck in self.trucks)
        total_active_min = sum(truck.total_active_min for truck in self.trucks)
        total_fuel_l = sum(truck.total_fuel_l for truck in self.trucks)

        kpis = compute_kpis(
            episode_minutes=episode_minutes,
            truck_count=len(self.trucks),
            total_tonnage_t=total_tonnage_t,
            total_wait_min=total_wait_min,
            total_active_min=total_active_min,
            total_fuel_l=total_fuel_l,
        )

        per_truck = [
            {
                "truck_id": truck.truck_id,
                "cycles_completed": float(truck.cycles_completed),
                "total_tonnage_t": float(truck.total_tonnage_t),
                "total_wait_min": float(truck.total_wait_min),
                "total_active_min": float(truck.total_active_min),
                "total_fuel_l": float(truck.total_fuel_l),
            }
            for truck in self.trucks
        ]

        return EpisodeResult(kpis=kpis, per_truck=per_truck)


def build_default_simulation(
    seed: int = 42,
    truck_count: int = 5,
    shovel_count: int = 2,
) -> MineSimulation:
    """Construit une simulation avec une configuration par défaut pour le MVP du Sprint A, en permettant de spécifier le nombre de camions et de pelles, ainsi qu'une graine pour la reproductibilité."""
    graph = build_default_graph()

    shovel_nodes = ["shovel_1", "shovel_2"]
    shovel_nodes = shovel_nodes[: max(1, min(shovel_count, len(shovel_nodes)))]

    shovels = [
        Shovel(
            shovel_id=f"S{i+1}",
            node_id=node_id,
            load_time_mean_min=2.0,
            load_time_std_min=0.3,
        )
        for i, node_id in enumerate(shovel_nodes)
    ]

    dumps = [
        DumpSite(
            dump_id="D1",
            node_id="dump_1",
            unload_time_mean_min=1.0,
            unload_time_std_min=0.2,
        )
    ]

    trucks = [
        Truck(truck_id=f"T{i+1}", capacity_tonnes=140.0)
        for i in range(max(1, truck_count))
    ]

    return MineSimulation(
        graph=graph,
        trucks=trucks,
        shovels=shovels,
        dumps=dumps,
        seed=seed,
        breakdown_probability=0.02,
    )
