"""Simulation à événements discrets (DES) du transport minier.

Simule tous les camions en parallèle via un tas d'événements trié par
temps, garantissant une concurrence réaliste aux pelles et aux dumps
(Section 4.3 du mémoire — Conception de l'environnement de simulation).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import log
from random import Random

from .entities import DumpSite, Shovel, Truck
from .fuel_model import estimate_idle_fuel_l, estimate_travel_fuel_l
from .graph_model import RoadGraph, build_mine_graph
from .kpi import compute_kpis


@dataclass(slots=True)
class EpisodeResult:
    """KPIs globaux et détails par camion d'un épisode."""

    kpis: dict[str, float]
    per_truck: list[dict[str, float | str]]


class MineSimulation:
    """Simulation à événements discrets du transport minier.

    Tous les camions opèrent en parallèle : à chaque instant, le
    camion dont l'événement est le plus proche dans le temps est
    traité en premier (file de priorité / tas min).
    """

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
        """Échantillonne une durée log-normale (Eq. 3.2)."""
        sigma = std_min / max(mean_min, 0.1)
        mu = log(max(0.2, mean_min)) - 0.5 * sigma * sigma
        return max(0.2, self.rng.lognormvariate(mu, sigma))

    def _select_shovel(self, time_min: float) -> Shovel:
        """Pelle disponible la plus tôt."""
        return min(
            self.shovels,
            key=lambda s: max(s.available_at_min, time_min),
        )

    def _update_road_state(self, src: str, dst: str) -> None:
        """Dégradation stochastique de la route (5 %)."""
        if self.rng.random() < 0.05:
            edge = self.graph.get_edge(src, dst)
            edge.road_state = max(0.6, edge.road_state - 0.02)

    def _travel(
        self, src: str, dst: str, loaded: bool,
    ) -> tuple[float, float]:
        """Trajet entre deux nœuds → (durée_min, carburant_l)."""
        travel_min = self.graph.sample_travel_time_minutes(
            src=src, dst=dst, loaded=loaded, rng=self.rng,
        )
        edge = self.graph.get_edge(src, dst)
        travel_fuel_l = estimate_travel_fuel_l(
            distance_km=edge.distance_km,
            slope_pct=edge.slope_pct,
            road_state=edge.road_state,
            loaded=loaded,
        )
        self._update_road_state(src, dst)
        return travel_min, travel_fuel_l

    def _find_route(self, src: str, dst: str) -> list[str]:
        """Trouve un chemin dans le graphe via BFS."""
        if (src, dst) in self.graph.edges:
            return [src, dst]
        visited: set[str] = {src}
        queue: list[tuple[str, list[str]]] = [(src, [src])]
        while queue:
            current, path = queue.pop(0)
            for (s, d) in self.graph.edges:
                if s == current and d not in visited:
                    new_path = [*path, d]
                    if d == dst:
                        return new_path
                    visited.add(d)
                    queue.append((d, new_path))
        return []

    def _travel_route(
        self, src: str, dst: str, loaded: bool,
    ) -> tuple[float, float]:
        """Trajet multi-sauts via le graphe → (durée, carburant)."""
        route = self._find_route(src, dst)
        if len(route) < 2:
            return 0.0, 0.0
        total_time = 0.0
        total_fuel = 0.0
        for i in range(len(route) - 1):
            t, f = self._travel(route[i], route[i + 1], loaded)
            total_time += t
            total_fuel += f
        return total_time, total_fuel

    def run_episode(
        self, episode_minutes: float = 8.0 * 60.0,
    ) -> EpisodeResult:
        """Simule un épisode complet en parallèle.

        Chaque camion est un événement dans un tas min trié par temps.
        Le camion dont le prochain événement est le plus proche est
        traité en premier, garantissant la concurrence réaliste.
        """
        dump = self.dumps[0]

        heap: list[tuple[float, int, str]] = []
        for i, truck in enumerate(self.trucks):
            heapq.heappush(heap, (truck.available_at_min, i, "yard"))

        while heap:
            time_min, truck_idx, location = heapq.heappop(heap)

            if time_min >= episode_minutes:
                continue

            truck = self.trucks[truck_idx]
            shovel = self._select_shovel(time_min)

            # Trajet vers la pelle
            if location != shovel.node_id:
                t, f = self._travel_route(location, shovel.node_id, False)
                time_min += t
                truck.total_active_min += t
                truck.total_fuel_l += f
                location = shovel.node_id

            if time_min >= episode_minutes:
                continue

            # Attente pelle
            wait_load = max(0.0, shovel.available_at_min - time_min)
            if wait_load > 0:
                time_min += wait_load
                truck.total_wait_min += wait_load
                truck.total_fuel_l += estimate_idle_fuel_l(wait_load)

            # Chargement
            load_min = self._sample_duration(
                shovel.load_time_mean_min, shovel.load_time_std_min,
            )
            time_min += load_min
            truck.total_active_min += load_min
            truck.total_fuel_l += estimate_idle_fuel_l(load_min)
            shovel.available_at_min = time_min

            if time_min >= episode_minutes:
                continue

            # Trajet chargé vers le dump
            t, f = self._travel_route(shovel.node_id, dump.node_id, True)
            time_min += t
            truck.total_active_min += t
            truck.total_fuel_l += f
            location = dump.node_id

            if time_min >= episode_minutes:
                continue

            # Attente dump
            wait_dump = max(0.0, dump.available_at_min - time_min)
            if wait_dump > 0:
                time_min += wait_dump
                truck.total_wait_min += wait_dump
                truck.total_fuel_l += estimate_idle_fuel_l(wait_dump)

            # Déchargement
            unload_min = self._sample_duration(
                dump.unload_time_mean_min, dump.unload_time_std_min,
            )
            time_min += unload_min
            truck.total_active_min += unload_min
            truck.total_fuel_l += estimate_idle_fuel_l(unload_min)
            dump.available_at_min = time_min

            if time_min >= episode_minutes:
                continue

            # Retour à vide
            t, f = self._travel_route(dump.node_id, shovel.node_id, False)
            time_min += t
            truck.total_active_min += t
            truck.total_fuel_l += f
            location = shovel.node_id

            # Panne éventuelle
            if self.rng.random() < self.breakdown_probability:
                repair_min = self.rng.uniform(10.0, 30.0)
                time_min += repair_min
                truck.total_wait_min += repair_min
                truck.total_fuel_l += estimate_idle_fuel_l(repair_min)

            # Cycle complet seulement si encore dans l'épisode
            if time_min < episode_minutes:
                truck.cycles_completed += 1
                truck.total_tonnage_t += truck.capacity_tonnes
                truck.history.append({
                    "cycle": truck.cycles_completed,
                    "time_min": round(time_min, 2),
                    "tonnage_t": truck.total_tonnage_t,
                })

            truck.available_at_min = time_min
            heapq.heappush(heap, (time_min, truck_idx, location))

        # Agrégation des KPIs
        total_tonnage = sum(t.total_tonnage_t for t in self.trucks)
        total_wait = sum(t.total_wait_min for t in self.trucks)
        total_active = sum(t.total_active_min for t in self.trucks)
        total_fuel = sum(t.total_fuel_l for t in self.trucks)

        kpis = compute_kpis(
            episode_minutes=episode_minutes,
            truck_count=len(self.trucks),
            total_tonnage_t=total_tonnage,
            total_wait_min=total_wait,
            total_active_min=total_active,
            total_fuel_l=total_fuel,
        )

        per_truck = [
            {
                "truck_id": t.truck_id,
                "cycles_completed": float(t.cycles_completed),
                "total_tonnage_t": float(t.total_tonnage_t),
                "total_wait_min": float(t.total_wait_min),
                "total_active_min": float(t.total_active_min),
                "total_fuel_l": float(t.total_fuel_l),
            }
            for t in self.trucks
        ]

        return EpisodeResult(kpis=kpis, per_truck=per_truck)


def build_simulation(
    seed: int = 42,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    capacity_tonnes: float = 140.0,
    breakdown_probability: float = 0.02,
    load_time_mean_min: float = 2.0,
    load_time_std_min: float = 0.3,
    unload_time_mean_min: float = 1.0,
    unload_time_std_min: float = 0.2,
) -> MineSimulation:
    """Construit une simulation paramétrable.

    Valeurs par défaut basées sur la littérature :
    - 12 camions Cat 785C (Kangwa 2021)
    - 3 pelles Hitachi 2500 (Mohtasham 2023)
    - 2 points de déchargement
    - Chargement NORM(2, 0.3) min (Mohtasham 2023, Table 2)
    - Déchargement NORM(1, 0.2) min
    - Pannes 2% par shift
    """
    graph = build_mine_graph(
        shovel_count=shovel_count,
        dump_count=dump_count,
    )

    shovels = [
        Shovel(
            shovel_id=f"S{i + 1}",
            node_id=f"shovel_{i + 1}",
            load_time_mean_min=load_time_mean_min,
            load_time_std_min=load_time_std_min,
        )
        for i in range(shovel_count)
    ]

    dumps = [
        DumpSite(
            dump_id=f"D{i + 1}",
            node_id=f"dump_{i + 1}",
            unload_time_mean_min=unload_time_mean_min,
            unload_time_std_min=unload_time_std_min,
        )
        for i in range(dump_count)
    ]

    trucks = [
        Truck(truck_id=f"T{i + 1}", capacity_tonnes=capacity_tonnes)
        for i in range(truck_count)
    ]

    return MineSimulation(
        graph=graph,
        trucks=trucks,
        shovels=shovels,
        dumps=dumps,
        seed=seed,
        breakdown_probability=breakdown_probability,
    )
