"""Environnement Gymnasium pour le dispatching minier.

Implémente l'interface standardisée Gymnasium (reset / step / render)
conformément au Chapitre 4 du mémoire (Section 4.3).

L'environnement orchestre la boucle Agent–Environnement :
  observation (s_t) → action (a_t) → récompense (r_t) → observation (s_{t+1})

Architecture conforme au Tableau 4.2 du mémoire :
  Truck   → x_c (localisation, statut)
  Shovel  → q_p (file d'attente), z_r
  DumpSite → z_r (disponibilité)
  RoadGraph → arcs (u,v), distances, pentes
  Environment → f(s_t, a_t) transition (Eq. 3.5–3.6)
"""

from __future__ import annotations

from math import log
from random import Random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from memoire_master_rl_logistique.simulation.entities import DumpSite, Shovel, Truck
from memoire_master_rl_logistique.simulation.fuel_model import (
    estimate_idle_fuel_l,
    estimate_travel_fuel_l,
)
from memoire_master_rl_logistique.simulation.graph_model import (
    RoadGraph,
    build_mine_graph,
)

from .observation_builder import build_observation, observation_size

STATUS_IDLE = 0
STATUS_TO_SHOVEL = 1
STATUS_LOADING = 2
STATUS_TO_DUMP = 3
STATUS_UNLOADING = 4
STATUS_RETURNING = 5


class MineEnv(gym.Env):
    """Environnement de simulation minière compatible Gymnasium.

    action_space : Discrete(num_shovels * num_dumps + 1)
        - 0..(S*D-1) : paire (pelle, dump), décodée par
          shovel_idx = action // dump_count,
          dump_idx   = action %  dump_count
        - S*D : ATTENDRE

    observation_space : Box([0,1]) de dimension observation_size(...)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        truck_count: int = 12,
        shovel_count: int = 3,
        dump_count: int = 2,
        episode_minutes: float = 480.0,
        seed: int = 42,
        breakdown_probability: float = 0.02,
        reward_weights: tuple[float, float, float] = (1.0, 0.1, 0.05),
        capacity_tonnes: float = 140.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.truck_count = truck_count
        self.shovel_count = shovel_count
        self.dump_count = dump_count
        self.episode_minutes = episode_minutes
        self._seed = seed
        self.breakdown_probability = breakdown_probability
        self.w1, self.w2, self.w3 = reward_weights
        self.capacity_tonnes = capacity_tonnes
        self.render_mode = render_mode

        self.num_actions = self.shovel_count * self.dump_count + 1
        self.action_space = spaces.Discrete(self.num_actions)

        obs_dim = observation_size(truck_count, self.shovel_count, dump_count)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        self.graph: RoadGraph | None = None
        self.trucks: list[Truck] = []
        self.shovels: list[Shovel] = []
        self.dumps: list[DumpSite] = []
        self.rng: Random | None = None
        self.current_time_min: float = 0.0
        self.current_truck_idx: int = 0
        self.truck_locations: list[str] = []
        self.truck_statuses: list[int] = []
        self._prev_tonnage: float = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Initialise la mine pour un nouvel épisode (Section 4.3.1)."""
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self.rng = Random(effective_seed)

        self.graph = build_mine_graph(
            shovel_count=self.shovel_count,
            dump_count=self.dump_count,
        )

        self.shovels = [
            Shovel(
                shovel_id=f"S{i + 1}",
                node_id=f"shovel_{i + 1}",
                load_time_mean_min=2.0,
                load_time_std_min=0.3,
            )
            for i in range(self.shovel_count)
        ]

        self.dumps = [
            DumpSite(
                dump_id=f"D{i + 1}",
                node_id=f"dump_{i + 1}",
                unload_time_mean_min=1.0,
                unload_time_std_min=0.2,
            )
            for i in range(self.dump_count)
        ]

        self.trucks = [
            Truck(
                truck_id=f"T{i + 1}",
                capacity_tonnes=self.capacity_tonnes,
            )
            for i in range(self.truck_count)
        ]

        self.current_time_min = 0.0
        self.current_truck_idx = 0
        self.truck_locations = ["yard"] * self.truck_count
        self.truck_statuses = [STATUS_IDLE] * self.truck_count
        self._prev_tonnage = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Exécute un cycle complet pour le camion courant.

        L'action encode une paire (pelle, dump) :
          shovel_idx = action // dump_count
          dump_idx   = action %  dump_count
        Si action == shovel_count * dump_count → ATTENDRE.
        """
        assert self.graph is not None and self.rng is not None

        truck = self.trucks[self.current_truck_idx]
        shovel, dump = self._decode_action(action)

        time_min = max(truck.available_at_min, self.current_time_min)
        location = self.truck_locations[self.current_truck_idx]

        # Trajet vers la pelle
        self.truck_statuses[self.current_truck_idx] = STATUS_TO_SHOVEL
        if location != shovel.node_id:
            t, f = self._travel_route(location, shovel.node_id, False)
            time_min += t
            truck.total_active_min += t
            truck.total_fuel_l += f
            location = shovel.node_id

        # Attente pelle
        self.truck_statuses[self.current_truck_idx] = STATUS_LOADING
        wait_load_min = max(0.0, shovel.available_at_min - time_min)
        if wait_load_min > 0:
            time_min += wait_load_min
            truck.total_wait_min += wait_load_min
            truck.total_fuel_l += estimate_idle_fuel_l(wait_load_min)

        # Chargement
        load_min = self._sample_duration(
            shovel.load_time_mean_min, shovel.load_time_std_min,
        )
        time_min += load_min
        truck.total_active_min += load_min
        truck.total_fuel_l += estimate_idle_fuel_l(load_min)
        shovel.available_at_min = time_min

        # Trajet chargé vers dump
        self.truck_statuses[self.current_truck_idx] = STATUS_TO_DUMP
        t, f = self._travel_route(shovel.node_id, dump.node_id, True)
        time_min += t
        truck.total_active_min += t
        truck.total_fuel_l += f
        location = dump.node_id

        # Attente dump
        self.truck_statuses[self.current_truck_idx] = STATUS_UNLOADING
        wait_dump_min = max(0.0, dump.available_at_min - time_min)
        if wait_dump_min > 0:
            time_min += wait_dump_min
            truck.total_wait_min += wait_dump_min
            truck.total_fuel_l += estimate_idle_fuel_l(wait_dump_min)

        # Déchargement
        unload_min = self._sample_duration(
            dump.unload_time_mean_min, dump.unload_time_std_min,
        )
        time_min += unload_min
        truck.total_active_min += unload_min
        truck.total_fuel_l += estimate_idle_fuel_l(unload_min)
        dump.available_at_min = time_min

        # Retour à vide
        self.truck_statuses[self.current_truck_idx] = STATUS_RETURNING
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

        # Compléter le cycle
        if time_min < self.episode_minutes:
            truck.cycles_completed += 1
            truck.total_tonnage_t += truck.capacity_tonnes

        truck.available_at_min = time_min
        self.truck_locations[self.current_truck_idx] = location
        self.truck_statuses[self.current_truck_idx] = STATUS_IDLE

        # Récompense (Eq. 3.8)
        reward = self._compute_reward(action)

        # Avancer le temps global et passer au camion suivant
        self.current_time_min = min(
            t.available_at_min for t in self.trucks
        )
        self.current_truck_idx = int(
            np.argmin([t.available_at_min for t in self.trucks]),
        )

        terminated = False
        truncated = self.current_time_min >= self.episode_minutes

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: int) -> float:
        """Récompense pondérée multi-objectif (Eq. 3.8).

        R_t = w1 * R_rendement + w2 * R_équité + w3 * R_coût
        """
        current_tonnage = sum(t.total_tonnage_t for t in self.trucks)
        r_rendement = current_tonnage - self._prev_tonnage
        self._prev_tonnage = current_tonnage

        if len(self.shovels) >= 2:
            wait_times = [
                max(0.0, s.available_at_min - self.current_time_min)
                for s in self.shovels
            ]
            mean_wait = sum(wait_times) / len(wait_times)
            variance = sum(
                (w - mean_wait) ** 2 for w in wait_times
            ) / len(wait_times)
            r_equite = -variance
        else:
            r_equite = 0.0

        r_cout = 0.0
        n_pair = self.shovel_count * self.dump_count
        if action < n_pair and self.graph is not None:
            s_idx = action // self.dump_count
            shovel = self.shovels[s_idx]
            truck_loc = self.truck_locations[self.current_truck_idx]
            route = self._find_route(truck_loc, shovel.node_id)
            for i in range(len(route) - 1):
                edge = self.graph.get_edge(route[i], route[i + 1])
                r_cout -= edge.distance_km

        r_rendement_norm = r_rendement / self.capacity_tonnes
        r_equite_norm = r_equite / 100.0
        r_cout_norm = r_cout / 5.0

        return (
            self.w1 * r_rendement_norm
            + self.w2 * r_equite_norm
            + self.w3 * r_cout_norm
        )

    def _decode_action(self, action: int) -> tuple[Shovel, DumpSite]:
        """Décode l'action en paire (pelle, dump).

        Actions 0..(S*D-1) : shovel_idx = action // D, dump_idx = action % D
        Action S*D          : ATTENDRE (pelle la plus tôt, dump le plus tôt)
        """
        n_pair = self.shovel_count * self.dump_count
        if action >= n_pair:
            shovel = min(
                self.shovels,
                key=lambda s: max(s.available_at_min, self.current_time_min),
            )
            dump = min(
                self.dumps,
                key=lambda d: max(d.available_at_min, self.current_time_min),
            )
        else:
            s_idx = action // self.dump_count
            d_idx = action % self.dump_count
            shovel = self.shovels[s_idx]
            dump = self.dumps[d_idx]
        return shovel, dump

    def _find_route(self, src: str, dst: str) -> list[str]:
        """Plus court chemin via Dijkstra (Section 3.2 du mémoire)."""
        assert self.graph is not None
        path = self.graph.shortest_path(src, dst)
        return path if path else [src]

    def _travel_route(
        self, src: str, dst: str, loaded: bool,
    ) -> tuple[float, float]:
        """Trajet multi-sauts via le graphe → (durée, carburant)."""
        assert self.graph is not None and self.rng is not None
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

    def _travel(
        self, src: str, dst: str, loaded: bool,
    ) -> tuple[float, float]:
        """Trajet direct avec stochasticité (Eq. 3.2)."""
        assert self.graph is not None and self.rng is not None
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
        if self.rng.random() < 0.05:
            edge.road_state = max(0.6, edge.road_state - 0.02)
        return travel_min, travel_fuel_l

    def _sample_duration(self, mean_min: float, std_min: float) -> float:
        """Durée log-normale (Eq. 3.2)."""
        assert self.rng is not None
        sigma = std_min / max(mean_min, 0.1)
        mu = log(max(0.2, mean_min)) - 0.5 * sigma * sigma
        return max(0.2, self.rng.lognormvariate(mu, sigma))

    def _get_obs(self) -> np.ndarray:
        """Construit l'observation courante."""
        return build_observation(
            trucks=self.trucks,
            shovels=self.shovels,
            dumps=self.dumps,
            graph=self.graph,
            current_time_min=self.current_time_min,
            episode_minutes=self.episode_minutes,
            truck_locations=self.truck_locations,
            truck_statuses=self.truck_statuses,
        )

    def _get_info(self) -> dict[str, Any]:
        """Dictionnaire d'informations de debug."""
        return {
            "current_time_min": self.current_time_min,
            "current_truck_idx": self.current_truck_idx,
            "total_tonnage_t": sum(t.total_tonnage_t for t in self.trucks),
            "total_fuel_l": sum(t.total_fuel_l for t in self.trucks),
            "total_wait_min": sum(t.total_wait_min for t in self.trucks),
            "total_cycles": sum(t.cycles_completed for t in self.trucks),
        }

    def render(self) -> str | None:
        """Affichage texte de l'état courant."""
        if self.render_mode in ("ansi", "human"):
            lines = [
                f"=== Mine t={self.current_time_min:.1f} min ===",
                f"Camion actif: T{self.current_truck_idx + 1}",
            ]
            for t in self.trucks:
                lines.append(
                    f"  {t.truck_id}: cycles={t.cycles_completed}, "
                    f"tonnage={t.total_tonnage_t:.0f}t, "
                    f"wait={t.total_wait_min:.1f}min, "
                    f"fuel={t.total_fuel_l:.1f}L"
                )
            output = "\n".join(lines)
            if self.render_mode == "human":
                print(output)  # noqa: T201
            return output
        return None
