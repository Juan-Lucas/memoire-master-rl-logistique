"""Environnement Gymnasium pour le dispatching camions–pelles.

Une décision est prise chaque fois qu'un camion devient disponible. ``step``
simule alors d'un seul tenant tout le cycle de ce camion (trajet, file,
chargement, trajet chargé, file, déversement, retour à vide, panne éventuelle)
en réservant les pelles et les verses (``available_at_min``), puis avance
l'horloge jusqu'au prochain camion disponible.

Aléas (T13) : la graine d'un épisode engendre des flux aléatoires indépendants
(``numpy.random.SeedSequence``), un par processus : trajets, chargements et
déversements, dégradation des pistes, et un flux de pannes **par camion**. Deux
politiques jouées avec la même graine partagent donc les mêmes aléas
(nombres aléatoires communs) ; en particulier, le k-ième cycle d'un camion donné a
la même panne, de même durée, quelle que soit la politique.
"""

from __future__ import annotations

from math import log
from random import Random
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import SeedSequence

from memoire_master_rl_logistique.simulation.entities import (
    CycleRecord,
    DumpSite,
    Shovel,
    Truck,
)
from memoire_master_rl_logistique.simulation.fuel_model import (
    DEFAULT_FUEL,
    FuelParams,
    estimate_idle_fuel_l,
    estimate_travel_fuel_l,
)
from memoire_master_rl_logistique.simulation.graph_model import (
    RoadGraph,
    build_mine_graph,
)
from memoire_master_rl_logistique.simulation.reward import (
    RewardTerms,
    RewardWeights,
    compute_reward,
    reward_terms,
)

from .observation_builder import build_observation, observation_size

STATUS_IDLE = 0
STATUS_TO_SHOVEL = 1
STATUS_LOADING = 2
STATUS_TO_DUMP = 3
STATUS_UNLOADING = 4
STATUS_RETURNING = 5

REWARD_TERM_NAMES = tuple(RewardTerms.__dataclass_fields__)

# Flux aléatoires communs à tous les camions ; viennent ensuite les flux de pannes, un par camion.
SHARED_STREAMS = ("travel", "service", "road")


def _stream(seed_sequence: SeedSequence) -> Random:
    """Générateur ``random.Random`` initialisé par un enfant d'une ``SeedSequence``."""
    return Random(int.from_bytes(seed_sequence.generate_state(4).tobytes(), "little"))


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
        reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
        capacity_tonnes: float = 140.0,
        wait_action_minutes: float = 5.0,
        render_mode: str | None = None,
        log_cycles: bool = False,
        fuel: FuelParams = DEFAULT_FUEL,
    ) -> None:
        super().__init__()

        self.truck_count = truck_count
        self.shovel_count = shovel_count
        self.dump_count = dump_count
        self.episode_minutes = episode_minutes
        self._seed = seed
        self.breakdown_probability = breakdown_probability
        self.reward_weights = RewardWeights.from_tuple(reward_weights)
        # w1 à w4 gardés comme attributs (lus par EnvConfig.from_env et par les tests).
        self.w1, self.w2, self.w3, self.w4 = (
            self.reward_weights.production, self.reward_weights.balance,
            self.reward_weights.distance, self.reward_weights.wait,
        )
        self.capacity_tonnes = capacity_tonnes
        self.wait_action_minutes = wait_action_minutes
        self.render_mode = render_mode
        # Journal des cycles (T12) : désactivé par défaut, sans effet sur la simulation.
        self.log_cycles = log_cycles
        # Modèle de carburant (T14) : calibré par défaut, modifiable pour une étude de sensibilité.
        self.fuel = fuel
        self.cycle_log: list[CycleRecord] = []
        # Décisions de l'épisode (T19), exposées dans ``info`` : toutes, et ATTENDRE seules.
        self.decision_count: int = 0
        self.wait_action_count: int = 0
        # Contributions pondérées des termes de la récompense (T25), remises à zéro au reset.
        self.last_reward_terms: dict[str, float] = dict.fromkeys(REWARD_TERM_NAMES, 0.0)
        self.episode_reward_terms: dict[str, float] = dict.fromkeys(REWARD_TERM_NAMES, 0.0)

        self.num_actions = self.shovel_count * self.dump_count + 1
        self.action_space = spaces.Discrete(self.num_actions)

        obs_dim = observation_size(self.shovel_count, self.dump_count)  # indépendant de N (T18)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        self.graph: RoadGraph | None = None
        self.trucks: list[Truck] = []
        self.shovels: list[Shovel] = []
        self.dumps: list[DumpSite] = []
        # Flux aléatoires indépendants (T13), créés à chaque reset à partir de la graine.
        self.rng_travel: Random | None = None
        self.rng_service: Random | None = None
        self.rng_road: Random | None = None
        self.rng_breakdowns: list[Random] = []
        self.current_time_min: float = 0.0
        self.current_truck_idx: int = 0
        self.truck_locations: list[str] = []
        self.truck_statuses: list[int] = []
        self._prev_tonnage: float = 0.0
        # Durée (ms) de la dernière construction d'observation : mesure seulement,
        # n'influence pas la simulation (voir run_single_episode).
        self.last_obs_time_ms: float = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Initialise un nouveau poste.

        Sans graine, un compteur interne est utilisé (graine du constructeur
        + 1, + 2, …) : c'est le cas pendant l'entraînement de PPO et DQN.
        """
        super().reset(seed=seed)
        if seed is None:
            self._episode_counter = getattr(self, "_episode_counter", self._seed) + 1
            effective_seed = self._episode_counter
        else:
            effective_seed = seed
        streams = SeedSequence(effective_seed).spawn(len(SHARED_STREAMS) + self.truck_count)
        self.rng_travel, self.rng_service, self.rng_road = (
            _stream(s) for s in streams[: len(SHARED_STREAMS)]
        )
        self.rng_breakdowns = [_stream(s) for s in streams[len(SHARED_STREAMS):]]

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
        self.cycle_log = []
        self.decision_count = 0
        self.wait_action_count = 0
        self.last_reward_terms = dict.fromkeys(REWARD_TERM_NAMES, 0.0)
        self.episode_reward_terms = dict.fromkeys(REWARD_TERM_NAMES, 0.0)

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
        assert self.graph is not None and self.rng_travel is not None
        action = int(action)
        if not self.action_space.contains(action):
            msg = f"Invalid action {action}; expected 0..{self.num_actions - 1}."
            raise ValueError(msg)

        truck = self.trucks[self.current_truck_idx]
        start_time = max(truck.available_at_min, self.current_time_min)
        time_min = start_time
        location = self.truck_locations[self.current_truck_idx]
        decision_time = self.current_time_min
        n_pair = self.shovel_count * self.dump_count
        self.decision_count += 1

        if action == n_pair:
            self.wait_action_count += 1
            reward = self._apply_wait_action(truck, time_min)
            self._advance_to_next_truck()
            truncated = self.current_time_min >= self.episode_minutes
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, False, truncated, info

        shovel, dump = self._decode_action(action)
        pre_action_route_km = (
            self._route_distance(location, shovel.node_id)
            + self._route_distance(shovel.node_id, dump.node_id)
            + self._route_distance(dump.node_id, shovel.node_id)
        )
        resource_waits = self._shovel_waits(decision_time)

        # Trajet vers la pelle
        self.truck_statuses[self.current_truck_idx] = STATUS_TO_SHOVEL
        travel_to_shovel_min = 0.0
        if location != shovel.node_id:
            t, f = self._travel_route(location, shovel.node_id, False)
            travel_to_shovel_min = t
            truck.total_active_min += self._in_shift(time_min, t)
            time_min += t
            truck.total_fuel_l += f
            location = shovel.node_id
            self.truck_locations[self.current_truck_idx] = location

        # Attente pelle
        self.truck_statuses[self.current_truck_idx] = STATUS_LOADING
        wait_load_min = max(0.0, shovel.available_at_min - time_min)
        if wait_load_min > 0:
            truck.queue_wait_min += self._in_shift(time_min, wait_load_min)
            time_min += wait_load_min
            truck.total_fuel_l += estimate_idle_fuel_l(wait_load_min, self.fuel)

        # Chargement
        load_min = self._sample_duration(
            shovel.load_time_mean_min, shovel.load_time_std_min,
        )
        truck.total_active_min += self._in_shift(time_min, load_min)
        shovel.busy_min += self._in_shift(time_min, load_min)
        time_min += load_min
        truck.total_fuel_l += estimate_idle_fuel_l(load_min, self.fuel)
        shovel.available_at_min = time_min

        # Trajet chargé vers dump
        self.truck_statuses[self.current_truck_idx] = STATUS_TO_DUMP
        t, f = self._travel_route(shovel.node_id, dump.node_id, True)
        travel_to_dump_min = t
        truck.total_active_min += self._in_shift(time_min, t)
        time_min += t
        truck.total_fuel_l += f
        location = dump.node_id
        self.truck_locations[self.current_truck_idx] = location

        # Attente dump
        self.truck_statuses[self.current_truck_idx] = STATUS_UNLOADING
        wait_dump_min = max(0.0, dump.available_at_min - time_min)
        if wait_dump_min > 0:
            truck.queue_wait_min += self._in_shift(time_min, wait_dump_min)
            time_min += wait_dump_min
            truck.total_fuel_l += estimate_idle_fuel_l(wait_dump_min, self.fuel)

        # Déchargement
        unload_min = self._sample_duration(
            dump.unload_time_mean_min, dump.unload_time_std_min,
        )
        truck.total_active_min += self._in_shift(time_min, unload_min)
        time_min += unload_min
        truck.total_fuel_l += estimate_idle_fuel_l(unload_min, self.fuel)
        dump.available_at_min = time_min

        # Retour à vide vers la même pelle : c'est là que le camion attendra
        # sa prochaine décision.
        self.truck_statuses[self.current_truck_idx] = STATUS_RETURNING
        t, f = self._travel_route(dump.node_id, shovel.node_id, False)
        travel_back_min = t
        truck.total_active_min += self._in_shift(time_min, t)
        time_min += t
        truck.total_fuel_l += f
        location = shovel.node_id
        self.truck_locations[self.current_truck_idx] = location

        # Panne éventuelle du camion ; la réparation a son propre compteur (T10).
        # Flux propre au camion, avec deux tirages à chaque cycle, même sans panne :
        # le k-ième cycle de ce camion reçoit ainsi la même panne quelle que soit la politique.
        breakdown_rng = self.rng_breakdowns[self.current_truck_idx]
        breaks_down = breakdown_rng.random() < self.breakdown_probability
        drawn_repair_min = breakdown_rng.uniform(10.0, 30.0)
        repair_min = drawn_repair_min if breaks_down else 0.0
        if breaks_down:
            truck.repair_min += self._in_shift(time_min, repair_min)
            time_min += repair_min
            truck.total_fuel_l += estimate_idle_fuel_l(repair_min, self.fuel)

        # Le tonnage n'est compté que si le cycle se termine avant la fin du poste.
        delivered_tonnage = 0.0
        if time_min < self.episode_minutes:
            truck.cycles_completed += 1
            truck.total_tonnage_t += truck.capacity_tonnes
            delivered_tonnage = truck.capacity_tonnes

        truck.available_at_min = time_min
        self.truck_locations[self.current_truck_idx] = location
        self.truck_statuses[self.current_truck_idx] = STATUS_IDLE

        if self.log_cycles:
            self.cycle_log.append(CycleRecord(
                kind="cycle", truck_id=truck.truck_id, start_min=start_time, end_min=time_min,
                shovel_id=shovel.shovel_id, dump_id=dump.dump_id,
                travel_to_shovel_min=travel_to_shovel_min, shovel_wait_min=wait_load_min,
                load_min=load_min, travel_to_dump_min=travel_to_dump_min,
                dump_wait_min=wait_dump_min, unload_min=unload_min,
                travel_back_min=travel_back_min, repair_min=repair_min,
                delivered_t=delivered_tonnage,
            ))

        # Récompense
        reward = self._compute_reward(
            delivered_tonnage_t=delivered_tonnage,
            route_distance_km=pre_action_route_km,
            resource_waits_min=resource_waits,
            operational_wait_min=wait_load_min + wait_dump_min,
        )
        self._prev_tonnage = sum(t.total_tonnage_t for t in self.trucks)

        # Avancer le temps global et passer au camion suivant
        self._advance_to_next_truck()

        terminated = False
        truncated = self.current_time_min >= self.episode_minutes

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _in_shift(self, start_min: float, duration_min: float) -> float:
        """Part d'une activité [start, start + durée] située avant la fin du poste (T11).

        Sert uniquement aux compteurs (temps actif, attentes, occupation des pelles) :
        la simulation, elle, continue d'avancer de la durée complète. Sans cette borne,
        le dernier cycle de chaque camion était compté jusqu'au bout, au-delà de 480 min.
        """
        return max(0.0, min(duration_min, self.episode_minutes - start_min))

    def _apply_wait_action(self, truck: Truck, time_min: float) -> float:
        """Applique l'action ATTENDRE sans effectuer de cycle de transport.

        Le camion reste immobile jusqu'au prochain événement du système (au plus
        ``wait_action_minutes``, au moins 0,2 min).

        Pourquoi la récompense peut rendre ATTENDRE attrayante (T19, à corriger en T25) :
        elle est calculée avec ``route_distance_km = 0``, donc sans coût de distance, et
        avec un coût d'attente proportionnel à une durée souvent minuscule
        (w4 × 0,2 / 30 ≈ 0,002 pour 0,2 min). Enchaîner de très courtes attentes coûte
        donc presque rien et évite parfois une file : en forte charge, PPO choisit
        ATTENDRE dans 87 % de ses décisions (T12 bis). ``info["wait_actions"]`` compte
        ces décisions.
        """
        future_events = [
            t.available_at_min
            for t in self.trucks
            if t.available_at_min > time_min
        ]
        future_events.extend(
            s.available_at_min for s in self.shovels if s.available_at_min > time_min
        )
        future_events.extend(
            d.available_at_min for d in self.dumps if d.available_at_min > time_min
        )

        if future_events:
            wait_min = min(self.wait_action_minutes, min(future_events) - time_min)
        else:
            wait_min = self.wait_action_minutes
        wait_min = max(0.2, wait_min)

        truck.voluntary_wait_min += self._in_shift(time_min, wait_min)
        if self.log_cycles:
            self.cycle_log.append(CycleRecord(
                kind="wait", truck_id=truck.truck_id, start_min=time_min,
                end_min=time_min + wait_min, voluntary_wait_min=wait_min,
            ))
        truck.total_fuel_l += estimate_idle_fuel_l(wait_min, self.fuel)
        truck.available_at_min = time_min + wait_min
        self.truck_statuses[self.current_truck_idx] = STATUS_IDLE
        self._prev_tonnage = sum(t.total_tonnage_t for t in self.trucks)

        resource_waits = self._shovel_waits(self.current_time_min)
        return self._compute_reward(
            delivered_tonnage_t=0.0,
            route_distance_km=0.0,
            resource_waits_min=resource_waits,
            operational_wait_min=wait_min,
        )

    def _shovel_waits(self, at_time: float) -> list[float]:
        """Temps d'attente estimés à chaque pelle à l'instant donné."""
        return [max(0.0, s.available_at_min - at_time) for s in self.shovels]

    def _advance_to_next_truck(self) -> None:
        """Avance l'horloge globale vers le prochain camion disponible."""
        availability = [t.available_at_min for t in self.trucks]
        self.current_time_min = min(availability)
        self.current_truck_idx = int(np.argmin(availability))

    def _compute_reward(
        self,
        delivered_tonnage_t: float,
        route_distance_km: float,
        resource_waits_min: list[float],
        operational_wait_min: float,
    ) -> float:
        """Récompense d'une décision, calculée par ``simulation.reward`` (T25).

        Mémorise aussi les contributions pondérées des quatre termes : celles de la
        dernière décision et leur cumul sur le poste, exposés dans ``info``.
        """
        terms = reward_terms(
            delivered_tonnage_t, self.capacity_tonnes, route_distance_km,
            resource_waits_min, operational_wait_min,
        )
        self.last_reward_terms = terms.weighted(self.reward_weights)
        for name, value in self.last_reward_terms.items():
            self.episode_reward_terms[name] += value
        return compute_reward(terms, self.reward_weights)

    def _decode_action(self, action: int) -> tuple[Shovel, DumpSite]:
        """Décode l'action en paire (pelle, dump).

        Actions 0..(S*D-1) : shovel_idx = action // D, dump_idx = action % D
        Action S*D          : pelle et verse libres le plus tôt (branche non
                              atteinte depuis ``step``, qui traite ATTENDRE avant)
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
        """Plus court chemin (Dijkstra sur la distance)."""
        assert self.graph is not None
        path = self.graph.shortest_path(src, dst)
        return path if path else [src]

    def _route_distance(self, src: str, dst: str) -> float:
        """Distance totale du plus court chemin entre deux noeuds."""
        assert self.graph is not None
        route = self._find_route(src, dst)
        total = 0.0
        for i in range(len(route) - 1):
            edge = self.graph.get_edge(route[i], route[i + 1])
            total += edge.distance_km
        return total

    def _travel_route(
        self, src: str, dst: str, loaded: bool,
    ) -> tuple[float, float]:
        """Trajet sur le plus court chemin → (durée en min, carburant en L)."""
        assert self.graph is not None and self.rng_travel is not None
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
        """Parcourt une arête : durée aléatoire, carburant, et 5 % de chances
        de dégrader la piste de 0,02 (plancher 0,6)."""
        assert self.graph is not None and self.rng_travel is not None
        travel_min = self.graph.sample_travel_time_minutes(
            src=src, dst=dst, loaded=loaded, rng=self.rng_travel,
        )
        edge = self.graph.get_edge(src, dst)
        travel_fuel_l = estimate_travel_fuel_l(
            distance_km=edge.distance_km,
            slope_pct=edge.slope_pct,
            road_state=edge.road_state,
            loaded=loaded,
            params=self.fuel,
        )
        if self.rng_road.random() < 0.05:
            edge.road_state = max(0.6, edge.road_state - 0.02)
        return travel_min, travel_fuel_l

    def _sample_duration(self, mean_min: float, std_min: float) -> float:
        """Durée log-normale de moyenne ``mean_min`` (minimum 0,2 min)."""
        assert self.rng_service is not None
        sigma = std_min / max(mean_min, 0.1)
        mu = log(max(0.2, mean_min)) - 0.5 * sigma * sigma
        return max(0.2, self.rng_service.lognormvariate(mu, sigma))

    def _get_obs(self) -> np.ndarray:
        """Construit l'observation courante et mémorise sa durée dans ``last_obs_time_ms``."""
        t0 = time.perf_counter()
        obs = build_observation(
            trucks=self.trucks,
            shovels=self.shovels,
            dumps=self.dumps,
            graph=self.graph,
            current_time_min=self.current_time_min,
            episode_minutes=self.episode_minutes,
            truck_locations=self.truck_locations,
            current_truck_idx=self.current_truck_idx,
        )
        self.last_obs_time_ms = (time.perf_counter() - t0) * 1_000
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Dictionnaire d'informations de debug."""
        return {
            "current_time_min": self.current_time_min,
            "current_truck_idx": self.current_truck_idx,
            "total_tonnage_t": sum(t.total_tonnage_t for t in self.trucks),
            "total_fuel_l": sum(t.total_fuel_l for t in self.trucks),
            "total_wait_min": sum(t.total_wait_min for t in self.trucks),
            "queue_wait_min": sum(t.queue_wait_min for t in self.trucks),
            "repair_min": sum(t.repair_min for t in self.trucks),
            "voluntary_wait_min": sum(t.voluntary_wait_min for t in self.trucks),
            "total_cycles": sum(t.cycles_completed for t in self.trucks),
            "decisions": self.decision_count,
            "wait_actions": self.wait_action_count,
            # Contributions pondérées des termes de la récompense (T25) : dernière décision
            # et cumul du poste. Leur somme égale la récompense (resp. la récompense cumulée).
            "reward_terms": dict(self.last_reward_terms),
            "episode_reward_terms": dict(self.episode_reward_terms),
            "shovel_available_at": [s.available_at_min for s in self.shovels],
            "dump_available_at": [d.available_at_min for d in self.dumps],
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
                print(output)
            return output
        return None
