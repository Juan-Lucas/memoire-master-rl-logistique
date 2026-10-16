"""Environnement Gymnasium pour le dispatching camions–pelles.

Une décision est prise chaque fois qu'un camion devient disponible : au parc au début
du poste, puis à la verse où il vient de déverser (T16 bis). Le cycle décidé (trajet à
vide, file et chargement à la pelle, trajet chargé, file et déversement à la verse, panne
éventuelle) est simulé par **événements** (T15) : arrivée à la pelle, arrivée à la verse,
traités dans l'ordre chronologique de toute la flotte. Pelles et verses servent donc les
camions dans leur ordre d'arrivée (``available_at_min`` : heure où la ressource se libère).

Avant T15 et T16 bis, ``step`` simulait tout le cycle d'un seul tenant au moment de la
décision, et le camion revenait à sa pelle avant de décider. Deux biais en découlaient :
un détour verse → ancienne pelle → nouvelle pelle pour toute politique qui change de
pelle (≈ 3,5 km ; FIFO y perdait 28 % de productivité), et des verses servies dans
l'ordre des décisions plutôt que des arrivées.

Aléas (T13) : la graine d'un épisode engendre des flux aléatoires indépendants
(``numpy.random.SeedSequence``), un par processus : trajets, chargements et
déversements, dégradation des pistes, et un flux de pannes **par camion**. Deux
politiques jouées avec la même graine partagent donc les mêmes aléas
(nombres aléatoires communs) ; en particulier, le k-ième cycle d'un camion donné a
la même panne, de même durée, quelle que soit la politique.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# État d'un camion (``truck_statuses``) : en attente de décision, en route vers la pelle
# (prochain événement : l'arrivée à la pelle), ou engagé vers la verse (file, chargement,
# trajet chargé ; prochain événement : l'arrivée à la verse).
STATUS_IDLE = 0
STATUS_TO_SHOVEL = 1
STATUS_TO_DUMP = 2

REWARD_TERM_NAMES = tuple(RewardTerms.__dataclass_fields__)

# Flux aléatoires communs à tous les camions ; viennent ensuite les flux de pannes, un par camion.
SHARED_STREAMS = ("travel", "service", "road")


@dataclass
class _CycleInProgress:
    """Ce qu'il faut garder d'un cycle entre sa décision et sa fin (moteur à événements)."""

    shovel_idx: int
    dump_idx: int
    start_min: float
    route_km: float
    shovel_waits_at_decision: list[float]
    travel_to_shovel_min: float
    shovel_wait_min: float = 0.0
    load_min: float = 0.0
    travel_to_dump_min: float = 0.0


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
        seed_step: int = 1,
    ) -> None:
        super().__init__()
        if seed_step < 1:
            raise ValueError(f"seed_step doit être ≥ 1 : {seed_step}")

        self.truck_count = truck_count
        self.shovel_count = shovel_count
        self.dump_count = dump_count
        self.episode_minutes = episode_minutes
        self._seed = seed
        # Pas du compteur de graines (T29) : avec n environnements d'entraînement en
        # parallèle, l'environnement i joue base + i, base + i + n, … sans doublon.
        self._seed_step = seed_step
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
        # Pelle du dernier cycle de chaque camion et fin de son chargement (T16 bis) :
        # un camion compte pour une pelle tant qu'il n'y a pas fini de charger.
        self.truck_shovels: list[int | None] = []
        self.truck_load_end_min: list[float] = []
        self._cycles: list[_CycleInProgress | None] = []
        # Durée (ms) de la dernière construction d'observation : mesure seulement,
        # n'influence pas la simulation (voir run_single_episode).
        self.last_obs_time_ms: float = 0.0
        self.last_episode_seed: int | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Initialise un nouveau poste.

        Sans graine, un compteur interne est utilisé : graine précédente + ``seed_step``
        (graine du constructeur au départ). C'est le cas pendant l'entraînement de PPO et
        DQN, où seule la première remise à zéro reçoit une graine. Depuis T29, une graine
        explicite repositionne le compteur ; avec un seul environnement, la suite des
        graines est inchangée (graine, + 1, + 2, …).
        """
        super().reset(seed=seed)
        if seed is None:
            self._episode_counter = getattr(self, "_episode_counter", self._seed) + self._seed_step
        else:
            self._episode_counter = seed
        effective_seed = self._episode_counter
        # Graine réellement utilisée par cet épisode (T26 : vérifier qu'aucune graine
        # d'évaluation ne sert pendant un entraînement, compteur interne compris).
        self.last_episode_seed = effective_seed
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
        self.truck_shovels = [None] * self.truck_count
        self.truck_load_end_min = [0.0] * self.truck_count
        self._cycles = [None] * self.truck_count
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
        """Applique la décision du camion courant, puis simule jusqu'à la décision suivante.

        L'action encode une paire (pelle, verse) :
          shovel_idx = action // dump_count
          dump_idx   = action %  dump_count
        Si action == shovel_count * dump_count → ATTENDRE.

        Un cycle va du lieu du camion (parc au début du poste, verse ensuite) à la pelle
        choisie, puis à la verse choisie, et **se termine à la verse**, après le
        déversement : c'est là que le camion reçoit sa décision suivante (T16 bis).

        Moteur à événements (T15) : la décision lance seulement le trajet vers la pelle.
        File et chargement sont simulés à l'**arrivée** du camion à la pelle, file et
        déversement à son arrivée à la verse, dans l'ordre chronologique de toute la
        flotte : pelles et verses servent les camions dans leur ordre d'arrivée.

        Récompense (option 1a) : celle d'un cycle est connue à sa fin (files réelles,
        tonnage livré) ; elle est versée au ``step`` pendant lequel le cycle se termine.
        La récompense d'un ``step`` est donc la somme des récompenses des cycles terminés
        depuis la décision précédente (de n'importe quel camion), plus celle d'ATTENDRE
        si c'est l'action choisie. La somme sur un poste est la somme des récompenses de
        tous ses cycles : à la fin du poste, les cycles en cours sont menés à terme.
        """
        assert self.graph is not None and self.rng_travel is not None
        action = int(action)
        if not self.action_space.contains(action):
            msg = f"Invalid action {action}; expected 0..{self.num_actions - 1}."
            raise ValueError(msg)

        idx = self.current_truck_idx
        time_min = self.current_time_min
        self.decision_count += 1
        self.last_reward_terms = dict.fromkeys(REWARD_TERM_NAMES, 0.0)

        reward = 0.0
        if action == self.shovel_count * self.dump_count:
            self.wait_action_count += 1
            reward += self._apply_wait_action(self.trucks[idx], time_min)
        else:
            self._start_cycle(idx, action, time_min)

        reward += self._run_until_decision()
        truncated = self.current_time_min >= self.episode_minutes
        if truncated:
            reward += self._finish_cycles()

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, False, truncated, info

    # --- Moteur à événements (T15, T16 bis) ------------------------------------------

    def _start_cycle(self, idx: int, action: int, time_min: float) -> None:
        """Décision d'un cycle : le camion part vers la pelle choisie."""
        truck = self.trucks[idx]
        shovel, dump = self._decode_action(action)
        location = self.truck_locations[idx]

        travel_min = 0.0
        if location != shovel.node_id:
            travel_min, fuel_l = self._travel_route(location, shovel.node_id, False)
            truck.total_active_min += self._in_shift(time_min, travel_min)
            truck.total_fuel_l += fuel_l

        self._cycles[idx] = _CycleInProgress(
            shovel_idx=self.shovels.index(shovel),
            dump_idx=self.dumps.index(dump),
            start_min=time_min,
            # Distance du cycle : lieu → pelle (à vide) → verse (chargé). Le retour à vide
            # est le premier trajet du cycle suivant (T16 bis).
            route_km=(self._route_distance(location, shovel.node_id)
                      + self._route_distance(shovel.node_id, dump.node_id)),
            shovel_waits_at_decision=self._shovel_waits(time_min),
            travel_to_shovel_min=travel_min,
        )
        self.truck_shovels[idx] = self._cycles[idx].shovel_idx
        self.truck_load_end_min[idx] = float("inf")  # en route : pas encore chargé
        self.truck_statuses[idx] = STATUS_TO_SHOVEL
        truck.available_at_min = time_min + travel_min  # prochain événement : l'arrivée

    def _next_event(self, include_decisions: bool = True) -> int | None:
        """Camion du prochain événement : le plus tôt ; à égalité, les arrivées passent
        avant les décisions (l'état d'une décision tient compte de tout ce qui se passe à
        cet instant), puis le plus petit indice."""
        candidates = [
            i for i in range(self.truck_count)
            if include_decisions or self.truck_statuses[i] != STATUS_IDLE
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda i: (self.trucks[i].available_at_min,
                                              self.truck_statuses[i] == STATUS_IDLE, i))

    def _run_until_decision(self) -> float:
        """Traite les arrivées dans l'ordre chronologique jusqu'au prochain camion qui doit
        décider ; renvoie la récompense des cycles terminés entre-temps."""
        reward = 0.0
        while True:
            idx = self._next_event()
            if self.truck_statuses[idx] == STATUS_IDLE:
                self.current_time_min = self.trucks[idx].available_at_min
                self.current_truck_idx = idx
                return reward
            reward += self._process_arrival(idx)

    def _finish_cycles(self) -> float:
        """Fin du poste : mène à terme les cycles en cours, sans nouvelle décision.

        Comme dans la v1, tout cycle décidé est simulé en entier (carburant et récompense
        compris) ; son tonnage ne compte que s'il est livré avant la fin du poste.
        """
        reward = 0.0
        while (idx := self._next_event(include_decisions=False)) is not None:
            reward += self._process_arrival(idx)
        return reward

    def _process_arrival(self, idx: int) -> float:
        if self.truck_statuses[idx] == STATUS_TO_SHOVEL:
            self._arrive_at_shovel(idx)
            return 0.0
        return self._arrive_at_dump(idx)

    def _arrive_at_shovel(self, idx: int) -> None:
        """Arrivée à la pelle : file (premier arrivé, premier servi), chargement, puis
        départ chargé vers la verse."""
        truck, cycle = self.trucks[idx], self._cycles[idx]
        shovel, dump = self.shovels[cycle.shovel_idx], self.dumps[cycle.dump_idx]
        time_min = truck.available_at_min
        self.truck_locations[idx] = shovel.node_id

        cycle.shovel_wait_min = max(0.0, shovel.available_at_min - time_min)
        if cycle.shovel_wait_min > 0:
            truck.queue_wait_min += self._in_shift(time_min, cycle.shovel_wait_min)
            truck.total_fuel_l += estimate_idle_fuel_l(cycle.shovel_wait_min, self.fuel)
            time_min += cycle.shovel_wait_min

        cycle.load_min = self._sample_duration(
            shovel.load_time_mean_min, shovel.load_time_std_min,
        )
        truck.total_active_min += self._in_shift(time_min, cycle.load_min)
        shovel.busy_min += self._in_shift(time_min, cycle.load_min)
        truck.total_fuel_l += estimate_idle_fuel_l(cycle.load_min, self.fuel)
        time_min += cycle.load_min
        shovel.available_at_min = time_min
        self.truck_load_end_min[idx] = time_min

        cycle.travel_to_dump_min, fuel_l = self._travel_route(shovel.node_id, dump.node_id,
                                                              True)
        truck.total_active_min += self._in_shift(time_min, cycle.travel_to_dump_min)
        truck.total_fuel_l += fuel_l
        self.truck_statuses[idx] = STATUS_TO_DUMP
        truck.available_at_min = time_min + cycle.travel_to_dump_min

    def _arrive_at_dump(self, idx: int) -> float:
        """Arrivée à la verse : file, déversement, panne éventuelle ; fin du cycle et
        calcul de sa récompense. Le camion reste à la verse pour sa prochaine décision."""
        truck, cycle = self.trucks[idx], self._cycles[idx]
        dump = self.dumps[cycle.dump_idx]
        time_min = truck.available_at_min
        self.truck_locations[idx] = dump.node_id

        dump_wait_min = max(0.0, dump.available_at_min - time_min)
        if dump_wait_min > 0:
            truck.queue_wait_min += self._in_shift(time_min, dump_wait_min)
            truck.total_fuel_l += estimate_idle_fuel_l(dump_wait_min, self.fuel)
            time_min += dump_wait_min

        unload_min = self._sample_duration(
            dump.unload_time_mean_min, dump.unload_time_std_min,
        )
        truck.total_active_min += self._in_shift(time_min, unload_min)
        truck.total_fuel_l += estimate_idle_fuel_l(unload_min, self.fuel)
        time_min += unload_min
        dump.available_at_min = time_min

        # Tonnage livré au déversement : compté s'il se termine avant la fin du poste.
        delivered_tonnage = 0.0
        if time_min < self.episode_minutes:
            truck.cycles_completed += 1
            truck.total_tonnage_t += truck.capacity_tonnes
            delivered_tonnage = truck.capacity_tonnes

        # Panne éventuelle du camion, à la verse ; la réparation a son propre compteur (T10).
        # Flux propre au camion, avec deux tirages à chaque cycle, même sans panne :
        # le k-ième cycle de ce camion reçoit ainsi la même panne quelle que soit la politique.
        breakdown_rng = self.rng_breakdowns[idx]
        breaks_down = breakdown_rng.random() < self.breakdown_probability
        drawn_repair_min = breakdown_rng.uniform(10.0, 30.0)
        repair_min = drawn_repair_min if breaks_down else 0.0
        if breaks_down:
            truck.repair_min += self._in_shift(time_min, repair_min)
            truck.total_fuel_l += estimate_idle_fuel_l(repair_min, self.fuel)
            time_min += repair_min

        truck.available_at_min = time_min
        self.truck_statuses[idx] = STATUS_IDLE
        self._cycles[idx] = None

        if self.log_cycles:
            self.cycle_log.append(CycleRecord(
                kind="cycle", truck_id=truck.truck_id, start_min=cycle.start_min,
                end_min=time_min, shovel_id=self.shovels[cycle.shovel_idx].shovel_id,
                dump_id=dump.dump_id, travel_to_shovel_min=cycle.travel_to_shovel_min,
                shovel_wait_min=cycle.shovel_wait_min, load_min=cycle.load_min,
                travel_to_dump_min=cycle.travel_to_dump_min, dump_wait_min=dump_wait_min,
                unload_min=unload_min, repair_min=repair_min, delivered_t=delivered_tonnage,
            ))

        return self._compute_reward(
            delivered_tonnage_t=delivered_tonnage,
            route_distance_km=cycle.route_km,
            resource_waits_min=cycle.shovel_waits_at_decision,
            operational_wait_min=cycle.shovel_wait_min + dump_wait_min,
        )

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

        Dans la v1, ATTENDRE ne coûtait presque rien (T19) : pas de coût de distance et
        un coût d'attente proportionnel à une durée souvent minuscule (w4 × 0,2 / 30 ≈ 0,002
        pour 0,2 min). En forte charge, PPO choisissait ATTENDRE dans 87 % de ses décisions
        (T12 bis). Depuis la v2, la récompense ajoute le coût d'opportunité de l'immobilité
        (``idle_min``, voir ``simulation.reward``). ``info["wait_actions"]`` compte ces
        décisions.
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

        resource_waits = self._shovel_waits(self.current_time_min)
        return self._compute_reward(
            delivered_tonnage_t=0.0,
            route_distance_km=0.0,
            resource_waits_min=resource_waits,
            operational_wait_min=wait_min,
            # Production perdue pendant l'attente, seulement la part comprise dans le poste.
            idle_min=self._in_shift(time_min, wait_min),
        )

    def _trucks_per_shovel(self) -> list[int]:
        """Camions engagés vers chaque pelle : en route, en file ou en chargement (T16 bis).

        Avant T16 bis, l'observation comptait les camions situés à chaque pelle, puisque
        tout camion y revenait en fin de cycle. Il finit maintenant son cycle à la verse :
        on compte donc ceux qui n'ont pas encore fini de charger à la pelle de leur cycle.
        """
        counts = [0] * self.shovel_count
        for shovel_idx, load_end in zip(self.truck_shovels, self.truck_load_end_min,
                                        strict=True):
            if shovel_idx is not None and load_end > self.current_time_min:
                counts[shovel_idx] += 1
        return counts

    def forecast_free_at(self) -> tuple[list[float], list[float]]:
        """Heure prévue de libération de chaque pelle et de chaque verse (T15).

        ``available_at_min`` ne tient compte que des camions déjà arrivés. La prévision
        ajoute les camions engagés, comme le ferait un système de dispatching qui connaît
        les camions en route : chacun, dans l'ordre de son arrivée, occupe la ressource
        pendant une durée **moyenne** de chargement (ou de déversement). Avant T15, la
        réservation faite à la décision jouait ce rôle.

        Sert à l'observation, à ``info`` (FIFO, MTWT) et au terme d'équilibre ; les
        files réelles, elles, sont simulées à l'arrivée.
        """
        shovels = [s.available_at_min for s in self.shovels]
        dumps = [d.available_at_min for d in self.dumps]
        for status, free, resources, attr, mean in (
            (STATUS_TO_SHOVEL, shovels, self.shovels, "shovel_idx", "load_time_mean_min"),
            (STATUS_TO_DUMP, dumps, self.dumps, "dump_idx", "unload_time_mean_min"),
        ):
            heading = sorted((self.trucks[i].available_at_min, i)
                             for i in range(self.truck_count) if self.truck_statuses[i] == status)
            for arrival, i in heading:
                k = getattr(self._cycles[i], attr)
                free[k] = max(free[k], arrival) + getattr(resources[k], mean)
        return shovels, dumps

    def _shovel_waits(self, at_time: float) -> list[float]:
        """Attente prévue à chaque pelle à l'instant donné (``forecast_free_at``)."""
        return [max(0.0, free - at_time) for free in self.forecast_free_at()[0]]

    def _compute_reward(
        self,
        delivered_tonnage_t: float,
        route_distance_km: float,
        resource_waits_min: list[float],
        operational_wait_min: float,
        idle_min: float = 0.0,
    ) -> float:
        """Récompense d'une décision, calculée par ``simulation.reward`` (T25).

        Mémorise aussi les contributions pondérées des termes, exposées dans ``info`` :
        leur somme sur le ``step`` en cours (``reward_terms``, plusieurs cycles peuvent s'y
        terminer) et leur cumul sur le poste (``episode_reward_terms``).
        """
        terms = reward_terms(
            delivered_tonnage_t, self.capacity_tonnes, route_distance_km,
            resource_waits_min, operational_wait_min, idle_min,
        )
        for name, value in terms.weighted(self.reward_weights).items():
            self.last_reward_terms[name] += value
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
            trucks_per_shovel=self._trucks_per_shovel(),
            free_at=self.forecast_free_at(),
        )
        self.last_obs_time_ms = (time.perf_counter() - t0) * 1_000
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Dictionnaire d'informations de debug."""
        shovel_free, dump_free = self.forecast_free_at()
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
            # Libération prévue, camions en route compris (T15) : lue par FIFO et MTWT.
            "shovel_available_at": shovel_free,
            "dump_available_at": dump_free,
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
