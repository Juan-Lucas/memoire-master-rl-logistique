"""Construction du vecteur d'observation, normalisé dans [0, 1] (T18 : variables agrégées).

La taille ne dépend plus du nombre de camions : 3S + 2D + 4 composantes, soit 17 avec
3 pelles et 2 verses, pour 12 comme pour 18 camions. Un agent entraîné dans un régime
peut donc être évalué dans un autre sans réentraînement.

Avant T18, l'observation contenait un bloc par camion (statut, disponibilité, tonnage,
carburant, position one-hot : 11 valeurs × N) et les distances pelle–verse : 147
composantes pour 12 camions, 213 pour 18, dont 54 constantes (T17).

Disposition (voir ``observation_layout``), dans cet ordre :
- ``shovel_wait`` (S) : temps avant que chaque pelle soit libre, camions en route compris
  (prévision, ``MineEnv.forecast_free_at``) ;
- ``dump_wait`` (D) : même chose pour chaque verse ;
- ``shovel_trucks`` (S) : camions engagés vers chaque pelle (en route, en file ou en
  chargement), divisé par 24 ;
- ``distance_to_shovel`` (S) : distance du camion qui décide à chaque pelle ;
- ``position`` (D + 1) : codage one-hot du lieu du camion qui décide (parc ou verse,
  les seuls lieux où une décision est prise depuis T16 bis) ;
- ``tonnage``, ``fuel`` : cumul du camion qui décide sur le poste ;
- ``time`` : progression du poste.

Les deux premiers blocs et les distances sont les mêmes qu'avant T18 : l'état discret
de Q-Learning et SARSA (``rl/discretize.py``) est construit de la même façon.

T16 bis (décision à la verse) : 18 → 17 composantes. Le bloc ``position`` décrit le parc
et les verses au lieu du parc et des pelles, et ``shovel_trucks`` compte les camions
engagés vers chaque pelle au lieu de ceux qui s'y trouvaient. Les modèles entraînés
avant T16 bis ne sont plus compatibles.
"""

from __future__ import annotations

import numpy as np

from memoire_master_rl_logistique.simulation.constants import FUEL_OBS_MAX_L
from memoire_master_rl_logistique.simulation.entities import DumpSite, Shovel, Truck
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph

# Constantes de normalisation
_MAX_DISTANCE_KM = 10.0
_MAX_TIME_MIN = 480.0  # 8 heures
MAX_TRUCKS = 24  # plafond du nombre de camions (forte charge : 18)
# 25 cycles de 140 t : au-delà du maximum observé (23 cycles, 3 220 t). L'ancien plafond
# de 2 000 t saturait avant la fin du poste (T17).
_MAX_TONNAGE_T = 25 * 140.0
_MAX_FUEL_L = FUEL_OBS_MAX_L


def _normalize_distance(distance_km: float) -> float:
    """Normalise une distance dans [0, 1], 1.0 si aucun chemin n'existe."""
    if distance_km == float("inf"):
        return 1.0
    return min(distance_km / _MAX_DISTANCE_KM, 1.0)


def observation_layout(num_shovels: int, num_dumps: int) -> dict[str, slice]:
    """Position de chaque bloc dans le vecteur d'observation."""
    sizes = [
        ("shovel_wait", num_shovels),
        ("dump_wait", num_dumps),
        ("shovel_trucks", num_shovels),
        ("distance_to_shovel", num_shovels),
        ("position", num_dumps + 1),
        ("tonnage", 1),
        ("fuel", 1),
        ("time", 1),
    ]
    layout, start = {}, 0
    for name, size in sizes:
        layout[name] = slice(start, start + size)
        start += size
    return layout


def observation_size(num_shovels: int, num_dumps: int) -> int:
    """Taille du vecteur d'observation : 3S + 2D + 4, indépendante du nombre de camions."""
    return observation_layout(num_shovels, num_dumps)["time"].stop


def build_observation(
    trucks: list[Truck],
    shovels: list[Shovel],
    dumps: list[DumpSite],
    graph: RoadGraph,
    current_time_min: float,
    episode_minutes: float,
    truck_locations: list[str],
    current_truck_idx: int = 0,
    trucks_per_shovel: list[int] | None = None,
    free_at: tuple[list[float], list[float]] | None = None,
) -> np.ndarray:
    """Construit le vecteur d'observation normalisé (voir la docstring du module).

    ``trucks_per_shovel`` : camions engagés vers chaque pelle (``MineEnv``) ; 0 par défaut.
    ``free_at`` : heures prévues de libération des pelles et des verses, camions en route
    compris (``MineEnv.forecast_free_at``) ; à défaut, ``available_at_min``.
    """
    obs: list[float] = []
    shovel_nodes = [s.node_id for s in shovels]
    dump_nodes = [d.node_id for d in dumps]
    if trucks_per_shovel is None:
        trucks_per_shovel = [0] * len(shovels)

    if free_at is None:
        free_at = ([s.available_at_min for s in shovels], [d.available_at_min for d in dumps])
    for free in (*free_at[0], *free_at[1]):
        wait = max(0.0, free - current_time_min)
        obs.append(min(wait / _MAX_TIME_MIN, 1.0))

    for count in trucks_per_shovel:
        obs.append(min(count / MAX_TRUCKS, 1.0))

    location = truck_locations[current_truck_idx]
    for node in shovel_nodes:
        obs.append(_normalize_distance(graph.shortest_distance(location, node)))

    obs.append(1.0 if location not in dump_nodes else 0.0)  # au parc (début de poste)
    obs.extend(1.0 if location == node else 0.0 for node in dump_nodes)

    truck = trucks[current_truck_idx]
    obs.append(min(truck.total_tonnage_t / _MAX_TONNAGE_T, 1.0))
    obs.append(min(truck.total_fuel_l / _MAX_FUEL_L, 1.0))
    obs.append(min(current_time_min / max(episode_minutes, 1.0), 1.0))

    return np.array(obs, dtype=np.float32)
