"""Construction du vecteur d'observation normalisé pour l'agent RL.

L'observation encode l'état s_t du MDP (Eq. 3.3 du mémoire) :
  s_t = ({q_p}, {x_c}, {z_r}, t_courant)

Les valeurs sont normalisées dans [0, 1] pour stabiliser l'apprentissage
(cf. Section 4.3.3 du mémoire — définition de l'espace d'observation).
"""

from __future__ import annotations

import numpy as np

from memoire_master_rl_logistique.simulation.entities import DumpSite, Shovel, Truck
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph

# Constantes de normalisation
_MAX_QUEUE = 10.0
_MAX_DISTANCE_KM = 10.0
_MAX_TIME_MIN = 480.0  # 8 heures


def build_observation(
    trucks: list[Truck],
    shovels: list[Shovel],
    dumps: list[DumpSite],
    graph: RoadGraph,
    current_time_min: float,
    episode_minutes: float,
    truck_locations: list[str],
    truck_statuses: list[int],
) -> np.ndarray:
    """Construit le vecteur d'observation normalisé (Section 4.3.3 du mémoire).

    Composantes conformes au mémoire :
    - Disponibilité des pelles : temps estimé avant disponibilité (proxy de la file)
    - Disponibilité des dumps : temps estimé avant disponibilité
    - Statut des camions : libre, en route vers pelle, en chargement,
      en route vers dump, en déchargement, en retour
    - Disponibilité temporelle des camions : temps avant disponibilité
    - Tonnage accumulé des camions : normalisé
    - Carburant consommé des camions : normalisé
    - Temps courant : progression de l'épisode normalisée

    Toutes les valeurs sont normalisées dans [0, 1] pour stabiliser l'apprentissage.
    """
    obs_parts: list[float] = []

    # --- Files d'attente aux pelles (q_p) ---
    for shovel in shovels:
        wait_until = max(0.0, shovel.available_at_min - current_time_min)
        obs_parts.append(min(wait_until / _MAX_TIME_MIN, 1.0))

    # --- Disponibilité des dumps (z_r) ---
    for dump in dumps:
        wait_until = max(0.0, dump.available_at_min - current_time_min)
        obs_parts.append(min(wait_until / _MAX_TIME_MIN, 1.0))

    # --- État des camions (x_c) ---
    for i, truck in enumerate(trucks):
        # Statut : 0=libre, 1=en_route_pelle, 2=chargement, 3=en_route_dump,
        #          4=déchargement, 5=en_route_retour
        status_norm = truck_statuses[i] / 5.0
        obs_parts.append(status_norm)

        # Disponibilité temporelle normalisée
        avail = max(0.0, truck.available_at_min - current_time_min)
        obs_parts.append(min(avail / _MAX_TIME_MIN, 1.0))

        # Tonnage accumulé normalisé
        obs_parts.append(min(truck.total_tonnage_t / 2000.0, 1.0))

        # Carburant consommé normalisé
        obs_parts.append(min(truck.total_fuel_l / 500.0, 1.0))

    # --- Temps courant normalisé ---
    obs_parts.append(min(current_time_min / max(episode_minutes, 1.0), 1.0))

    return np.array(obs_parts, dtype=np.float32)


def observation_size(num_trucks: int, num_shovels: int, num_dumps: int) -> int:
    """Retourne la taille du vecteur d'observation."""
    shovel_feats = num_shovels  # 1 feature par pelle
    dump_feats = num_dumps  # 1 feature par dump
    truck_feats = num_trucks * 4  # 4 features par camion
    time_feat = 1
    return shovel_feats + dump_feats + truck_feats + time_feat
