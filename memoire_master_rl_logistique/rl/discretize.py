"""Discrétisation des observations pour Q-Learning et SARSA (T20 : représentation documentée).

L'état discret garde seulement ce qui sert à décider, pour éviter la malédiction de
la dimension :
- l'attente à chacune des S pelles et des D verses, découpée en ``n_bins`` intervalles ;
- l'indice de la pelle la plus proche du camion qui décide.

Avec 3 pelles, 2 verses et 8 intervalles : 8⁵ × 3 = 98 304 états possibles
(``state_space_size``). Le nombre de camions n'y apparaît pas : une Q-table apprise avec
12 camions peut être utilisée telle quelle avec 18.

Les valeurs ci-dessous sont celles qui ont servi à entraîner les Q-tables : les
modifier change l'état discret, donc oblige à réentraîner Q-Learning et SARSA.
"""

import numpy as np

from memoire_master_rl_logistique.env.observation_builder import observation_layout

# Une attente est observée en minutes / 480 (``observation_builder``). Multipliée par
# WAIT_SCALE puis bornée à 1, elle couvre 0 à 480 / WAIT_SCALE = 9,6 min : les attentes
# plus longues tombent toutes dans le dernier intervalle.
WAIT_SCALE = 50.0
WAIT_SATURATION_MIN = 480.0 / WAIT_SCALE

# Nombre d'intervalles par attente : 8 intervalles de 9,6 / 8 = 1,2 min.
DEFAULT_N_BINS = 8


def state_space_size(n_bins: int = DEFAULT_N_BINS, num_shovels: int = 3, num_dumps: int = 2) -> int:
    """Nombre d'états discrets possibles : n_bins^(S + D) × S (pelle la plus proche)."""
    return n_bins ** (num_shovels + num_dumps) * num_shovels


def _extract_decision_features(obs: np.ndarray, num_shovels: int, num_dumps: int) -> np.ndarray:
    """Attentes aux pelles et aux verses, étalées sur [0, 1] (saturation à 9,6 min)."""
    layout = observation_layout(num_shovels, num_dumps)
    features = np.concatenate([obs[layout["shovel_wait"]], obs[layout["dump_wait"]]])
    return np.clip(features * WAIT_SCALE, 0.0, 1.0)


def _discretize_obs(
    obs: np.ndarray, n_bins: int = DEFAULT_N_BINS, num_shovels: int = 3, num_dumps: int = 2,
) -> tuple[int, ...]:
    """État discret : S + D indices d'intervalle (0 à n_bins − 1), puis la pelle la plus proche.

    Les blocs sont lus par leur nom (``observation_layout``) : depuis T18, ces valeurs sont
    les mêmes qu'avant, donc les Q-tables déjà entraînées restent valables.
    """
    reduced = _extract_decision_features(obs, num_shovels, num_dumps)
    indices = list(np.minimum((reduced * n_bins).astype(int), n_bins - 1))
    distances = obs[observation_layout(num_shovels, num_dumps)["distance_to_shovel"]]
    indices.append(int(np.argmin(distances)) if len(distances) else 0)
    return tuple(indices)
