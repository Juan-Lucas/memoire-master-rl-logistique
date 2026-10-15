"""Discrétisation des observations pour Q-Learning et SARSA.

Extrait uniquement les features décisionnelles essentielles pour réduire
l'espace d'état et éviter le problème de la malédiction de la dimension.
"""

import numpy as np

from memoire_master_rl_logistique.env.observation_builder import observation_layout


def _extract_decision_features(obs: np.ndarray, num_shovels: int, num_dumps: int) -> np.ndarray:
    """Extrait les attentes aux pelles et aux verses et les étale sur [0, 1].

    Dans l'observation, une attente vaut minutes / 480. Multipliée par 50 et
    bornée à 1, elle sature au-delà de 480 / 50 = 9,6 min.
    """
    layout = observation_layout(num_shovels, num_dumps)
    features = np.concatenate([obs[layout["shovel_wait"]], obs[layout["dump_wait"]]])
    return np.clip(features * 50.0, 0.0, 1.0)


def _discretize_obs(obs: np.ndarray, n_bins: int, num_shovels: int = 3, num_dumps: int = 2) -> tuple[int, ...]:
    """État discret : 5 attentes (3 pelles, 2 verses) en ``n_bins`` intervalles,
    plus l'indice de la pelle la plus proche du camion courant.

    Avec les réglages par défaut : 8⁵ × 3 = 98 304 états possibles. Le nombre
    de camions n'apparaît pas, d'où l'indépendance vis-à-vis de N. Les blocs sont
    lus par leur nom (``observation_layout``) : depuis T18, ces valeurs sont les
    mêmes qu'avant, donc les Q-tables déjà entraînées restent valables.
    """
    reduced = _extract_decision_features(obs, num_shovels, num_dumps)
    indices = list(np.minimum((reduced * n_bins).astype(int), n_bins - 1))
    distances = obs[observation_layout(num_shovels, num_dumps)["distance_to_shovel"]]
    indices.append(int(np.argmin(distances)) if len(distances) else 0)
    return tuple(indices)
