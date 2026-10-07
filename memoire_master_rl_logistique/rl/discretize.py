"""Discrétisation des observations pour Q-Learning et SARSA.

Extrait uniquement les features décisionnelles essentielles pour réduire
l'espace d'état et éviter le problème de la malédiction de la dimension.
"""

import numpy as np


def _extract_decision_features(obs: np.ndarray, num_shovels: int, num_dumps: int) -> np.ndarray:
    """Extrait les attentes aux pelles et aux verses et les étale sur [0, 1].

    Dans l'observation, une attente vaut minutes / 480. Multipliée par 50 et
    bornée à 1, elle sature au-delà de 480 / 50 = 9,6 min.
    """
    shovel_waits = obs[:num_shovels]
    dump_waits = obs[num_shovels:num_shovels + num_dumps]
    features = np.concatenate([shovel_waits, dump_waits])

    features = np.clip(features * 50.0, 0.0, 1.0)
    return features


def _discretize_obs(obs: np.ndarray, n_bins: int, num_shovels: int = 3, num_dumps: int = 2) -> tuple[int, ...]:
    """État discret : 5 attentes (3 pelles, 2 verses) en ``n_bins`` intervalles,
    plus l'indice de la pelle la plus proche du camion courant.

    Avec les réglages par défaut : 8⁵ × 3 = 98 304 états possibles. Le nombre
    de camions n'apparaît pas, d'où l'indépendance vis-à-vis de N.
    """
    reduced = _extract_decision_features(obs, num_shovels, num_dumps)
    clipped = np.clip(reduced, 0.0, 1.0)
    indices = list(np.minimum((clipped * n_bins).astype(int), n_bins - 1))

    # Pelle la plus proche du camion courant
    # Layout depuis la fin : [time(1), shovel_to_dump(S*D), cur_to_shovel(S), ...]
    sd_feats = num_shovels * num_dumps
    offset = 1 + sd_feats
    cur_to_shovel = obs[-(offset + num_shovels):-offset]
    if len(cur_to_shovel) == num_shovels and num_shovels > 0:
        indices.append(int(np.argmin(cur_to_shovel)))
    else:
        indices.append(0)

    return tuple(indices)
