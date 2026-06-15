"""Discrétisation des observations pour Q-Learning et SARSA.

Extrait uniquement les features décisionnelles essentielles pour réduire
l'espace d'état et éviter le problème de la malédiction de la dimension.
"""

import numpy as np


def _extract_decision_features(obs: np.ndarray, num_shovels: int, num_dumps: int) -> np.ndarray:
    """Extrait et renormalise les features décisionnelles.

    Normalisation par max réaliste :
    - Attente pelle max réaliste : 30 min → 30/480 = 0.0625
    - On renormalise pour étaler sur [0,1] avec facteur ×50
    """
    shovel_waits = obs[:num_shovels]
    dump_waits = obs[num_shovels:num_shovels + num_dumps]
    features = np.concatenate([shovel_waits, dump_waits])

    # Renormalisation : les valeurs sont dans [0, 0.02] en pratique
    # On les étale sur [0, 1] en multipliant par 50
    features = np.clip(features * 50.0, 0.0, 1.0)
    return features


def _discretize_obs(obs: np.ndarray, n_bins: int, num_shovels: int = 3, num_dumps: int = 2) -> tuple[int, ...]:
    """Discrétise uniquement les features essentielles."""
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
