"""Discrétisation des observations pour Q-Learning et SARSA (T20 : représentation documentée).

L'état discret garde seulement ce qui sert à décider, pour éviter la malédiction de
la dimension :
- l'attente à chacune des S pelles et des D verses, découpée en ``n_bins`` intervalles ;
- une dernière variable, au choix (``state_feature``) :
  - ``"nearest_shovel"`` (valeur d'origine) : l'indice de la pelle la plus proche du camion
    qui décide (0 à S − 1) ;
  - ``"best_route"`` (variante du point 9 de la règle de mise au point, 17 octobre 2026) :
    l'indice de la route (pelle, verse) de plus petite attente prévue pour ce camion
    (0 à S·D − 1, dans l'ordre des actions), lu dans le bloc ``route_wait`` de
    l'observation, c'est-à-dire la même prévision que MTWT (``simulation.forecast``). À
    égalité (par exemple aucune file), la première route. Les attentes elles-mêmes ne
    sont pas ajoutées : la table passerait de 6⁵ à 6¹¹ états et plus. Essayée une fois
    (essais 24 et 25, 22 octobre 2026), elle n'a pas été retenue.

Avec 3 pelles, 2 verses et 6 intervalles : 6⁵ × 3 = 23 328 états possibles
(``state_space_size``), 6⁵ × 6 = 46 656 avec ``"best_route"``. Le nombre de camions
n'y apparaît pas : une Q-table apprise avec 12 camions peut être utilisée telle quelle
avec 18.

Les valeurs ci-dessous sont celles qui ont servi à entraîner les Q-tables : les
modifier change l'état discret, donc oblige à réentraîner Q-Learning et SARSA. Le
nombre d'intervalles est enregistré avec chaque table (``TabularPolicy``) : une table
apprise avec 8 intervalles (avant le réglage) se relit correctement.
"""

import numpy as np

from memoire_master_rl_logistique.env.observation_builder import observation_layout

# Une attente est observée en minutes / 480 (``observation_builder``). Multipliée par
# WAIT_SCALE puis bornée à 1, elle couvre 0 à 480 / WAIT_SCALE = 9,6 min : les attentes
# plus longues tombent toutes dans le dernier intervalle.
WAIT_SCALE = 50.0
WAIT_SATURATION_MIN = 480.0 / WAIT_SCALE

# Nombre d'intervalles par attente : 6 intervalles de 9,6 / 6 = 1,6 min (réglage du
# 22 octobre 2026, essais 20 et 21 ; 8 intervalles de 1,2 min avant). Égal à n_bins de
# configs/tabular_hyperparameters.yaml (tests/test_config.py).
DEFAULT_N_BINS = 6

# Dernière variable de l'état discret (voir la docstring du module).
STATE_FEATURES = ("nearest_shovel", "best_route")
DEFAULT_STATE_FEATURE = "nearest_shovel"


def _check_feature(state_feature: str) -> None:
    if state_feature not in STATE_FEATURES:
        raise ValueError(f"state_feature inconnu : {state_feature!r} (attendu : {STATE_FEATURES})")


def state_space_size(
    n_bins: int = DEFAULT_N_BINS, num_shovels: int = 3, num_dumps: int = 2,
    state_feature: str = DEFAULT_STATE_FEATURE,
) -> int:
    """Nombre d'états discrets possibles : n_bins^(S + D) × S (pelle la plus proche) ou
    × S·D (route de plus petite attente prévue)."""
    _check_feature(state_feature)
    choices = num_shovels if state_feature == "nearest_shovel" else num_shovels * num_dumps
    return n_bins ** (num_shovels + num_dumps) * choices


def _extract_decision_features(obs: np.ndarray, num_shovels: int, num_dumps: int) -> np.ndarray:
    """Attentes aux pelles et aux verses, étalées sur [0, 1] (saturation à 9,6 min)."""
    layout = observation_layout(num_shovels, num_dumps)
    features = np.concatenate([obs[layout["shovel_wait"]], obs[layout["dump_wait"]]])
    return np.clip(features * WAIT_SCALE, 0.0, 1.0)


def _discretize_obs(
    obs: np.ndarray, n_bins: int = DEFAULT_N_BINS, num_shovels: int = 3, num_dumps: int = 2,
    state_feature: str = DEFAULT_STATE_FEATURE,
) -> tuple[int, ...]:
    """État discret : S + D indices d'intervalle (0 à n_bins − 1), puis la pelle la plus
    proche ou la route de plus petite attente prévue (``state_feature``).

    Les blocs sont lus par leur nom (``observation_layout``) : depuis T18, ces valeurs sont
    les mêmes qu'avant, donc les Q-tables déjà entraînées restent valables.
    """
    _check_feature(state_feature)
    layout = observation_layout(num_shovels, num_dumps)
    reduced = _extract_decision_features(obs, num_shovels, num_dumps)
    indices = list(np.minimum((reduced * n_bins).astype(int), n_bins - 1))
    block = "distance_to_shovel" if state_feature == "nearest_shovel" else "route_wait"
    values = obs[layout[block]]
    indices.append(int(np.argmin(values)) if len(values) else 0)
    return tuple(indices)
