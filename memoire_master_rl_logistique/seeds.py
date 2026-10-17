"""Toutes les graines du projet, et la garantie qu'elles ne se recouvrent pas (T26, M7).

Trois usages, trois ensembles disjoints :
- **évaluation** : graines 42 à 71 (``Scenario.seeds``), les 30 réplications appariées ;
- **sélection des hyperparamètres** (T25 bis) : graines 20 000 à 20 009 ;
- **entraînement** : un agent est entraîné avec une *graine d'entraînement* s parmi
  ``TRAIN_SEEDS`` (1 à 5). Elle fixe l'initialisation du réseau et l'exploration ; les
  épisodes d'entraînement utilisent les graines s × 1 000 000, s × 1 000 000 + 1, …

Avant T26, l'entraînement utilisait la graine 42 puis 43, 44… (épisodes successifs) :
les graines d'évaluation 42 à 71 étaient exactement les premiers épisodes
d'entraînement, et une seule graine d'entraînement existait.

Écart volontaire avec le plan, qui proposait de déplacer l'évaluation vers 10 000 à
10 029 : c'est l'entraînement qui est déplacé. Le résultat est le même (ensembles
disjoints, vérifié par les tests), sans changer les graines d'évaluation des tâches
précédentes.

Valeurs lues dans ``configs/experiment_config.yaml`` (rubrique ``seeds``, T28).
"""

from __future__ import annotations

from memoire_master_rl_logistique.config import experiment_config

_SEEDS = experiment_config()["seeds"]


def _seed_range(block: dict) -> tuple[int, ...]:
    return tuple(range(block["start"], block["start"] + block["count"]))


EVAL_SEEDS = _seed_range(_SEEDS["evaluation"])
TUNING_SEEDS = _seed_range(_SEEDS["tuning"])
TRAIN_SEEDS = tuple(_SEEDS["training"])

# Écart entre les graines d'épisode de deux graines d'entraînement : bien plus que le
# nombre d'épisodes d'un entraînement (30 000 au plus, pour Q-Learning et SARSA).
TRAINING_SEED_STRIDE = _SEEDS["training_episode_stride"]


def training_episode_seed(train_seed: int, episode: int = 0) -> int:
    """Graine du ``episode``-ième épisode d'entraînement d'un agent de graine ``train_seed``."""
    if not 0 <= episode < TRAINING_SEED_STRIDE:
        raise ValueError(f"Épisode hors plage : {episode}")
    return train_seed * TRAINING_SEED_STRIDE + episode


def training_episode_range(train_seed: int) -> range:
    """Toutes les graines d'épisode possibles pour une graine d'entraînement."""
    start = training_episode_seed(train_seed)
    return range(start, start + TRAINING_SEED_STRIDE)
