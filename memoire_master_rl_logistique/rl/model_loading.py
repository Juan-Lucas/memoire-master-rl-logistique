"""Chargement contrôlé des modèles entraînés (T8).

Point d'entrée unique pour recharger un agent : ``load_model`` vérifie d'abord le
``metadata.json`` du modèle (``check_model_compatibility``), puis seulement le charge.
Tous les scripts qui évaluent un agent passent par ici ; aucun ne doit appeler
``DQN.load``, ``PPO.load`` ou ``TabularPolicy.load`` directement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO

from memoire_master_rl_logistique.rl.model_metadata import EnvConfig, check_model_compatibility
from memoire_master_rl_logistique.rl.train_q_learning import QLearningPolicy
from memoire_master_rl_logistique.rl.train_sarsa import SarsaPolicy

# Nom du fichier de modèle dans son dossier, et fonction de chargement, par algorithme.
_LOADERS = {
    "Q-Learning": ("q_table.pkl", QLearningPolicy.load),
    "SARSA": ("sarsa_table.pkl", SarsaPolicy.load),
    "DQN": ("dqn_mine_agent.zip", lambda path: DQN.load(str(path))),
    "PPO": ("ppo_mine_agent.zip", lambda path: PPO.load(str(path))),
}
ALGORITHMS = tuple(_LOADERS)


def model_file(model_dir: str | Path, algorithm: str) -> Path:
    """Chemin du fichier de modèle d'un algorithme dans son dossier."""
    if algorithm not in _LOADERS:
        raise ValueError(f"Algorithme inconnu : {algorithm!r} (attendu : {ALGORITHMS})")
    return Path(model_dir) / _LOADERS[algorithm][0]


def load_model(
    model_dir: str | Path,
    algorithm: str,
    env: Any,
    trained_on: EnvConfig | None = None,
    allow_legacy: bool = False,
) -> Any:
    """Vérifie puis charge le modèle ``algorithm`` rangé dans ``model_dir``.

    ``env`` est l'environnement d'évaluation : il fournit la taille d'observation
    que le modèle recevra et, par défaut, la configuration attendue.
    ``trained_on`` remplace cette configuration quand on évalue volontairement un
    modèle hors de son régime d'entraînement (scénarios ``unseen_*`` : modèle nominal).

    Lève ``MissingMetadataError`` ou ``ModelCompatibilityError`` sans rien charger.
    """
    path = model_file(model_dir, algorithm)
    check_model_compatibility(
        model_dir,
        algorithm,
        trained_on if trained_on is not None else EnvConfig.from_env(env),
        observation_size=env.observation_space.shape[0],
        allow_legacy=allow_legacy,
    )
    return _LOADERS[algorithm][1](path)
