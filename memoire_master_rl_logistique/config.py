"""Lecture des fichiers de configuration de ``configs/`` (T28).

Les régimes, les graines et les hyperparamètres ne sont écrits qu'à un seul endroit :
les fichiers YAML de ``configs/``. Ce module les lit, les valide et les fournit au code
(``seeds.py``, ``experiments/scenarios.py``, ``rl/train_*.py``, ``tune_hyperparams``).
Avant T28, ces fichiers existaient mais aucun script ne les lisait : les valeurs étaient
recopiées en dur et pouvaient diverger sans que personne ne le voie.

Validation stricte : une clé manquante ou inconnue (faute de frappe) lève ``ValueError``,
au lieu d'être ignorée en silence.

Dossier lu : ``configs/`` du dépôt, ou celui de la variable d'environnement
``MINE_RL_CONFIG_DIR``. Chaque lancement archive les fichiers utilisés dans son dossier
de résultats (``snapshot_configs``) ; pour rejouer ce lancement, il suffit de pointer
``MINE_RL_CONFIG_DIR`` vers cette archive.
"""

from __future__ import annotations

import copy
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR_ENV = "MINE_RL_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

EXPERIMENT_FILE = "experiment_config.yaml"
HYPERPARAMETER_FILES = {
    "PPO": "ppo_hyperparameters.yaml",
    "DQN": "dqn_hyperparameters.yaml",
    "Q-Learning": "tabular_hyperparameters.yaml",
    "SARSA": "tabular_hyperparameters.yaml",
}
CONFIG_FILES = (EXPERIMENT_FILE, *dict.fromkeys(HYPERPARAMETER_FILES.values()))

# Clés attendues, ni plus ni moins.
EXPERIMENT_KEYS = {"project", "seeds", "scenario_defaults", "scenarios", "main_scenarios",
                   "tuning", "heuristics", "reproducibility"}
SEED_KEYS = {"evaluation", "tuning", "training", "training_episode_stride"}
HEURISTIC_KEYS = {"lp_max_utilization", "lp_utilization_candidates"}
HYPERPARAMETER_KEYS = {
    "PPO": {"learning_rate", "gamma", "batch_size", "gae_lambda", "clip_range", "n_steps",
            "n_epochs", "ent_coef", "net_arch", "n_envs", "normalize_reward"},
    "DQN": {"learning_rate", "gamma", "batch_size", "buffer_size", "learning_starts",
            "target_update_interval", "exploration_fraction", "exploration_initial_eps",
            "exploration_final_eps", "net_arch"},
    "tabular": {"n_episodes", "alpha", "gamma", "epsilon_start", "epsilon_min", "n_bins"},
}


def config_dir() -> Path:
    """Dossier de configuration utilisé : ``MINE_RL_CONFIG_DIR`` s'il est défini, sinon ``configs/``."""
    return Path(os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR)


def _check_keys(found: set[str], expected: set[str], where: str) -> None:
    missing, unknown = expected - found, found - expected
    if missing or unknown:
        raise ValueError(f"{where} : clés manquantes {sorted(missing)}, inconnues {sorted(unknown)}")


@lru_cache(maxsize=None)
def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"{path} : un dictionnaire YAML est attendu")
    return content


def experiment_config(directory: str | Path | None = None) -> dict[str, Any]:
    """Contenu validé de ``experiment_config.yaml`` (copie : le modifier est sans effet)."""
    path = Path(directory or config_dir()) / EXPERIMENT_FILE
    config = copy.deepcopy(_read_yaml(str(path)))
    _check_keys(set(config), EXPERIMENT_KEYS, str(path))
    _check_keys(set(config["seeds"]), SEED_KEYS, f"{path} (seeds)")
    _check_keys(set(config["heuristics"]), HEURISTIC_KEYS, f"{path} (heuristics)")
    unknown = set(config["main_scenarios"]) - set(config["scenarios"])
    if unknown:
        raise ValueError(f"{path} : main_scenarios cite des scénarios non définis {sorted(unknown)}")
    return config


def hyperparameters(
    algorithm: str, directory: str | Path | None = None, **overrides: Any,
) -> dict[str, Any]:
    """Hyperparamètres validés d'un algorithme ; ``overrides`` remplace des valeurs du fichier.

    Une valeur ``None`` dans ``overrides`` est ignorée (paramètre non fourni par l'appelant).
    """
    if algorithm not in HYPERPARAMETER_FILES:
        raise ValueError(f"Algorithme inconnu : {algorithm!r} (attendu : {tuple(HYPERPARAMETER_FILES)})")
    path = Path(directory or config_dir()) / HYPERPARAMETER_FILES[algorithm]
    params = copy.deepcopy(_read_yaml(str(path)))
    family = params.pop("algorithm", None)
    if family not in HYPERPARAMETER_KEYS or (family != "tabular" and family != algorithm):
        raise ValueError(f"{path} : « algorithm: {family} » ne correspond pas à {algorithm}")
    expected = HYPERPARAMETER_KEYS[family]
    _check_keys(set(params), expected, str(path))

    given = {k: v for k, v in overrides.items() if v is not None}
    if set(given) - expected:
        raise ValueError(f"Hyperparamètres inconnus pour {algorithm} : {sorted(set(given) - expected)}")
    return params | given


def snapshot_configs(destination: str | Path) -> Path:
    """Archive les fichiers de configuration utilisés dans ``destination`` et la renvoie.

    Si l'archive existe déjà (reprise d'un lancement), elle doit être identique : une
    configuration modifiée en cours de route lève ``FileExistsError``.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        content = (config_dir() / name).read_bytes()
        target = destination / name
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"{target} existe avec un contenu différent de {config_dir() / name}")
            continue
        target.write_bytes(content)
    return destination
