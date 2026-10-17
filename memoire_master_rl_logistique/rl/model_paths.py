"""Où sont rangés les modèles entraînés (T9, point N8).

Un seul endroit décide du dossier d'un modèle : ``<results_dir>/<préfixe>_<scénario>``.
L'entraînement l'utilise pour écrire, le benchmark pour relire. Avant T9, le
pipeline de ``main.py`` entraînait PPO dans ``models/ppo_mine``, un dossier que le
benchmark ne relisait jamais.

Module volontairement sans dépendance lourde : les fonctions d'entraînement
l'importent pour leur dossier par défaut.
"""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path("data/results")

MODEL_DIR_PREFIX = {
    "Q-Learning": "q_learning",
    "SARSA": "sarsa",
    "DQN": "dqn",
    "PPO": "ppo",
}


def ensure_new_model(model_path: str | Path, overwrite: bool) -> None:
    """Refuse d'écraser un modèle existant (un entraînement coûte jusqu'à 2 h) sauf si ``overwrite``.

    ``model_path`` peut être donné sans l'extension ``.zip`` qu'ajoute Stable-Baselines3.
    """
    path = Path(model_path)
    candidates = [path, path.with_name(path.name + ".zip")]
    existing = [p for p in candidates if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Modèle déjà présent : {existing[0]} (passer overwrite=True pour le remplacer)"
        )


def model_dir(
    results_dir: str | Path, algorithm: str, scenario_name: str, train_seed: int | None = None,
) -> Path:
    """Dossier du modèle ``algorithm`` entraîné sur ``scenario_name`` avec ``train_seed`` (T26).

    Exemples : ``model_dir("data/results", "PPO", "nominal", 3)`` →
    ``data/results/ppo_nominal_seed3`` ; sans graine (modèles d'avant T26, dont ceux de
    l'article v1) → ``data/results/ppo_nominal``.
    """
    if algorithm not in MODEL_DIR_PREFIX:
        raise ValueError(f"Algorithme inconnu : {algorithm!r} (attendu : {tuple(MODEL_DIR_PREFIX)})")
    suffix = "" if train_seed is None else f"_seed{train_seed}"
    return Path(results_dir) / f"{MODEL_DIR_PREFIX[algorithm]}_{scenario_name}{suffix}"
