"""Composants partagés par les entraînements tabulaires (Q-Learning, SARSA).

Les deux algorithmes partagent l'initialisation de l'environnement et de la
Q-table ainsi que la sauvegarde des artefacts d'entraînement. Seule la règle
de mise à jour (Eq. 4.1 pour Q-Learning, Eq. 4.3 pour SARSA) diffère et reste
implémentée séparément dans train_q_learning.py / train_sarsa.py.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.rl.discretize import _discretize_obs


class TabularPolicy:
    """Politique gloutonne basée sur une Q-table tabulaire entraînée."""

    def __init__(
        self,
        q_table: dict[tuple[int, ...], np.ndarray],
        n_bins: int = 8,
        num_shovels: int = 3,
        num_dumps: int = 2,
    ) -> None:
        self.q_table = q_table
        self.n_bins = n_bins
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps

    @classmethod
    def load(cls, model_path: str | Path) -> TabularPolicy:
        """Charge une Q-table depuis un fichier pickle."""
        with Path(model_path).open("rb") as f:
            data = pickle.load(f)  # noqa: S301
        return cls(
            q_table=data["q_table"],
            n_bins=data["n_bins"],
            num_shovels=data.get("num_shovels", 3),
            num_dumps=data.get("num_dumps", 2),
        )

    def predict(self, observation: np.ndarray) -> int:
        """Retourne l'action gloutonne pour l'observation donnée."""
        state = _discretize_obs(observation, self.n_bins, self.num_shovels, self.num_dumps)
        if state in self.q_table:
            return int(np.argmax(self.q_table[state]))
        return 0


def init_tabular_training(
    *,
    seed: int,
    truck_count: int,
    shovel_count: int,
    dump_count: int,
    episode_minutes: float,
    breakdown_probability: float,
    reward_weights: tuple[float, ...],
    epsilon_start: float,
    epsilon_min: float,
    n_episodes: int,
) -> tuple[MineEnv, dict[tuple[int, ...], np.ndarray], np.random.Generator, float, float, int]:
    """Initialise l'environnement, la Q-table et le générateur aléatoire.

    Returns
    -------
    (env, q_table, rng, epsilon, epsilon_decay, n_actions)
    """
    env = MineEnv(
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
        episode_minutes=episode_minutes,
        seed=seed,
        breakdown_probability=breakdown_probability,
        reward_weights=reward_weights,
    )

    n_actions = env.action_space.n
    q_table: dict[tuple[int, ...], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    epsilon_decay = (epsilon_start - epsilon_min) / max(n_episodes, 1)

    return env, q_table, rng, epsilon_start, epsilon_decay, n_actions


def save_tabular_artifacts(
    out_path: Path,
    model_filename: str,
    model_label: str,
    q_table: dict[tuple[int, ...], np.ndarray],
    n_bins: int,
    shovel_count: int,
    dump_count: int,
    rewards_per_episode: list[float],
    epsilons_per_episode: list[float],
    td_errors_per_episode: list[float],
) -> None:
    """Sauvegarde la Q-table, les récompenses et les diagnostics d'entraînement."""
    model_path = out_path / model_filename
    with model_path.open("wb") as f:
        pickle.dump({
            "q_table": q_table,
            "n_bins": n_bins,
            "num_shovels": shovel_count,
            "num_dumps": dump_count,
        }, f)
    print(f"{model_label} sauvegardée : {model_path}")

    rewards_path = out_path / "training_rewards.csv"
    with rewards_path.open("w", encoding="utf-8") as f:
        f.write("episode,total_reward\n")
        for i, r in enumerate(rewards_per_episode):
            f.write(f"{i},{r:.4f}\n")
    print(f"Récompenses d'entraînement : {rewards_path}")

    diagnostics_path = out_path / "training_diagnostics.csv"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        f.write("episode,total_reward,epsilon,mean_abs_td_error\n")
        for i, (r, eps, td) in enumerate(zip(rewards_per_episode, epsilons_per_episode, td_errors_per_episode)):
            f.write(f"{i},{r:.4f},{eps:.6f},{td:.6f}\n")
    print(f"Diagnostics d'entraînement : {diagnostics_path}")