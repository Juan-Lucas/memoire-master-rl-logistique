"""Composants partagés par les entraînements tabulaires (Q-Learning, SARSA).

Les deux algorithmes partagent l'initialisation de l'environnement et de la
Q-table ainsi que la sauvegarde des artefacts. Seule la règle de mise à jour
diffère (train_q_learning.py, train_sarsa.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.rl.checkpoints import (
    load_tabular_checkpoint,
    save_tabular_checkpoint,
)
from memoire_master_rl_logistique.rl.discretize import DEFAULT_N_BINS, _discretize_obs

# Action choisie pour un état jamais rencontré pendant l'entraînement : 0 = (pelle 1, verse 1).
# Sa fréquence est mesurée pendant l'évaluation (``TabularPolicy.coverage``).
DEFAULT_ACTION = 0


class TabularPolicy:
    """Politique gloutonne basée sur une Q-table tabulaire entraînée.

    Compte, depuis le dernier ``reset_coverage``, les décisions prises et celles où
    l'état était inconnu (action par défaut) : voir ``coverage``.
    """

    def __init__(
        self,
        q_table: dict[tuple[int, ...], np.ndarray],
        n_bins: int = DEFAULT_N_BINS,
        num_shovels: int = 3,
        num_dumps: int = 2,
    ) -> None:
        self.q_table = q_table
        self.n_bins = n_bins
        self.num_shovels = num_shovels
        self.num_dumps = num_dumps
        self.reset_coverage()

    def reset_coverage(self) -> None:
        """Remet à zéro le comptage des décisions (au début d'un épisode d'évaluation)."""
        self.decisions = 0
        self.unknown_states = 0

    def coverage(self) -> dict[str, float]:
        """Part des décisions prises sur un état déjà vu à l'entraînement, et sur un état
        inconnu (donc avec l'action par défaut), en %."""
        known = 100.0 * (self.decisions - self.unknown_states) / max(self.decisions, 1)
        return {"known_state_pct": known, "default_action_pct": 100.0 - known if self.decisions else 0.0}

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
        """Action gloutonne ; ``DEFAULT_ACTION`` si l'état n'a jamais été vu."""
        state = _discretize_obs(observation, self.n_bins, self.num_shovels, self.num_dumps)
        self.decisions += 1
        if state in self.q_table:
            return int(np.argmax(self.q_table[state]))
        self.unknown_states += 1
        return DEFAULT_ACTION


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


@dataclass
class TabularProgress:
    """Ce qui évolue pendant un entraînement tabulaire : tout ce qu'une sauvegarde doit
    contenir pour une reprise exacte (voir ``rl/checkpoints.py``)."""

    q_table: dict[tuple[int, ...], np.ndarray]
    epsilon: float
    next_episode: int = 0
    rewards: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    tonnages: list[float] = field(default_factory=list)  # productivité d'entraînement (T30)
    elapsed_before_s: float = 0.0
    resumed_from_episodes: list[int] = field(default_factory=list)

    def record(self, reward: float, epsilon: float, td_error: float, tonnage: float) -> None:
        """Journal de l'épisode qui vient de finir."""
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.td_errors.append(td_error)
        self.tonnages.append(tonnage)
        self.next_episode += 1


def start_or_resume_tabular(
    out_path: Path,
    resume: bool,
    signature: dict,
    q_table: dict[tuple[int, ...], np.ndarray],
    rng: np.random.Generator,
    epsilon: float,
) -> TabularProgress:
    """État de départ : neuf, ou celui de la dernière sauvegarde (``resume``).

    À la reprise, l'état du générateur ``rng`` est restauré sur place.
    """
    if not resume:
        return TabularProgress(q_table=q_table, epsilon=epsilon)
    saved = load_tabular_checkpoint(out_path, signature)
    rng.bit_generator.state = saved["rng_state"]
    progress = saved["progress"]
    progress.resumed_from_episodes.append(progress.next_episode)
    print(f"Reprise de l'entraînement à l'épisode {progress.next_episode + 1}...")
    return progress


def save_tabular_progress(
    folder: Path, progress: TabularProgress, rng: np.random.Generator, signature: dict,
    elapsed_s: float,
) -> None:
    """Sauvegarde entre deux épisodes (``elapsed_s`` : durée cumulée, reprises comprises)."""
    snapshot = TabularProgress(**{**progress.__dict__, "elapsed_before_s": elapsed_s})
    save_tabular_checkpoint(folder, {
        "next_episode": progress.next_episode,
        "progress": snapshot,
        "rng_state": rng.bit_generator.state,
        "signature": signature,
        "elapsed_s": elapsed_s,
    })


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
    tonnages_per_episode: list[float],
) -> None:
    """Sauvegarde la Q-table, les récompenses et les diagnostics d'entraînement.

    ``training_diagnostics.csv`` est la source des courbes d'apprentissage et de diagnostic
    (T30) : elles viennent donc du modèle évalué, pas d'un réentraînement séparé.
    """
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
        f.write("episode,total_reward,epsilon,mean_abs_td_error,tonnage_t\n")
        rows = zip(rewards_per_episode, epsilons_per_episode, td_errors_per_episode,
                   tonnages_per_episode, strict=True)
        for i, (r, eps, td, ton) in enumerate(rows):
            f.write(f"{i},{r:.4f},{eps:.6f},{td:.6f},{ton:.2f}\n")
    print(f"Diagnostics d'entraînement : {diagnostics_path}")