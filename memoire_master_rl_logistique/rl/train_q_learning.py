"""Entraînement Q-Learning tabulaire pour le dispatching minier.

Conforme au Chapitre 4, Section 4.4.1 du mémoire :
- Algorithme : Q-Learning (off-policy, tabulaire)
- Hyperparamètres : Tableau 4.4 du mémoire (Section 4.6.1)
- Politique : ε-greedy avec décroissance linéaire

L'observation continue est discrétisée en bins pour permettre
l'utilisation d'une Q-table. Chaque dimension de l'observation
est divisée en ``n_bins`` intervalles égaux dans [0, 1].
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.rl.discretize import _discretize_obs


def train_q_learning(
    n_episodes: int = 10_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    n_bins: int = 8,
    seed: int = 42,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    output_dir: str = "models/q_learning",
) -> dict[tuple[int, ...], np.ndarray]:
    """Entraîne un agent Q-Learning tabulaire sur l'environnement minier.

    Returns
    -------
    q_table : dict mapping discretized state -> Q-values array
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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

    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_min) / max(n_episodes, 1)

    rewards_per_episode: list[float] = []
    epsilons_per_episode: list[float] = []
    td_errors_per_episode: list[float] = []

    print(f"Début de l'entraînement Q-Learning ({n_episodes} épisodes)...")

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep)
        state = _discretize_obs(obs, n_bins, shovel_count, dump_count)
        total_reward = 0.0
        td_errors: list[float] = []

        while True:
            if state not in q_table:
                q_table[state] = np.zeros(n_actions)

            # ε-greedy action selection (Eq. 4.1)
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q_table[state]))

            obs_next, reward, terminated, truncated, _info = env.step(action)
            next_state = _discretize_obs(obs_next, n_bins, shovel_count, dump_count)
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            # Q-Learning update (off-policy): Eq. 4.1
            best_next = float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            td_errors.append(abs(td_error))
            q_table[state][action] += alpha * td_error

            state = next_state

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)
        epsilons_per_episode.append(epsilon)
        td_errors_per_episode.append(float(np.mean(td_errors)))
        epsilon = max(epsilon_min, epsilon - epsilon_decay)

        if (ep + 1) % 1000 == 0 or ep == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(
                f"  Épisode {ep + 1}/{n_episodes} — "
                f"récompense moy. (100 derniers) = {avg_reward:.2f}, "
                f"ε = {epsilon:.4f}, "
                f"|Q-table| = {len(q_table)}"
            )

    # Sauvegarder
    model_path = out_path / "q_table.pkl"
    with model_path.open("wb") as f:
        pickle.dump({
            "q_table": q_table,
            "n_bins": n_bins,
            "num_shovels": shovel_count,
            "num_dumps": dump_count,
        }, f)
    print(f"Q-table sauvegardée : {model_path}")

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

    return q_table


class QLearningPolicy:
    """Politique gloutonne basée sur une Q-table entraînée."""

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
    def load(cls, model_path: str | Path) -> QLearningPolicy:
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
        from memoire_master_rl_logistique.rl.discretize import _discretize_obs
        state = _discretize_obs(observation, self.n_bins, self.num_shovels, self.num_dumps)
        if state in self.q_table:
            return int(np.argmax(self.q_table[state]))
        return 0


def main() -> None:
    """Point d'entrée pour l'entraînement Q-Learning."""
    train_q_learning(
        n_episodes=30_000,
        alpha=0.2,
        gamma=0.99,
        seed=42,
    )


if __name__ == "__main__":
    main()
