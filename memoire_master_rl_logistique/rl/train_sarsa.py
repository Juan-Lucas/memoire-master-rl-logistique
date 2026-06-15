"""Entraînement SARSA pour le dispatching minier.

Conforme au Chapitre 4, Section 4.4.3 du mémoire :
- Algorithme : SARSA (on-policy, tabulaire)
- Hyperparamètres : Tableau 4.5 du mémoire (Section 4.6.2)
- Politique : ε-greedy avec décroissance linéaire

Contrairement au Q-Learning (off-policy), SARSA met à jour la Q-table
en utilisant l'action effectivement choisie par la politique ε-greedy
(Eq. 4.3 du mémoire).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from memoire_master_rl_logistique.rl.discretize import _discretize_obs
from memoire_master_rl_logistique.rl.tabular_training import (
    TabularPolicy,
    init_tabular_training,
    save_tabular_artifacts,
)


def train_sarsa(
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
    output_dir: str = "models/sarsa",
) -> dict[tuple[int, ...], np.ndarray]:
    """Entraîne un agent SARSA tabulaire sur l'environnement minier.

    Returns
    -------
    q_table : dict mapping discretized state -> Q-values array
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env, q_table, rng, epsilon, epsilon_decay, n_actions = init_tabular_training(
        seed=seed,
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
        episode_minutes=episode_minutes,
        breakdown_probability=breakdown_probability,
        reward_weights=reward_weights,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        n_episodes=n_episodes,
    )

    rewards_per_episode: list[float] = []
    epsilons_per_episode: list[float] = []
    td_errors_per_episode: list[float] = []

    def _select_action(state: tuple[int, ...], eps: float) -> int:
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)
        if rng.random() < eps:
            return int(rng.integers(n_actions))
        return int(np.argmax(q_table[state]))

    print(f"Début de l'entraînement SARSA ({n_episodes} épisodes)...")

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep)
        state = _discretize_obs(obs, n_bins, shovel_count, dump_count)
        action = _select_action(state, epsilon)
        total_reward = 0.0
        td_errors: list[float] = []

        while True:
            obs_next, reward, terminated, truncated, _info = env.step(action)
            next_state = _discretize_obs(obs_next, n_bins, shovel_count, dump_count)
            total_reward += reward

            # SARSA: choose next action BEFORE update (on-policy)
            next_action = _select_action(next_state, epsilon)

            # SARSA update (on-policy): Eq. 4.3
            if state not in q_table:
                q_table[state] = np.zeros(n_actions)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            td_target = reward + gamma * q_table[next_state][next_action]
            td_error = td_target - q_table[state][action]
            td_errors.append(abs(td_error))
            q_table[state][action] += alpha * td_error

            state = next_state
            action = next_action

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

    save_tabular_artifacts(
        out_path, "sarsa_table.pkl", "SARSA table",
        q_table, n_bins, shovel_count, dump_count,
        rewards_per_episode, epsilons_per_episode, td_errors_per_episode,
    )

    return q_table


class SarsaPolicy(TabularPolicy):
    """Politique gloutonne basée sur une table SARSA entraînée."""


def main() -> None:
    """Point d'entrée pour l'entraînement SARSA."""
    train_sarsa(
        n_episodes=30_000,
        alpha=0.2,
        gamma=0.99,
        seed=42,
    )


if __name__ == "__main__":
    main()
