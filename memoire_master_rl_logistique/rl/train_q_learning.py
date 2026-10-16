"""Entraînement Q-Learning tabulaire (off-policy, ε-greedy à décroissance linéaire).

L'observation est réduite à 6 variables discrètes (voir ``rl/discretize.py``) :
les 5 attentes aux pelles et aux verses, et la pelle la plus proche.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from memoire_master_rl_logistique.config import hyperparameters
from memoire_master_rl_logistique.rl.discretize import _discretize_obs
from memoire_master_rl_logistique.rl.model_metadata import (
    EnvConfig,
    build_metadata,
    write_metadata,
)
from memoire_master_rl_logistique.rl.model_paths import RESULTS_DIR, ensure_new_model, model_dir
from memoire_master_rl_logistique.rl.tabular_training import (
    TabularPolicy,
    init_tabular_training,
    save_tabular_artifacts,
)
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, training_episode_seed


def train_q_learning(
    n_episodes: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
    epsilon_start: float | None = None,
    epsilon_min: float | None = None,
    n_bins: int | None = None,
    seed: int = TRAIN_SEEDS[0],
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    output_dir: str = str(model_dir(RESULTS_DIR, "Q-Learning", "nominal", TRAIN_SEEDS[0])),
    overwrite: bool = False,
) -> dict[tuple[int, ...], np.ndarray]:
    """Entraîne un agent Q-Learning tabulaire sur l'environnement minier.

    Returns
    -------
    q_table : dict mapping discretized state -> Q-values array
    """
    # Hyperparamètres de configs/tabular_hyperparameters.yaml ; un argument donné les remplace (T28).
    params = hyperparameters(
        "Q-Learning", n_episodes=n_episodes, alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min, n_bins=n_bins,
    )
    n_episodes, alpha, gamma = params["n_episodes"], params["alpha"], params["gamma"]
    epsilon_start, epsilon_min, n_bins = (
        params["epsilon_start"], params["epsilon_min"], params["n_bins"])

    out_path = Path(output_dir)
    ensure_new_model(out_path / "q_table.pkl", overwrite)
    out_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
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

    print(f"Début de l'entraînement Q-Learning ({n_episodes} épisodes)...")

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=training_episode_seed(seed, ep))  # disjointes de l'évaluation (T26)
        state = _discretize_obs(obs, n_bins, shovel_count, dump_count)
        total_reward = 0.0
        td_errors: list[float] = []

        while True:
            if state not in q_table:
                q_table[state] = np.zeros(n_actions)

            # Choix ε-greedy
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q_table[state]))

            obs_next, reward, terminated, truncated, _info = env.step(action)
            next_state = _discretize_obs(obs_next, n_bins, shovel_count, dump_count)
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            # Mise à jour off-policy : cible r + γ · max_a' Q(s', a')
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

    save_tabular_artifacts(
        out_path, "q_table.pkl", "Q-table",
        q_table, n_bins, shovel_count, dump_count,
        rewards_per_episode, epsilons_per_episode, td_errors_per_episode,
    )
    write_metadata(out_path, build_metadata(
        "Q-Learning",
        EnvConfig(truck_count, shovel_count, dump_count, episode_minutes,
                  breakdown_probability, reward_weights),
        observation_size=env.observation_space.shape[0],
        training_time_s=time.perf_counter() - start,
        seed=seed,
        training=params,
    ))

    return q_table


class QLearningPolicy(TabularPolicy):
    """Politique gloutonne basée sur une Q-table entraînée (Q-Learning)."""


def main() -> None:
    """Point d'entrée pour l'entraînement Q-Learning."""
    # Régime nominal ; hyperparamètres de configs/ (T28).
    train_q_learning(seed=TRAIN_SEEDS[0])


if __name__ == "__main__":
    main()
