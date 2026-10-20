"""Entraînement SARSA tabulaire (on-policy, ε-greedy à décroissance linéaire).

Contrairement à Q-Learning, la cible utilise l'action suivante réellement
choisie par la politique ε-greedy, et non le maximum sur les actions.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from memoire_master_rl_logistique.config import hyperparameters
from memoire_master_rl_logistique.rl.checkpoints import (
    DEFAULT_CHECKPOINT_EVERY,
    clear_checkpoint,
    prepare_checkpoint,
    training_signature,
)
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
    save_tabular_progress,
    start_or_resume_tabular,
)
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, training_episode_seed


def train_sarsa(
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
    output_dir: str = str(model_dir(RESULTS_DIR, "SARSA", "nominal", TRAIN_SEEDS[0])),
    overwrite: bool = False,
    checkpoint_every: int | None = DEFAULT_CHECKPOINT_EVERY,
    resume: bool = False,
) -> dict[tuple[int, ...], np.ndarray]:
    """Entraîne un agent SARSA tabulaire sur l'environnement minier.

    Returns
    -------
    q_table : dict mapping discretized state -> Q-values array
    """
    # Hyperparamètres de configs/tabular_hyperparameters.yaml ; un argument donné les remplace (T28).
    params = hyperparameters(
        "SARSA", n_episodes=n_episodes, alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min, n_bins=n_bins,
    )
    n_episodes, alpha, gamma = params["n_episodes"], params["alpha"], params["gamma"]
    epsilon_start, epsilon_min, n_bins = (
        params["epsilon_start"], params["epsilon_min"], params["n_bins"])

    out_path = Path(output_dir)
    ensure_new_model(out_path / "sarsa_table.pkl", overwrite)
    folder = prepare_checkpoint(out_path, resume, overwrite)
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
    env_config = EnvConfig(truck_count, shovel_count, dump_count, episode_minutes,
                           breakdown_probability, reward_weights)
    signature = training_signature(algorithm="SARSA", seed=seed, params=params,
                                   env=asdict(env_config))
    # Sauvegarde tous les ``checkpoint_every`` épisodes ; ``resume`` reprend exactement
    # là où l'entraînement s'était arrêté (rl/checkpoints.py).
    progress = start_or_resume_tabular(out_path, resume, signature, q_table, rng, epsilon)
    q_table, epsilon = progress.q_table, progress.epsilon

    def _select_action(state: tuple[int, ...], eps: float) -> int:
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)
        if rng.random() < eps:
            return int(rng.integers(n_actions))
        return int(np.argmax(q_table[state]))

    print(f"Début de l'entraînement SARSA ({n_episodes} épisodes)...")

    for ep in range(progress.next_episode, n_episodes):
        obs, _info = env.reset(seed=training_episode_seed(seed, ep))  # disjointes de l'évaluation (T26)
        state = _discretize_obs(obs, n_bins, shovel_count, dump_count)
        action = _select_action(state, epsilon)
        total_reward = 0.0
        td_errors: list[float] = []

        while True:
            obs_next, reward, terminated, truncated, _info = env.step(action)
            next_state = _discretize_obs(obs_next, n_bins, shovel_count, dump_count)
            total_reward += reward

            # On choisit l'action suivante AVANT la mise à jour (on-policy).
            next_action = _select_action(next_state, epsilon)

            # Mise à jour on-policy : cible r + γ · Q(s', a')
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

        progress.record(total_reward, epsilon, float(np.mean(td_errors)),
                        float(_info["total_tonnage_t"]))
        epsilon = progress.epsilon = max(epsilon_min, epsilon - epsilon_decay)
        if checkpoint_every and progress.next_episode % checkpoint_every == 0 \
                and progress.next_episode < n_episodes:
            save_tabular_progress(folder, progress, rng, signature,
                                  progress.elapsed_before_s + time.perf_counter() - start)

        if (ep + 1) % 1000 == 0 or ep == 0:
            avg_reward = np.mean(progress.rewards[-100:])
            print(
                f"  Épisode {ep + 1}/{n_episodes} — "
                f"récompense moy. (100 derniers) = {avg_reward:.2f}, "
                f"ε = {epsilon:.4f}, "
                f"|Q-table| = {len(q_table)}"
            )

    save_tabular_artifacts(
        out_path, "sarsa_table.pkl", "SARSA table",
        q_table, n_bins, shovel_count, dump_count,
        progress.rewards, progress.epsilons, progress.td_errors, progress.tonnages,
    )
    write_metadata(out_path, build_metadata(
        "SARSA",
        env_config,
        observation_size=env.observation_space.shape[0],
        training_time_s=progress.elapsed_before_s + time.perf_counter() - start,
        seed=seed,
        training={**params, **({"resumed_from_episodes": progress.resumed_from_episodes}
                              if progress.resumed_from_episodes else {})},
    ))
    clear_checkpoint(out_path)

    return q_table


class SarsaPolicy(TabularPolicy):
    """Politique gloutonne basée sur une table SARSA entraînée."""


def main() -> None:
    """Point d'entrée pour l'entraînement SARSA."""
    # Régime nominal ; hyperparamètres de configs/ (T28).
    train_sarsa(seed=TRAIN_SEEDS[0])


if __name__ == "__main__":
    main()
