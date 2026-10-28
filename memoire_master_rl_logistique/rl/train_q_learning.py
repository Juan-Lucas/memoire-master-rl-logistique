"""Entraînement Q-Learning tabulaire (off-policy, ε-greedy à décroissance linéaire).

L'observation est réduite à 6 variables discrètes (voir ``rl/discretize.py``) :
les 5 attentes aux pelles et aux verses, et la pelle la plus proche.
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
from memoire_master_rl_logistique.rl.time_discount import (
    per_minute_gamma,
    transition_discount,
)
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, training_episode_seed


def train_q_learning(
    n_episodes: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
    time_discount: bool | None = None,
    epsilon_start: float | None = None,
    epsilon_min: float | None = None,
    n_bins: int | None = None,
    state_feature: str | None = None,
    seed: int = TRAIN_SEEDS[0],
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    output_dir: str = str(model_dir(RESULTS_DIR, "Q-Learning", "nominal", TRAIN_SEEDS[0])),
    overwrite: bool = False,
    checkpoint_every: int | None = DEFAULT_CHECKPOINT_EVERY,
    resume: bool = False,
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
        time_discount=time_discount, state_feature=state_feature,
    )
    n_episodes, alpha, gamma = params["n_episodes"], params["alpha"], params["gamma"]
    # Actualisation par minute (rl/time_discount.py) : γ_min^Δt au lieu de γ dans la cible.
    gamma_min = per_minute_gamma(gamma) if params["time_discount"] else None
    epsilon_start, epsilon_min, n_bins = (
        params["epsilon_start"], params["epsilon_min"], params["n_bins"])
    state_feature = params["state_feature"]

    out_path = Path(output_dir)
    ensure_new_model(out_path / "q_table.pkl", overwrite)
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
    signature = training_signature(algorithm="Q-Learning", seed=seed, params=params,
                                   env=asdict(env_config))
    # Sauvegarde tous les ``checkpoint_every`` épisodes ; ``resume`` reprend exactement
    # là où l'entraînement s'était arrêté (rl/checkpoints.py).
    progress = start_or_resume_tabular(out_path, resume, signature, q_table, rng, epsilon)
    q_table, epsilon = progress.q_table, progress.epsilon

    print(f"Début de l'entraînement Q-Learning ({n_episodes} épisodes)...")

    for ep in range(progress.next_episode, n_episodes):
        obs, _info = env.reset(seed=training_episode_seed(seed, ep))  # disjointes de l'évaluation (T26)
        state = _discretize_obs(obs, n_bins, shovel_count, dump_count, state_feature)
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
            next_state = _discretize_obs(obs_next, n_bins, shovel_count, dump_count,
                                         state_feature)
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            # Mise à jour off-policy : cible r + γ · max_a' Q(s', a') (γ_min^Δt par minute)
            best_next = float(np.max(q_table[next_state]))
            td_target = reward + transition_discount(gamma, gamma_min, _info) * best_next
            td_error = td_target - q_table[state][action]
            td_errors.append(abs(td_error))
            q_table[state][action] += alpha * td_error

            state = next_state

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
        out_path, "q_table.pkl", "Q-table",
        q_table, n_bins, shovel_count, dump_count,
        progress.rewards, progress.epsilons, progress.td_errors, progress.tonnages,
        state_feature,
    )
    write_metadata(out_path, build_metadata(
        "Q-Learning",
        env_config,
        observation_size=env.observation_space.shape[0],
        training_time_s=progress.elapsed_before_s + time.perf_counter() - start,
        seed=seed,
        training={**params, **({"gamma_per_minute": gamma_min} if gamma_min else {}),
                  **({"resumed_from_episodes": progress.resumed_from_episodes}
                              if progress.resumed_from_episodes else {})},
    ))
    clear_checkpoint(out_path)

    return q_table


class QLearningPolicy(TabularPolicy):
    """Politique gloutonne basée sur une Q-table entraînée (Q-Learning)."""


def main() -> None:
    """Point d'entrée pour l'entraînement Q-Learning."""
    # Régime nominal ; hyperparamètres de configs/ (T28).
    train_q_learning(seed=TRAIN_SEEDS[0])


if __name__ == "__main__":
    main()
