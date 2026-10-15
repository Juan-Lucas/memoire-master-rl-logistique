"""Entraînement de l'agent DQN (Stable-Baselines3, MLP [128, 128], ReLU).

DQN remplace la Q-table par un réseau qui approxime Q(s, a), avec tampon de
rejeu et réseau cible.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from memoire_master_rl_logistique.rl.callbacks import KPILoggerCallback
from memoire_master_rl_logistique.rl.env_factory import create_env
from memoire_master_rl_logistique.rl.model_metadata import (
    EnvConfig,
    build_metadata,
    write_metadata,
)
from memoire_master_rl_logistique.rl.model_paths import RESULTS_DIR, ensure_new_model, model_dir


def train_dqn(
    total_timesteps: int = 2_000_000,
    seed: int = 42,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    output_dir: str = str(model_dir(RESULTS_DIR, "DQN", "nominal")),
    log_freq: int = 1000,
    overwrite: bool = False,
) -> DQN:
    """Entraîne un agent DQN et l'enregistre dans ``output_dir``.

    ε décroît de 1,0 à 0,02 sur les 30 % premiers pas d'entraînement.
    """
    out_path = Path(output_dir)
    ensure_new_model(out_path / "dqn_mine_agent", overwrite)
    out_path.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        lambda: create_env(
            truck_count=truck_count,
            shovel_count=shovel_count,
            dump_count=dump_count,
            episode_minutes=episode_minutes,
            seed=seed,
            breakdown_probability=breakdown_probability,
            reward_weights=reward_weights,
        ),
        n_envs=1,
        seed=seed,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=200_000,
        learning_starts=1000,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        seed=seed,
        tensorboard_log=None,  # métriques écrites en CSV dans sb3_logs/
    )

    kpi_callback = KPILoggerCallback(
        log_path=out_path / "training_kpis.csv",
        log_freq=log_freq,
        verbose=1,
    )

    new_logger = configure(str(out_path / "sb3_logs"), ["csv"])
    model.set_logger(new_logger)

    print(f"Début de l'entraînement DQN ({total_timesteps} steps)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=kpi_callback,
        progress_bar=True,
    )

    model_path = out_path / "dqn_mine_agent"
    model.save(str(model_path))
    write_metadata(out_path, build_metadata(
        "DQN",
        EnvConfig(truck_count, shovel_count, dump_count, episode_minutes,
                  breakdown_probability, reward_weights),
        observation_size=env.observation_space.shape[0],
        seed=seed,
        training={"total_timesteps": total_timesteps},
    ))
    print(f"Modèle DQN sauvegardé : {model_path}")

    env.close()
    return model


def main() -> None:
    """Point d'entrée pour l'entraînement DQN."""
    train_dqn(
        total_timesteps=2_000_000,
        seed=42,
        truck_count=12,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
    )


if __name__ == "__main__":
    main()
