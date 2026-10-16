"""Entraînement de l'agent DQN (Stable-Baselines3, MLP [128, 128], ReLU).

DQN remplace la Q-table par un réseau qui approxime Q(s, a), avec tampon de
rejeu et réseau cible.
"""

from __future__ import annotations

from pathlib import Path
import time

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from memoire_master_rl_logistique.config import hyperparameters
from memoire_master_rl_logistique.rl.callbacks import KPILoggerCallback
from memoire_master_rl_logistique.rl.env_factory import (
    make_training_vec_env,
    reseed_training_vec_env,
)
from memoire_master_rl_logistique.rl.model_metadata import (
    EnvConfig,
    build_metadata,
    write_metadata,
)
from memoire_master_rl_logistique.rl.model_paths import RESULTS_DIR, ensure_new_model, model_dir
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS


def train_dqn(
    total_timesteps: int = 2_000_000,
    seed: int = TRAIN_SEEDS[0],
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    output_dir: str = str(model_dir(RESULTS_DIR, "DQN", "nominal", TRAIN_SEEDS[0])),
    log_freq: int = 1000,
    overwrite: bool = False,
    learning_rate: float | None = None,
    exploration_fraction: float | None = None,
    progress_bar: bool = True,
) -> DQN:
    """Entraîne un agent DQN et l'enregistre dans ``output_dir``.

    ε décroît de 1,0 à 0,02 sur les ``exploration_fraction`` premiers pas (30 % par défaut).
    Hyperparamètres lus dans ``configs/dqn_hyperparameters.yaml`` (T28) ;
    ``learning_rate`` et ``exploration_fraction``, s'ils sont donnés, remplacent ceux du
    fichier (balayage de T25 bis). Les valeurs utilisées sont écrites dans ``metadata.json``.
    """
    params = hyperparameters("DQN", learning_rate=learning_rate,
                             exploration_fraction=exploration_fraction)
    out_path = Path(output_dir)
    ensure_new_model(out_path / "dqn_mine_agent", overwrite)
    out_path.mkdir(parents=True, exist_ok=True)

    env = make_training_vec_env(
        seed,
        truck_count=truck_count, shovel_count=shovel_count, dump_count=dump_count,
        episode_minutes=episode_minutes, breakdown_probability=breakdown_probability,
        reward_weights=reward_weights,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": params["net_arch"]},
        **{k: v for k, v in params.items() if k != "net_arch"},
        verbose=1,
        seed=seed,
        tensorboard_log=None,  # métriques écrites en CSV dans sb3_logs/
    )

    # Stable-Baselines3 ré-ensemence l'environnement avec la graine du modèle
    # (set_random_seed → env.seed(seed)) : sans cette ligne, le premier épisode
    # utiliserait la graine ``seed`` elle-même, et non seed × 10⁶ (T26, vu par test_seeds).
    reseed_training_vec_env(env, seed)

    kpi_callback = KPILoggerCallback(
        log_path=out_path / "training_kpis.csv",
        log_freq=log_freq,
        verbose=1,
    )

    new_logger = configure(str(out_path / "sb3_logs"), ["csv"])
    model.set_logger(new_logger)

    print(f"Début de l'entraînement DQN ({total_timesteps} steps)...")
    start = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=kpi_callback,
        progress_bar=progress_bar,
    )

    model_path = out_path / "dqn_mine_agent"
    model.save(str(model_path))
    write_metadata(out_path, build_metadata(
        "DQN",
        EnvConfig(truck_count, shovel_count, dump_count, episode_minutes,
                  breakdown_probability, reward_weights),
        observation_size=env.observation_space.shape[0],
        training_time_s=time.perf_counter() - start,
        seed=seed,
        training={"total_timesteps": total_timesteps, **params},
    ))
    print(f"Modèle DQN sauvegardé : {model_path}")

    env.close()
    return model


def main() -> None:
    """Point d'entrée pour l'entraînement DQN."""
    # Régime nominal ; hyperparamètres de configs/ (T28).
    train_dqn(seed=TRAIN_SEEDS[0])


if __name__ == "__main__":
    main()
