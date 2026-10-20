"""Fabrique d'environnements partagée par les entraînements Deep RL (DQN, PPO)."""

from __future__ import annotations

from functools import partial
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.seeds import training_episode_seed


def create_env(
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    seed: int = 42,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
    seed_step: int = 1,
) -> MineEnv:
    """Crée une instance de l'environnement minier."""
    return MineEnv(
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
        episode_minutes=episode_minutes,
        seed=seed,
        breakdown_probability=breakdown_probability,
        reward_weights=reward_weights,
        seed_step=seed_step,
    )


def _monitored_env(**kwargs: Any) -> Monitor:
    # Monitor : récompense et longueur d'épisode dans les journaux SB3 (rollout/ep_rew_mean).
    return Monitor(create_env(**kwargs))


def make_training_vec_env(train_seed: int, n_envs: int = 1, **env_kwargs: Any) -> VecEnv:
    """Environnements d'entraînement de graine ``train_seed`` (T26), éventuellement parallèles (T29).

    Les épisodes sont entrelacés sans doublon : l'environnement i joue les graines
    base + i, base + i + n, base + i + 2n, … avec base = ``training_episode_seed(train_seed)``,
    toutes dans la plage de cette graine d'entraînement. Avec ``n_envs = 1``, c'est la suite
    base, base + 1, … d'avant T29. Au-delà d'un environnement, ``SubprocVecEnv`` les fait
    tourner dans des processus séparés.

    À rappeler après la création du modèle SB3, qui ré-ensemence les environnements avec
    sa propre graine : ``reseed_training_vec_env``.
    """
    base = training_episode_seed(train_seed)
    factories = [
        partial(_monitored_env, seed=base + i, seed_step=n_envs, **env_kwargs) for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories)
    reseed_training_vec_env(vec_env, train_seed)
    return vec_env


def reseed_training_vec_env(vec_env: VecEnv, train_seed: int, first_episode: int = 0) -> None:
    """Première remise à zéro de l'environnement i sur base + ``first_episode`` + i (SB3
    ajoute l'indice).

    ``first_episode`` > 0 sert à la reprise d'un entraînement interrompu (un seul
    environnement) : le premier épisode joué est le premier épisode non terminé.
    """
    vec_env.seed(training_episode_seed(train_seed, first_episode))
