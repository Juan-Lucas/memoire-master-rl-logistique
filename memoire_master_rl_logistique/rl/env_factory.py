"""Fabrique d'environnement partagée par les entraînements Deep RL (DQN, PPO)."""

from __future__ import annotations

from memoire_master_rl_logistique.env.mine_env import MineEnv


def create_env(
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
    seed: int = 42,
    breakdown_probability: float = 0.02,
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3),
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
    )