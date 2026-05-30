"""Scénarios expérimentaux (Section 5.4 du mémoire).

Chaque scénario est une configuration paramétrique permettant
d'évaluer la robustesse et la performance des stratégies de
dispatching sur des conditions variées.

Valeurs par défaut basées sur la littérature :
- 12 camions Cat 785C (Kangwa 2021, Kansanshi)
- 3 pelles Hitachi 2500 (Mohtasham 2023)
- 2 dumps, shift de 8 heures
- 100 réplications par scénario
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Configuration d'un scénario expérimental."""

    name: str
    description: str
    truck_count: int = 12
    shovel_count: int = 3
    dump_count: int = 2
    episode_minutes: float = 480.0
    breakdown_probability: float = 0.02
    reward_weights: tuple[float, ...] = (1.0, 0.1, 0.05, 0.3)
    seeds: list[int] = field(default_factory=lambda: list(range(42, 52)))
    total_timesteps: int = 2_000_000


SCENARIOS: dict[str, Scenario] = {
    "nominal": Scenario(
        name="nominal",
        description="Conditions nominales : 12 camions, 3 pelles, 2 dumps.",
        truck_count=12,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,  # Augmenté à 2M pour améliorer PPO/DQN
    ),
    "nominal_asymétrique": Scenario(
        name="nominal_asymétrique",
        description="Configuration asymétrique : 13 camions, 3 pelles, 2 dumps (MF=4.33, désavantage Fixed Assignment).",
        truck_count=13,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,
    ),
    "high_load": Scenario(
        name="high_load",
        description="Charge élevée : 18 camions pour 3 pelles (surcharge).",
        truck_count=18,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,
    ),
    "low_load": Scenario(
        name="low_load",
        description="Charge faible : 6 camions pour 3 pelles.",
        truck_count=6,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,
    ),
    "high_breakdown": Scenario(
        name="high_breakdown",
        description="Taux de pannes élevé : 10% par cycle (robustesse).",
        truck_count=12,
        shovel_count=3,
        dump_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.10,
        total_timesteps=2_000_000,
    ),
    "single_shovel": Scenario(
        name="single_shovel",
        description="Une seule pelle : goulot d'étranglement.",
        truck_count=12,
        shovel_count=1,
        dump_count=1,
        episode_minutes=480.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,
    ),
    "short_shift": Scenario(
        name="short_shift",
        description="Shift court : 4 heures au lieu de 8.",
        truck_count=12,
        shovel_count=3,
        dump_count=2,
        episode_minutes=240.0,
        breakdown_probability=0.02,
        total_timesteps=2_000_000,
    ),
}
