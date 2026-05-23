"""Définition des scénarios expérimentaux (Section 5.4 du mémoire).

Chaque scénario est une configuration paramétrique de la simulation
permettant d'évaluer la robustesse et la performance des différentes
stratégies de dispatching.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Configuration d'un scénario expérimental."""

    name: str
    description: str
    truck_count: int = 5
    shovel_count: int = 2
    episode_minutes: float = 480.0
    breakdown_probability: float = 0.02
    reward_weights: tuple[float, float, float] = (1.0, 0.1, 0.05)
    seeds: list[int] = field(default_factory=lambda: list(range(42, 52)))
    total_timesteps: int = 50_000


# Scénarios de référence pour l'expérimentation
SCENARIOS: dict[str, Scenario] = {
    "nominal": Scenario(
        name="nominal",
        description="Scénario nominal : 5 camions, 2 pelles, conditions standards.",
        truck_count=5,
        shovel_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
    ),
    "high_load": Scenario(
        name="high_load",
        description="Charge élevée : 8 camions pour 2 pelles (surcharge).",
        truck_count=8,
        shovel_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
    ),
    "low_load": Scenario(
        name="low_load",
        description="Charge faible : 3 camions pour 2 pelles (sous-utilisation).",
        truck_count=3,
        shovel_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.02,
    ),
    "high_breakdown": Scenario(
        name="high_breakdown",
        description="Taux de pannes élevé : 10% par cycle (test de robustesse).",
        truck_count=5,
        shovel_count=2,
        episode_minutes=480.0,
        breakdown_probability=0.10,
    ),
    "single_shovel": Scenario(
        name="single_shovel",
        description="Une seule pelle : test de goulot d'étranglement.",
        truck_count=5,
        shovel_count=1,
        episode_minutes=480.0,
        breakdown_probability=0.02,
    ),
    "short_shift": Scenario(
        name="short_shift",
        description="Shift court : 4 heures au lieu de 8.",
        truck_count=5,
        shovel_count=2,
        episode_minutes=240.0,
        breakdown_probability=0.02,
    ),
}
