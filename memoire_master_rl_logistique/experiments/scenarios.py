"""Scénarios expérimentaux (régimes, ablation, scénarios non vus).

Valeurs par défaut inspirées de la littérature :
- 12 camions Cat 785C (Kangwa 2021, Kansanshi)
- 3 pelles Hitachi 2500 (Mohtasham 2023)
- 2 verses, poste de 8 heures
- 30 réplications par scénario (graines 42 à 71). Attention : ``seeds[0]``
  sert aussi de graine d'entraînement dans ``run_benchmark``.
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
    seeds: list[int] = field(default_factory=lambda: list(range(42, 72)))  # 30 réplications
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
        total_timesteps=2_000_000,
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
}


# ---------------------------------------------------------------------------
# Ablation sur w4 (poids de la pénalité d'attente), régime nominal sinon.
# w4 = 0,3 correspond au scénario "nominal" lui-même.
# ---------------------------------------------------------------------------
_ABLATION_W4 = [0.0, 0.1, 0.5]  # w4=0.3 déjà dans nominal
for _w4 in _ABLATION_W4:
    _key = f"ablation_w4_{str(_w4).replace('.', '')}"
    SCENARIOS[_key] = Scenario(
        name=_key,
        description=f"Ablation reward : w4={_w4} (nominal sinon).",
        reward_weights=(1.0, 0.1, 0.05, _w4),
    )

# ---------------------------------------------------------------------------
# Scénarios non vus, évalués avec les modèles nominaux (voir run_all_benchmarks).
# unseen_medium_load (15 camions) est retiré depuis T9 : la taille de l'observation
# dépend du nombre de camions (147 entrées à 12 camions, 180 à 15), donc les modèles
# nominaux ne peuvent pas le lire. À réintroduire avec T18 (observation de taille fixe) :
#   Scenario(name="unseen_medium_load", truck_count=15,
#            description="Non vu : 15 camions, pb=2 % (entre nominal et high_load).")
# ---------------------------------------------------------------------------
SCENARIOS["unseen_medium_breakdown"] = Scenario(
    name="unseen_medium_breakdown",
    description="Non vu : 12 camions, pb=5 % (entre nominal et high_breakdown).",
    breakdown_probability=0.05,
)

# Les trois régimes comparés dans l'article (les autres entrées servent aux
# expériences annexes : ablation, généralisation).
MAIN_SCENARIOS: list[str] = ["nominal", "high_load", "high_breakdown"]
PERTURBED_SCENARIOS: list[str] = [s for s in MAIN_SCENARIOS if s != "nominal"]


@dataclass(frozen=True)
class ScenarioDisplay:
    """Métadonnées d'affichage pour les scripts de génération de figures.

    key : clé canonique (= clé de ``SCENARIOS`` / nom de répertoire, ex. "high_load").
    label : libellé d'affichage utilisé dans les CSV Power BI (ex. "High Load").
    suffix : suffixe de nom de fichier de figure, sans underscore (ex. "highload").
    """

    key: str
    label: str
    suffix: str


SCENARIO_DISPLAYS: tuple[ScenarioDisplay, ...] = (
    ScenarioDisplay("nominal", "Nominal", "nominal"),
    ScenarioDisplay("high_load", "High Load", "highload"),
    ScenarioDisplay("high_breakdown", "High Breakdown", "highbreakdown"),
)
