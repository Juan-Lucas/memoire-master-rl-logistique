"""Scénarios expérimentaux (régimes, ablation, scénarios non vus).

Décrits dans ``configs/experiment_config.yaml`` depuis T28. Valeurs inspirées de la
littérature :
- 12 camions Cat 785C (Kangwa 2021, Kansanshi) ; le carburant est calé sur le Cat 785D,
  de consommation quasi identique (Caterpillar Performance Handbook, éd. 49 :
  voir ``simulation/constants.py``)
- 3 pelles Hitachi 2500 (Mohtasham 2023)
- 2 verses, poste de 8 heures
- 30 réplications par scénario (graines d'évaluation 42 à 71, ``seeds.EVAL_SEEDS``).
  Depuis T26, aucune n'est utilisée pour entraîner (voir ``seeds.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoire_master_rl_logistique.config import experiment_config
from memoire_master_rl_logistique.seeds import EVAL_SEEDS


@dataclass
class Scenario:
    """Configuration d'un scénario expérimental.

    Pas de valeur par défaut (sauf les graines) : toutes viennent de
    ``configs/experiment_config.yaml`` (``scenario_defaults`` puis le scénario, T28).
    """

    name: str
    description: str
    truck_count: int
    shovel_count: int
    dump_count: int
    episode_minutes: float
    breakdown_probability: float
    reward_weights: tuple[float, ...]
    total_timesteps: int
    # Agents entraînés et évalués dans ce scénario (les heuristiques le sont toujours) :
    # les quatre, sauf pour l'ablation sur w4 (PPO et DQN, décision du 22 octobre 2026).
    algorithms: tuple[str, ...]
    seeds: list[int] = field(default_factory=lambda: list(EVAL_SEEDS))  # 30 réplications d'évaluation


ALGORITHMS = ("Q-Learning", "SARSA", "DQN", "PPO")


def _build_scenarios(config: dict) -> dict[str, Scenario]:
    """Un ``Scenario`` par entrée de ``scenarios`` : valeurs communes, puis ce qui change."""
    scenarios = {}
    for name, overrides in config["scenarios"].items():
        values = {**config["scenario_defaults"], **overrides}
        values["reward_weights"] = tuple(values["reward_weights"])
        # Clé absente (configuration archivée avant le 22 octobre 2026) : les quatre.
        values["algorithms"] = tuple(values.get("algorithms", ALGORITHMS))
        unknown = set(values["algorithms"]) - set(ALGORITHMS)
        if unknown or not values["algorithms"]:
            raise ValueError(f"Scénario {name!r} : algorithmes {list(values['algorithms'])} "
                             f"(attendus : une partie non vide de {ALGORITHMS})")
        try:
            scenarios[name] = Scenario(name=name, **values)
        except TypeError as error:  # clé inconnue (faute de frappe dans le YAML)
            raise ValueError(f"Scénario {name!r} mal décrit dans la configuration : {error}") from None
    return scenarios


_CONFIG = experiment_config()

# Régimes lus dans configs/experiment_config.yaml (T28) : régimes principaux,
# configuration asymétrique, ablation sur w4, scénarios non vus.
SCENARIOS: dict[str, Scenario] = _build_scenarios(_CONFIG)

# Les trois régimes comparés dans l'article (les autres entrées servent aux
# expériences annexes : ablation, généralisation). Le premier est la référence.
MAIN_SCENARIOS: list[str] = list(_CONFIG["main_scenarios"])
PERTURBED_SCENARIOS: list[str] = MAIN_SCENARIOS[1:]

# Mode d'évaluation d'un agent (T27, colonne ``mode`` du CSV) : entraîné dans le régime
# évalué (spécialisé) ou entraîné en nominal puis évalué tel quel (transféré).
SPECIALISED = "spécialisé"
TRANSFERRED = "transféré"
TRANSFER_SOURCE = MAIN_SCENARIOS[0]


def training_regimes(scenario_name: str) -> list[str]:
    """Régimes d'entraînement des agents évalués sur ``scenario_name`` (T27).

    - régime perturbé principal : ses agents spécialisés **et** les agents nominaux
      transférés, pour mesurer à la fois l'adaptation et la robustesse ;
    - scénario non vu (``unseen_*``) : seulement les agents nominaux, par construction ;
    - autres (nominal, ablations) : seulement leurs propres agents.
    """
    if scenario_name.startswith("unseen_"):
        return [TRANSFER_SOURCE]
    if scenario_name in PERTURBED_SCENARIOS:
        return [scenario_name, TRANSFER_SOURCE]
    return [scenario_name]


def evaluation_mode(scenario_name: str, training_regime: str) -> str:
    """« spécialisé » si l'agent est évalué dans son régime d'entraînement, sinon « transféré »."""
    return SPECIALISED if training_regime == scenario_name else TRANSFERRED


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
