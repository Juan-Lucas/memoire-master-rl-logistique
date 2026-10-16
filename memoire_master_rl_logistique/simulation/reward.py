"""Récompense d'une décision de dispatching (T25 : module séparé, §5.5 de l'article).

R = w1 · (production + opportunité) + w2 · équilibre + w3 · distance + w4 · attente,
avec cinq termes sans unité, chacun normalisé pour être de l'ordre de 1 :

- **production** = tonnage livré / capacité du camion. Vaut 1 pour un cycle terminé
  avant la fin du poste (140 t / 140 t), 0 sinon et pour ATTENDRE ;
- **équilibre** = − variance des attentes aux pelles (min²) / 100, mesurée au moment de
  la décision : pénalise des files très inégales entre pelles ;
- **distance** = − distance du cycle décidé (km) / 5 : trajet vers la pelle + pelle →
  verse + retour. 0 pour ATTENDRE ;
- **attente** = − min(attente / 30 min, 1) : file à la pelle + file à la verse pour un
  cycle, durée de l'ATTENDRE sinon. Plafonnée à −1 à partir de 30 min ;
- **opportunité** (v2) = − durée de l'ATTENDRE / ``OPPORTUNITY_CYCLE_MIN`` : la production
  perdue par un camion immobile, en cycles. Une minute d'attente coûte environ 1/24 de
  cycle. 0 pour un cycle. Ce terme est pondéré par w1, le poids de la production, car il
  est dans la même unité : il n'ajoute aucun poids à régler.

Poids par défaut (``Scenario.reward_weights``) : w1 = 1, w2 = 0,1, w3 = 0,05, w4 = 0,3.
La production domine : la récompense cumulée d'un poste est proche du nombre de cycles
terminés, diminuée surtout par la distance.

Ce module est un déplacement du calcul de ``MineEnv._compute_reward`` (T25). Les quatre
premiers termes sont identiques à ceux de la v1 (test de non-régression).

Pourquoi le terme d'opportunité (T19, décision du 16 octobre 2026) : dans la v1, ATTENDRE
ne coûtait presque rien (w4 × 0,2 / 30 ≈ 0,002 pour 0,2 min, sans coût de distance), et
PPO choisissait ATTENDRE dans 87 % de ses décisions en forte charge. Avec ce terme,
attendre 0,2 min coûte 0,0083 de plus, et attendre la durée d'un cycle coûte un cycle,
ce qu'un exploitant perd réellement. Les politiques qui n'attendent jamais (les cinq
heuristiques) ont exactement la même récompense qu'en v1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

VARIANCE_SCALE_MIN2 = 100.0
DISTANCE_SCALE_KM = 5.0
WAIT_SCALE_MIN = 30.0
# Durée de référence d'un cycle, en minutes : celle de l'Affectation fixe en nominal
# (24,1 min, T22). Vérifiée par tests/test_reward.py.
OPPORTUNITY_CYCLE_MIN = 24.0


@dataclass(frozen=True)
class RewardWeights:
    """Poids w1 (production), w2 (équilibre), w3 (distance), w4 (attente)."""

    production: float = 1.0
    balance: float = 0.1
    distance: float = 0.05
    wait: float = 0.3

    @property
    def opportunity(self) -> float:
        """Poids du coût d'opportunité : celui de la production (même unité, des cycles)."""
        return self.production

    @classmethod
    def from_tuple(cls, weights: Sequence[float]) -> RewardWeights:
        """(w1, w2, w3[, w4]) ; w4 = 0,3 s'il manque, comme dans la v1."""
        w4 = weights[3] if len(weights) > 3 else 0.3
        return cls(float(weights[0]), float(weights[1]), float(weights[2]), float(w4))


@dataclass(frozen=True)
class RewardTerms:
    """Les cinq termes, avant pondération."""

    production: float
    balance: float
    distance: float
    wait: float
    opportunity: float = 0.0

    def weighted(self, weights: RewardWeights) -> dict[str, float]:
        """Contribution de chaque terme à la récompense : poids × terme."""
        return {name: getattr(weights, name) * value for name, value in asdict(self).items()}


def reward_terms(
    delivered_tonnage_t: float,
    capacity_t: float,
    route_distance_km: float,
    shovel_waits_min: Sequence[float],
    operational_wait_min: float,
    idle_min: float = 0.0,
) -> RewardTerms:
    """Les termes d'une décision (voir la docstring du module).

    ``idle_min`` : durée d'une ATTENDRE comptée dans le poste (0 pour un cycle).
    """
    production = delivered_tonnage_t / max(capacity_t, 1e-9)

    if len(shovel_waits_min) >= 2:
        mean_wait = sum(shovel_waits_min) / len(shovel_waits_min)
        variance = sum((w - mean_wait) ** 2 for w in shovel_waits_min) / len(shovel_waits_min)
        balance = -variance / VARIANCE_SCALE_MIN2
    else:
        balance = 0.0

    distance = -route_distance_km / DISTANCE_SCALE_KM
    wait = -min(operational_wait_min / WAIT_SCALE_MIN, 1.0)
    opportunity = -idle_min / OPPORTUNITY_CYCLE_MIN
    return RewardTerms(production, balance, distance, wait, opportunity)


def compute_reward(terms: RewardTerms, weights: RewardWeights) -> float:
    """R = w1 · (production + opportunité) + w2 · équilibre + w3 · distance + w4 · attente."""
    return (
        weights.production * terms.production
        + weights.balance * terms.balance
        + weights.distance * terms.distance
        + weights.wait * terms.wait
        + weights.opportunity * terms.opportunity
    )
