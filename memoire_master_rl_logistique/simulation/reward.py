"""Récompense d'une décision de dispatching (T25 : module séparé, §5.5 de l'article).

R = w1 · production + w2 · équilibre + w3 · distance + w4 · attente, avec quatre termes
sans unité, chacun normalisé pour être de l'ordre de 1 :

- **production** = tonnage livré / capacité du camion. Vaut 1 pour un cycle terminé
  avant la fin du poste (140 t / 140 t), 0 sinon et pour ATTENDRE ;
- **équilibre** = − variance des attentes aux pelles (min²) / 100, mesurée au moment de
  la décision : pénalise des files très inégales entre pelles ;
- **distance** = − distance du cycle décidé (km) / 5 : trajet vers la pelle + pelle →
  verse + retour. 0 pour ATTENDRE ;
- **attente** = − min(attente / 30 min, 1) : file à la pelle + file à la verse pour un
  cycle, durée de l'ATTENDRE sinon. Plafonnée à −1 à partir de 30 min.

Poids par défaut (``Scenario.reward_weights``) : w1 = 1, w2 = 0,1, w3 = 0,05, w4 = 0,3.
La production domine : la récompense cumulée d'un poste est proche du nombre de cycles
terminés, diminuée surtout par la distance.

Ce module est un déplacement du calcul de ``MineEnv._compute_reward`` : les valeurs
sont identiques à celles de la v1 (test de non-régression).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

VARIANCE_SCALE_MIN2 = 100.0
DISTANCE_SCALE_KM = 5.0
WAIT_SCALE_MIN = 30.0


@dataclass(frozen=True)
class RewardWeights:
    """Poids w1 (production), w2 (équilibre), w3 (distance), w4 (attente)."""

    production: float = 1.0
    balance: float = 0.1
    distance: float = 0.05
    wait: float = 0.3

    @classmethod
    def from_tuple(cls, weights: Sequence[float]) -> RewardWeights:
        """(w1, w2, w3[, w4]) ; w4 = 0,3 s'il manque, comme dans la v1."""
        w4 = weights[3] if len(weights) > 3 else 0.3
        return cls(float(weights[0]), float(weights[1]), float(weights[2]), float(w4))


@dataclass(frozen=True)
class RewardTerms:
    """Les quatre termes, avant pondération."""

    production: float
    balance: float
    distance: float
    wait: float

    def weighted(self, weights: RewardWeights) -> dict[str, float]:
        """Contribution de chaque terme à la récompense : poids × terme."""
        return {name: getattr(weights, name) * value for name, value in asdict(self).items()}


def reward_terms(
    delivered_tonnage_t: float,
    capacity_t: float,
    route_distance_km: float,
    shovel_waits_min: Sequence[float],
    operational_wait_min: float,
) -> RewardTerms:
    """Les quatre termes d'une décision (voir la docstring du module)."""
    production = delivered_tonnage_t / max(capacity_t, 1e-9)

    if len(shovel_waits_min) >= 2:
        mean_wait = sum(shovel_waits_min) / len(shovel_waits_min)
        variance = sum((w - mean_wait) ** 2 for w in shovel_waits_min) / len(shovel_waits_min)
        balance = -variance / VARIANCE_SCALE_MIN2
    else:
        balance = 0.0

    distance = -route_distance_km / DISTANCE_SCALE_KM
    wait = -min(operational_wait_min / WAIT_SCALE_MIN, 1.0)
    return RewardTerms(production, balance, distance, wait)


def compute_reward(terms: RewardTerms, weights: RewardWeights) -> float:
    """R = w1 · production + w2 · équilibre + w3 · distance + w4 · attente."""
    return (
        weights.production * terms.production
        + weights.balance * terms.balance
        + weights.distance * terms.distance
        + weights.wait * terms.wait
    )
