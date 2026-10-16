"""Politique fondée sur un programme linéaire, sur le principe de DISPATCH (T24, M25).

DISPATCH (Soumis et al., 1989 ; White et Olson, 1986) sépare deux étages :

1. **Planification** (une fois, en début de poste) : un programme linéaire fixe des
   **taux de flux cibles** x_jk (cycles par heure) sur chaque route pelle j → verse k,
   ainsi que les flux à vide y_kj de la verse k vers la pelle j. On maximise la
   production Σ x_jk sous les contraintes :
   - capacité de chaque pelle : Σ_k x_jk ≤ u × 60 / durée moyenne de chargement ;
   - capacité de chaque verse : Σ_j x_jk ≤ u × 60 / durée moyenne de déversement ;
   - conservation des camions : à chaque verse, ce qui arrive chargé repart à vide
     (Σ_j x_jk = Σ_j y_kj) ; à chaque pelle, ce qui arrive à vide repart chargé
     (Σ_k y_kj = Σ_k x_jk) ;
   - heures-camions : le temps de tous les cycles planifiés tient dans la flotte
     disponible, Σ x_jk (chargement + trajet chargé + déversement + réparation moyenne)
     + Σ y_kj (trajet à vide) ≤ 60 × nombre de camions.

   u ≤ 1 est l'**utilisation maximale** planifiée d'une ressource. Avec u = 1, le plan
   sature la pelle la plus proche (30 chargements par heure) : optimal sans aléa, mais
   avec des durées aléatoires, une ressource utilisée à 100 % accumule une file sans
   limite (théorie des files d'attente). u est choisi sur les graines de sélection
   (``select_max_utilization``), jamais sur celles de l'évaluation.

   Un second programme garde la production maximale (à 0,1 % près) et minimise le
   temps de trajet total : entre deux plans aussi productifs, le moins coûteux.

2. **Affectation en temps réel** (à chaque décision) : le camion est envoyé sur la route
   la plus **en retard sur son plan**. Pour la route (j, k), le nombre de cycles qui
   devraient avoir été lancés à l'heure où le camion atteindrait la pelle j est
   x_jk × (t + trajet vers j) / 60 ; le retard est l'écart avec le nombre de cycles déjà
   lancés, exprimé en minutes de plan (÷ x_jk / 60). En cas d'égalité, la route dont la
   pelle est la plus proche.

Données utilisées : temps de trajet **moyens** du graphe, durées moyennes de service,
probabilité de panne et durée moyenne de réparation. Aucune information sur les aléas
d'un épisode. La règle ne mémorise que les cycles lancés pendant l'épisode en cours :
ce compteur est remis à zéro au début de chaque épisode (``info["decisions"] == 0``),
pour qu'un épisode ne dépende que de sa graine (évaluation parallèle, T29).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.simulation.graph_model import RoadGraph

# Durée moyenne d'une réparation (min) : uniforme sur [10, 30] dans MineEnv._arrive_at_dump.
MEAN_REPAIR_MIN = 20.0
# Tolérance sur la production maximale dans le second programme (minimiser les trajets).
PRODUCTION_TOLERANCE = 1e-3


class LPPlan:
    """Taux de flux cibles (cycles par heure) issus du programme linéaire de planification."""

    def __init__(
        self,
        graph: RoadGraph,
        shovel_node_ids: Sequence[str],
        dump_node_ids: Sequence[str],
        truck_count: int,
        load_time_min: Sequence[float],
        unload_time_min: Sequence[float],
        breakdown_probability: float,
        max_utilization: float = 1.0,
    ) -> None:
        if not 0 < max_utilization <= 1:
            raise ValueError(f"max_utilization doit être dans ]0, 1] : {max_utilization}")
        self.max_utilization = max_utilization
        n_s, n_d = len(shovel_node_ids), len(dump_node_ids)
        self.shovel_count, self.dump_count = n_s, n_d
        loaded = np.array([[graph.mean_route_time_minutes(s, d, loaded=True)
                            for d in dump_node_ids] for s in shovel_node_ids])
        empty = np.array([[graph.mean_route_time_minutes(d, s, loaded=False)
                           for s in shovel_node_ids] for d in dump_node_ids])
        # Minutes-camion d'un cycle planifié sur (j, k), hors trajet à vide (compté dans y).
        cycle_min = (np.asarray(load_time_min)[:, None] + loaded
                     + np.asarray(unload_time_min)[None, :]
                     + breakdown_probability * MEAN_REPAIR_MIN)

        # Variables : x (n_s × n_d, chargé), puis y (n_d × n_s, à vide).
        n_x, n_y = n_s * n_d, n_d * n_s

        def x(j, k):
            return j * n_d + k

        def y(k, j):
            return n_x + k * n_s + j

        a_ub, b_ub = [], []
        for j in range(n_s):  # capacité des pelles
            row = np.zeros(n_x + n_y)
            row[[x(j, k) for k in range(n_d)]] = 1.0
            a_ub.append(row)
            b_ub.append(max_utilization * 60.0 / load_time_min[j])
        for k in range(n_d):  # capacité des verses
            row = np.zeros(n_x + n_y)
            row[[x(j, k) for j in range(n_s)]] = 1.0
            a_ub.append(row)
            b_ub.append(max_utilization * 60.0 / unload_time_min[k])
        truck_row = np.concatenate([cycle_min.ravel(), empty.ravel()])  # heures-camions
        a_ub.append(truck_row)
        b_ub.append(60.0 * truck_count)

        a_eq, b_eq = [], []
        for k in range(n_d):  # verse : arrivées chargées = départs à vide
            row = np.zeros(n_x + n_y)
            row[[x(j, k) for j in range(n_s)]] = 1.0
            row[[y(k, j) for j in range(n_s)]] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)
        for j in range(n_s):  # pelle : arrivées à vide = départs chargés
            row = np.zeros(n_x + n_y)
            row[[y(k, j) for k in range(n_d)]] = 1.0
            row[[x(j, k) for k in range(n_d)]] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)

        production = np.concatenate([np.ones(n_x), np.zeros(n_y)])
        first = linprog(-production, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                        bounds=(0, None), method="highs")
        if not first.success:
            raise RuntimeError(f"Programme linéaire de planification sans solution : {first.message}")
        self.max_production_per_h = float(-first.fun)

        # Second programme : production maximale (à la tolérance près), trajets minimaux.
        travel = np.concatenate([loaded.ravel(), empty.ravel()])
        second = linprog(
            travel,
            A_ub=[*a_ub, -production],
            b_ub=[*b_ub, -(1 - PRODUCTION_TOLERANCE) * self.max_production_per_h],
            A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs",
        )
        solution = second.x if second.success else first.x
        self.rates_per_h = solution[:n_x].reshape(n_s, n_d)
        self.empty_rates_per_h = solution[n_x:].reshape(n_d, n_s)
        self.truck_minutes_per_h = float(truck_row @ solution)

    @property
    def routes(self) -> list[tuple[int, int]]:
        """Routes (pelle, verse) qui reçoivent un flux dans le plan."""
        return [(j, k) for j in range(self.shovel_count) for k in range(self.dump_count)
                if self.rates_per_h[j, k] > 1e-9]


def select_max_utilization(
    candidates: Sequence[float], evaluate: Callable[[float], float],
) -> tuple[float, dict[float, float]]:
    """Utilisation maximale de meilleur critère, et le critère de chaque candidat.

    ``evaluate(u)`` renvoie la productivité moyenne de la règle avec ce plafond, sur les
    graines de **sélection** de tous les régimes (voir ``tests/test_lp_policy.py`` et
    ``experiments.policies``). À égalité, le plus petit plafond (le plan le plus prudent).
    """
    scores = {u: evaluate(u) for u in candidates}
    best = max(sorted(scores), key=lambda u: scores[u])
    return best, scores


class LPDispatchPolicy:
    """Route la plus en retard sur le plan du programme linéaire (DISPATCH)."""

    def __init__(self, graph: RoadGraph, shovel_node_ids: Sequence[str],
                 dump_node_ids: Sequence[str], plan: LPPlan) -> None:
        self.graph = graph
        self.shovel_node_ids = list(shovel_node_ids)
        self.dump_node_ids = list(dump_node_ids)
        self.plan = plan
        self.dispatched = np.zeros_like(plan.rates_per_h)

    def choose(self, truck_location: str, now: float) -> tuple[int, int]:
        """(pelle, verse) de plus grand retard, en minutes de plan."""
        def lateness(route: tuple[int, int]) -> tuple[float, float]:
            j, k = route
            travel = self.graph.mean_route_time_minutes(truck_location, self.shovel_node_ids[j],
                                                        loaded=False)
            rate_per_min = self.plan.rates_per_h[j, k] / 60.0
            due = rate_per_min * (now + travel)
            return (due - self.dispatched[j, k]) / rate_per_min, -travel

        return max(self.plan.routes, key=lateness)

    def predict(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
        truck_location: str | None = None,
    ) -> int:
        """Action de la route la plus en retard ; action 0 sans ``info``."""
        if info is None or truck_location is None:
            return 0
        if info["decisions"] == 0:  # nouvel épisode : le plan repart de zéro
            self.dispatched[:] = 0
        shovel, dump = self.choose(truck_location, info["current_time_min"])
        self.dispatched[shovel, dump] += 1
        return encode_action(shovel, dump, len(self.dump_node_ids))
