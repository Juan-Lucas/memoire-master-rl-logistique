"""Attente prévue d'un camion sur une route pelle → verse (calcul commun).

Utilisé par l'heuristique MTWT (T23) pour choisir sa route, et par l'observation des agents
(bloc ``route_wait``, piste 1 de la mise au point) : les deux reçoivent ainsi exactement la
même information, calculée au même endroit.

Le camion part de ``location`` à l'heure ``now`` ; il arrive à la pelle après le trajet
moyen à vide, attend qu'elle soit libre (libération prévue, camions en route compris :
``MineEnv.forecast_free_at``), charge pendant la durée moyenne, puis rejoint la verse
après le trajet moyen chargé et attend qu'elle soit libre.
"""

from __future__ import annotations

from dataclasses import dataclass

from memoire_master_rl_logistique.simulation.graph_model import RoadGraph


@dataclass(frozen=True)
class RouteForecast:
    """Arrivées et attentes prévues sur une route, en minutes (heures absolues du poste)."""

    shovel_arrival: float
    shovel_wait: float
    dump_arrival: float
    dump_wait: float

    @property
    def total_wait(self) -> float:
        return self.shovel_wait + self.dump_wait


def forecast_route(
    graph: RoadGraph, location: str, now: float, shovel_node: str, dump_node: str,
    shovel_free_at: float, dump_free_at: float, load_time_min: float,
) -> RouteForecast:
    """Arrivées et attentes prévues pour la route ``shovel_node`` → ``dump_node``."""
    shovel_arrival = now + graph.mean_route_time_minutes(location, shovel_node, loaded=False)
    shovel_wait = max(0.0, shovel_free_at - shovel_arrival)
    dump_arrival = (shovel_arrival + shovel_wait + load_time_min
                    + graph.mean_route_time_minutes(shovel_node, dump_node, loaded=True))
    return RouteForecast(shovel_arrival, shovel_wait, dump_arrival,
                         max(0.0, dump_free_at - dump_arrival))
