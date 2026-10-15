"""Entités du système de transport : camions, pelles et verses.

Les durées sont en minutes ; ``available_at_min`` est l'heure à laquelle
l'entité redevient libre.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Truck:
    """Camion : disponibilité et compteurs cumulés sur le poste.

    La position et le statut du camion sont gardés par ``MineEnv``
    (``truck_locations``, ``truck_statuses``). ``history`` n'est pas utilisé.

    Tous les compteurs de temps sont bornés à la fin du poste (T11) : seule la part
    d'une activité située avant 480 min est comptée.

    Le temps d'immobilisation est décomposé en trois compteurs (T10) :
    - ``queue_wait_min`` : file d'attente à la pelle ou à la verse ;
    - ``repair_min`` : réparation après une panne ;
    - ``voluntary_wait_min`` : action ATTENDRE choisie par la politique.
    ``total_wait_min`` est leur somme (l'ancien compteur unique).
    """

    truck_id: str
    capacity_tonnes: float = 140.0
    available_at_min: float = 0.0
    queue_wait_min: float = 0.0
    repair_min: float = 0.0
    voluntary_wait_min: float = 0.0
    total_active_min: float = 0.0
    cycles_completed: int = 0
    total_tonnage_t: float = 0.0
    total_fuel_l: float = 0.0
    history: list[dict] = field(default_factory=list)

    @property
    def total_wait_min(self) -> float:
        """Immobilisation totale : file + réparation + ATTENDRE."""
        return self.queue_wait_min + self.repair_min + self.voluntary_wait_min


@dataclass(slots=True)
class Shovel:
    """Pelle : durée de chargement log-normale (moyenne et écart-type en min).

    ``busy_min`` : temps passé à charger pendant le poste (borné à la fin du poste, T11).
    """

    shovel_id: str
    node_id: str
    load_time_mean_min: float = 2.0
    load_time_std_min: float = 0.3
    available_at_min: float = 0.0
    busy_min: float = 0.0


@dataclass(slots=True)
class DumpSite:
    """Verse (point de déchargement) : durée de déversement log-normale."""

    dump_id: str
    node_id: str
    unload_time_mean_min: float = 1.0
    unload_time_std_min: float = 0.2
    available_at_min: float = 0.0
