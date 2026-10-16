"""Entités du système de transport : camions, pelles et verses.

Les durées sont en minutes ; ``available_at_min`` est l'heure à laquelle
l'entité redevient libre.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(slots=True)
class Truck:
    """Camion : disponibilité et compteurs cumulés sur le poste.

    La position et le statut du camion sont gardés par ``MineEnv``
    (``truck_locations``, ``truck_statuses``). Le détail de chaque cycle est dans
    ``MineEnv.cycle_log`` (``log_cycles=True``, voir ``CycleRecord``).

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


@dataclass(frozen=True, slots=True)
class CycleRecord:
    """Une décision d'un camion, minute par minute (journal des cycles, T12).

    ``kind`` vaut ``"cycle"`` (une affectation pelle → verse) ou ``"wait"`` (action
    ATTENDRE, seule ``voluntary_wait_min`` est alors non nulle). Les durées sont les
    durées simulées complètes, sans borne au poste : ``end_min − start_min`` est
    exactement la somme des composantes (``components_min``).

    Depuis T16 bis, un cycle se termine à la verse : ``travel_to_shovel_min`` est le
    trajet à vide depuis le lieu de la décision (parc ou verse du cycle précédent), et il
    n'y a plus de retour vers la pelle (l'ancienne composante ``travel_back_min``).
    """

    kind: str
    truck_id: str
    start_min: float
    end_min: float
    shovel_id: str | None = None
    dump_id: str | None = None
    travel_to_shovel_min: float = 0.0
    shovel_wait_min: float = 0.0
    load_min: float = 0.0
    travel_to_dump_min: float = 0.0
    dump_wait_min: float = 0.0
    unload_min: float = 0.0
    repair_min: float = 0.0
    voluntary_wait_min: float = 0.0
    delivered_t: float = 0.0  # 0 si le cycle se termine après la fin du poste

    COMPONENTS = (
        "travel_to_shovel_min", "shovel_wait_min", "load_min", "travel_to_dump_min",
        "dump_wait_min", "unload_min", "repair_min", "voluntary_wait_min",
    )

    @property
    def components_min(self) -> float:
        """Somme des durées élémentaires du cycle."""
        return sum(getattr(self, name) for name in self.COMPONENTS)

    def as_dict(self) -> dict[str, object]:
        """Dictionnaire plat, pratique pour construire un DataFrame."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
