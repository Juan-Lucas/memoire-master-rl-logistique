"""Indicateurs d'un poste (KPI), calculés à partir des compteurs des camions.

L'immobilisation est décomposée (T10) : file d'attente, réparation après panne,
action ATTENDRE. Avant T10, un seul compteur mélangeait les trois, puis il était
divisé par le nombre de camions : « l'attente moyenne » était donc une somme sur
tout le poste, par camion, réparations comprises, et non une attente par cycle.
"""

from __future__ import annotations

from collections.abc import Iterable

from memoire_master_rl_logistique.simulation.entities import Truck


def compute_kpis(
    episode_minutes: float,
    truck_count: int,
    total_tonnage_t: float,
    queue_wait_min: float,
    repair_min: float,
    voluntary_wait_min: float,
    total_active_min: float,
    total_fuel_l: float,
    total_cycles: int = 0,
) -> dict[str, float]:
    """Calcule les indicateurs d'un poste à partir des totaux de la flotte.

    Unités des indicateurs d'immobilisation :
    - ``*_min`` : total de la flotte sur le poste ;
    - ``*_min_per_truck`` : par camion, cumulé sur tout le poste ;
    - ``queue_wait_min_per_cycle`` : file d'attente moyenne par cycle de transport.
    ``avg_wait_min_per_truck`` (file + réparation + ATTENDRE, par camion et par poste)
    est conservé pour comparer avec la v1.

    ``utilization_pct`` compte aussi le temps actif du dernier cycle quand il
    déborde après la fin du poste.
    """
    trucks = max(float(truck_count), 1.0)
    cycles = max(float(total_cycles), 1.0)
    episode_hours = max(episode_minutes / 60.0, 1e-9)
    max_available_minutes = max(float(truck_count) * episode_minutes, 1e-9)
    total_wait_min = queue_wait_min + repair_min + voluntary_wait_min

    return {
        "episode_minutes": float(episode_minutes),
        "truck_count": float(truck_count),
        "total_tonnage_t": float(total_tonnage_t),
        "productivity_tph": float(total_tonnage_t / episode_hours),
        "total_wait_min": float(total_wait_min),
        "avg_wait_min_per_truck": float(total_wait_min / trucks),
        "queue_wait_min": float(queue_wait_min),
        "repair_min": float(repair_min),
        "voluntary_wait_min": float(voluntary_wait_min),
        "queue_wait_min_per_truck": float(queue_wait_min / trucks),
        "repair_min_per_truck": float(repair_min / trucks),
        "voluntary_wait_min_per_truck": float(voluntary_wait_min / trucks),
        "queue_wait_min_per_cycle": float(queue_wait_min / cycles),
        "utilization_pct": float(total_active_min / max_available_minutes * 100.0),
        "total_fuel_l": float(total_fuel_l),
        "specific_fuel_l_per_ton": float(total_fuel_l / max(total_tonnage_t, 1e-9)),
        "total_cycles": float(total_cycles),
        "cost_per_cycle": float(total_fuel_l / cycles),  # carburant moyen par cycle (L)
    }


def fleet_kpis(trucks: Iterable[Truck], episode_minutes: float) -> dict[str, float]:
    """Indicateurs d'un poste à partir des camions de la flotte (fin d'épisode)."""
    trucks = list(trucks)
    return compute_kpis(
        episode_minutes=episode_minutes,
        truck_count=len(trucks),
        total_tonnage_t=sum(t.total_tonnage_t for t in trucks),
        queue_wait_min=sum(t.queue_wait_min for t in trucks),
        repair_min=sum(t.repair_min for t in trucks),
        voluntary_wait_min=sum(t.voluntary_wait_min for t in trucks),
        total_active_min=sum(t.total_active_min for t in trucks),
        total_fuel_l=sum(t.total_fuel_l for t in trucks),
        total_cycles=sum(t.cycles_completed for t in trucks),
    )
