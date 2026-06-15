"""Entités du système de transport minier.

Correspondance avec le MDP (Tableau 4.2 du mémoire) :
- Truck  → x_c (localisation, statut) — Eq. 3.3
- Shovel → q_p (file d'attente), z_r  — Eq. 3.3
- DumpSite → z_r (disponibilité)      — Eq. 3.3
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Truck:
    """Camion minier — encode la composante x_c de l'état s_t.

    Suit en temps réel la position, le statut et les KPIs
    individuels (tonnage, carburant, temps d'attente).
    """

    truck_id: str
    capacity_tonnes: float = 140.0
    available_at_min: float = 0.0
    total_wait_min: float = 0.0
    total_active_min: float = 0.0
    cycles_completed: int = 0
    total_tonnage_t: float = 0.0
    total_fuel_l: float = 0.0
    history: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class Shovel:
    """Pelle excavatrice — encode q_p et z_r dans l'état s_t.

    Les paramètres load_time_mean_min et load_time_std_min
    permettent d'échantillonner le bruit stochastique η (Eq. 3.2).
    """

    shovel_id: str
    node_id: str
    load_time_mean_min: float = 2.0
    load_time_std_min: float = 0.3
    available_at_min: float = 0.0


@dataclass(slots=True)
class DumpSite:
    """Point de déchargement — encode z_r dans l'état s_t.

    Modélise les goulots d'étranglement potentiels en fin de
    parcours avec la même stochasticité gaussienne que les pelles.
    """

    dump_id: str
    node_id: str
    unload_time_mean_min: float = 1.0
    unload_time_std_min: float = 0.2
    available_at_min: float = 0.0
