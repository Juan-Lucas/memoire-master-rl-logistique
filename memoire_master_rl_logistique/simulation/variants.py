"""Variantes du simulateur pour l'analyse de sensibilité (M2 du rapport du 17/11/2026).

Une variante change un paramètre physique de la mine : durées de chargement et de
déversement, vitesses, pente des retours à vide, usure des pistes. La variante nominale
(``NOMINAL_VARIANT``) laisse le simulateur exactement tel qu'il est : c'est celle de toute
la campagne. Le carburant a ses propres paramètres (``fuel_model.FuelParams``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Probabilité, à chaque passage sur une piste, qu'elle perde 0,02 d'état (plancher 0,6).
ROAD_WEAR_PROBABILITY = 0.05


@dataclass(frozen=True)
class SimulationVariant:
    """Facteurs appliqués au simulateur nominal.

    - ``load_time_scale``, ``unload_time_scale`` : moyenne et écart type des durées de
      chargement et de déversement (coefficient de variation inchangé) ;
    - ``speed_scale`` : vitesses de base, à vide et chargé ;
    - ``flat_returns`` : retours à vide verse → pelle sans pente (au lieu d'une montée de
      2 à 5 %), borne entre les montées du modèle et les descentes d'une mine réelle ;
    - ``road_wear_probability`` : usure des pistes à chaque passage.
    """

    load_time_scale: float = 1.0
    unload_time_scale: float = 1.0
    speed_scale: float = 1.0
    flat_returns: bool = False
    road_wear_probability: float = ROAD_WEAR_PROBABILITY

    def __post_init__(self) -> None:
        for name in ("load_time_scale", "unload_time_scale", "speed_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être > 0 : {getattr(self, name)}")
        if not 0.0 <= self.road_wear_probability <= 1.0:
            raise ValueError(f"road_wear_probability hors de [0, 1] : {self.road_wear_probability}")


NOMINAL_VARIANT = SimulationVariant()
