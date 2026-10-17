"""Modèle de consommation de carburant (T14 : paramétrable et calibré).

Les valeurs par défaut viennent de ``constants.py``, avec leur source et leur
calibration. ``FuelParams`` permet de les changer pour une étude de sensibilité,
sans modifier le code.
"""

from __future__ import annotations

from dataclasses import dataclass

from memoire_master_rl_logistique.simulation.constants import (
    FUEL_IDLE_L_PER_H,
    FUEL_L_PER_KM_EMPTY,
    FUEL_L_PER_KM_LOADED,
    FUEL_ROAD_STATE_PENALTY,
    SLOPE_PENALTY_PER_PCT,
)


@dataclass(frozen=True)
class FuelParams:
    """Paramètres du modèle de carburant."""

    l_per_km_loaded: float = FUEL_L_PER_KM_LOADED
    l_per_km_empty: float = FUEL_L_PER_KM_EMPTY
    idle_l_per_h: float = FUEL_IDLE_L_PER_H
    road_state_penalty: float = FUEL_ROAD_STATE_PENALTY
    slope_penalty_per_pct: float = SLOPE_PENALTY_PER_PCT


DEFAULT_FUEL = FuelParams()


def estimate_travel_fuel_l(
    distance_km: float,
    slope_pct: float,
    road_state: float,
    loaded: bool,
    params: FuelParams = DEFAULT_FUEL,
) -> float:
    """Carburant (L) sur un segment : consommation au km (chargé ou à vide), majorée
    par la pente (montée seulement) et par la dégradation de la piste."""
    base_l_per_km = params.l_per_km_loaded if loaded else params.l_per_km_empty
    slope_factor = 1.0 + max(0.0, slope_pct) * params.slope_penalty_per_pct
    road_factor = 1.0 + max(0.0, 1.0 - road_state) * params.road_state_penalty
    return max(0.0, distance_km * base_l_per_km * slope_factor * road_factor)


def estimate_idle_fuel_l(idle_minutes: float, params: FuelParams = DEFAULT_FUEL) -> float:
    """Carburant (L) moteur tournant à l'arrêt : file, chargement, déversement, panne, ATTENDRE."""
    return max(0.0, params.idle_l_per_h * (idle_minutes / 60.0))
