"""Modèle de consommation de carburant (T14 ; revu le 22 octobre 2026).

Les valeurs par défaut viennent de ``constants.py``, avec leurs sources : ralenti et
effet de la pente mesurés par Kubler (2015) sur un Cat 785D, niveau moyen ramené à la
consommation horaire publiée dans le Caterpillar Performance Handbook. ``FuelParams``
permet de les changer pour une étude de sensibilité, sans modifier le code.
"""

from __future__ import annotations

from dataclasses import dataclass

from memoire_master_rl_logistique.simulation.constants import (
    FUEL_IDLE_L_PER_H,
    FUEL_KUBLER_EMPTY,
    FUEL_KUBLER_LOADED,
    FUEL_KUBLER_MAX_GRADE,
    FUEL_ROAD_STATE_PENALTY,
    FUEL_TRAVEL_SCALE,
)


@dataclass(frozen=True)
class FuelParams:
    """Paramètres du modèle de carburant.

    ``loaded_poly`` et ``empty_poly`` : coefficients (x⁰ à x³) de la consommation en L/km
    en fonction de la pente x (fraction), mesurée dans le domaine [0, ``max_grade``].
    """

    loaded_poly: tuple[float, ...] = FUEL_KUBLER_LOADED
    empty_poly: tuple[float, ...] = FUEL_KUBLER_EMPTY
    max_grade: float = FUEL_KUBLER_MAX_GRADE
    travel_scale: float = FUEL_TRAVEL_SCALE
    idle_l_per_h: float = FUEL_IDLE_L_PER_H
    road_state_penalty: float = FUEL_ROAD_STATE_PENALTY


DEFAULT_FUEL = FuelParams()


def l_per_km(slope_pct: float, loaded: bool, params: FuelParams = DEFAULT_FUEL) -> float:
    """Consommation au km sur piste en bon état, avant le facteur d'échelle.

    La pente est bornée au domaine mesuré : une descente compte comme du plat (le
    simulateur n'en a pas), une pente plus forte que ``max_grade`` comme ``max_grade``.
    """
    x = min(max(slope_pct / 100.0, 0.0), params.max_grade)
    coefficients = params.loaded_poly if loaded else params.empty_poly
    return sum(c * x ** k for k, c in enumerate(coefficients))


def estimate_travel_fuel_l(
    distance_km: float,
    slope_pct: float,
    road_state: float,
    loaded: bool,
    params: FuelParams = DEFAULT_FUEL,
) -> float:
    """Carburant (L) sur un segment : courbe de Kubler (chargé ou à vide) selon la pente,
    facteur d'échelle, majoration selon la dégradation de la piste."""
    road_factor = 1.0 + max(0.0, 1.0 - road_state) * params.road_state_penalty
    return max(0.0, distance_km * params.travel_scale * l_per_km(slope_pct, loaded, params)
               * road_factor)


def estimate_idle_fuel_l(idle_minutes: float, params: FuelParams = DEFAULT_FUEL) -> float:
    """Carburant (L) moteur tournant à l'arrêt : file, chargement, déversement, panne, ATTENDRE."""
    return max(0.0, params.idle_l_per_h * (idle_minutes / 60.0))
