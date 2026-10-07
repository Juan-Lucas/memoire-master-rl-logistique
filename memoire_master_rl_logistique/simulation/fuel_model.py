from memoire_master_rl_logistique.simulation.constants import SLOPE_PENALTY_PER_PCT


def estimate_travel_fuel_l(
    distance_km: float,
    slope_pct: float,
    road_state: float,
    loaded: bool,
) -> float:
    """Carburant (L) sur un segment : 0,65 L/km chargé, 0,45 L/km à vide,
    majoré par la pente et par l'état de la piste. Valeurs non calibrées."""

    base_l_per_km = 0.65 if loaded else 0.45
    slope_factor = 1.0 + max(0.0, slope_pct) * SLOPE_PENALTY_PER_PCT
    road_factor = 1.0 + max(0.0, 1.0 - road_state) * 0.5
    return max(0.0, distance_km * base_l_per_km * slope_factor * road_factor)


def estimate_idle_fuel_l(idle_minutes: float, idle_l_per_hour: float = 10.0) -> float:
    """Carburant (L) moteur au ralenti : file, chargement, déversement, panne."""

    return max(0.0, idle_l_per_hour * (idle_minutes / 60.0))
