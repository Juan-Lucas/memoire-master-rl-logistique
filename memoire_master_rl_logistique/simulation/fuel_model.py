def estimate_travel_fuel_l(
    distance_km: float,
    slope_pct: float,
    road_state: float,
    loaded: bool,
) -> float:
    """Approximation simple de la consommation sur un segment."""

    base_l_per_km = 0.65 if loaded else 0.45
    slope_factor = 1.0 + max(0.0, slope_pct) * 0.03
    road_factor = 1.0 + max(0.0, 1.0 - road_state) * 0.5
    return max(0.0, distance_km * base_l_per_km * slope_factor * road_factor)


def estimate_idle_fuel_l(idle_minutes: float, idle_l_per_hour: float = 10.0) -> float:
    """Carburant consommé en attente/chargement/déchargement moteur allumé."""

    return max(0.0, idle_l_per_hour * (idle_minutes / 60.0))
