def compute_kpis(
    episode_minutes: float,
    truck_count: int,
    total_tonnage_t: float,
    total_wait_min: float,
    total_active_min: float,
    total_fuel_l: float,
    total_cycles: int = 0,
    total_distance_km: float = 0.0,
) -> dict[str, float]:
    """Calcule les indicateurs d'un poste.

    ``avg_wait_min_per_truck`` est l'attente cumulée par camion sur tout le
    poste (files, réparations et action ATTENDRE confondues), pas par cycle.
    ``utilization_pct`` compte aussi le temps actif du dernier cycle quand il
    déborde après la fin du poste.
    """
    episode_hours = max(episode_minutes / 60.0, 1e-9)
    max_available_minutes = max(float(truck_count) * episode_minutes, 1e-9)

    productivity_tph = total_tonnage_t / episode_hours
    avg_wait_min_per_truck = total_wait_min / max(float(truck_count), 1.0)
    utilization_pct = (total_active_min / max_available_minutes) * 100.0
    specific_fuel_l_per_ton = total_fuel_l / max(total_tonnage_t, 1e-9)

    # Carburant moyen par cycle (L)
    cost_per_cycle = total_fuel_l / max(float(total_cycles), 1.0)

    return {
        "episode_minutes": float(episode_minutes),
        "truck_count": float(truck_count),
        "total_tonnage_t": float(total_tonnage_t),
        "productivity_tph": float(productivity_tph),
        "total_wait_min": float(total_wait_min),
        "avg_wait_min_per_truck": float(avg_wait_min_per_truck),
        "utilization_pct": float(utilization_pct),
        "total_fuel_l": float(total_fuel_l),
        "specific_fuel_l_per_ton": float(specific_fuel_l_per_ton),
        "total_cycles": float(total_cycles),
        "cost_per_cycle": float(cost_per_cycle),
    }
