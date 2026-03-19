def compute_kpis(
    episode_minutes: float,
    truck_count: int,
    total_tonnage_t: float,
    total_wait_min: float,
    total_active_min: float,
    total_fuel_l: float,
) -> dict[str, float]:
    """Calcule les KPIs de performance de la simulation minière à partir des métriques agrégées."""
    episode_hours = max(episode_minutes / 60.0, 1e-9)
    max_available_minutes = max(float(truck_count) * episode_minutes, 1e-9)

    productivity_tph = total_tonnage_t / episode_hours
    avg_wait_min_per_truck = total_wait_min / max(float(truck_count), 1.0)
    utilization_pct = (total_active_min / max_available_minutes) * 100.0
    specific_fuel_l_per_ton = total_fuel_l / max(total_tonnage_t, 1e-9)

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
    }
