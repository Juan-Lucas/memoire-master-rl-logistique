from dataclasses import dataclass, field


@dataclass(slots=True)
class Truck:
    """Entité minimale de camion utilisée dans la simulation MVP."""

    truck_id: str
    capacity_tonnes: float
    available_at_min: float = 0.0
    total_wait_min: float = 0.0
    total_active_min: float = 0.0
    cycles_completed: int = 0
    total_tonnage_t: float = 0.0
    total_fuel_l: float = 0.0
    history: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class Shovel:
    """Ressource de chargement."""

    shovel_id: str
    node_id: str
    load_time_mean_min: float
    load_time_std_min: float
    available_at_min: float = 0.0


@dataclass(slots=True)
class DumpSite:
    """Ressource de déchargement."""

    dump_id: str
    node_id: str
    unload_time_mean_min: float
    unload_time_std_min: float
    available_at_min: float = 0.0
