import csv
from pathlib import Path

from .events import build_default_simulation


def write_single_row_csv(path: Path, row: dict[str, float]) -> None:
    """Écrit une seule ligne de données dans un fichier CSV, en créant les répertoires parents si nécessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_table_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Écrit une table de données dans un fichier CSV, en créant les répertoires parents si nécessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Point d'entrée pour exécuter la simulation du MVP du Sprint A et exporter les résultats au format CSV."""
    simulation = build_default_simulation(seed=42, truck_count=5, shovel_count=2)
    result = simulation.run_episode(episode_minutes=8.0 * 60.0)

    out_dir = Path("data") / "interim"
    kpi_path = out_dir / "sprint_a_kpis.csv"
    truck_path = out_dir / "sprint_a_truck_details.csv"

    write_single_row_csv(kpi_path, result.kpis)
    write_table_csv(truck_path, result.per_truck)

    print("MVP Sprint A termine")
    print(f"CSV KPI: {kpi_path}")
    print(f"CSV details camions: {truck_path}")
    print(f"Productivite (t/h): {result.kpis['productivity_tph']:.2f}")
    print(f"Attente moyenne/camion (min): {result.kpis['avg_wait_min_per_truck']:.2f}")


if __name__ == "__main__":
    main()
