"""Point d'entrée principal du système de dispatching minier.

Pipeline complet :
  1. Simulation DES pour validation du moteur
  2. Entraînement de l'agent PPO (Section 4.4)
  3. Évaluation comparative baselines vs PPO (Section 4.5)
  4. Génération du rapport statistique (Chapitre 6)

Usage :
  python -m memoire_master_rl_logistique.main
  python -m memoire_master_rl_logistique.main --sim-only
  python -m memoire_master_rl_logistique.main --train-only
  python -m memoire_master_rl_logistique.main --benchmark-only
"""

from __future__ import annotations

import argparse

from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.simulation.events import build_simulation


def run_simulation_demo(
    seed: int = 42,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
) -> None:
    """Lance une simulation DES pour valider le moteur."""
    print("=" * 60)
    print("Simulation DES — Validation du moteur")
    print("=" * 60)

    sim = build_simulation(
        seed=seed,
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
    )

    result = sim.run_episode(episode_minutes=480.0)

    print("\nKPIs globaux :")
    for key, val in result.kpis.items():
        print(f"  {key}: {val:.2f}")

    print("\nDétails par camion :")
    for truck_info in result.per_truck:
        tid = truck_info["truck_id"]
        cycles = truck_info["cycles_completed"]
        tonnage = truck_info["total_tonnage_t"]
        wait = truck_info["total_wait_min"]
        print(f"  {tid}: {int(cycles)} cycles, {tonnage:.0f}t, wait={wait:.1f}min")


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Système de dispatching minier par RL",
    )
    parser.add_argument(
        "--sim-only", action="store_true",
        help="Exécuter uniquement la simulation DES",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Exécuter uniquement l'entraînement PPO",
    )
    parser.add_argument(
        "--benchmark-only", action="store_true",
        help="Exécuter uniquement le benchmark comparatif",
    )
    parser.add_argument(
        "--truck-count", type=int, default=12,
        help="Nombre de camions (défaut: 12)",
    )
    parser.add_argument(
        "--shovel-count", type=int, default=3,
        help="Nombre de pelles (défaut: 3)",
    )
    parser.add_argument(
        "--dump-count", type=int, default=2,
        help="Nombre de dumps (défaut: 2)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000,
        help="Nombre de steps d'entraînement PPO (défaut: 50000)",
    )

    args = parser.parse_args()

    if args.sim_only:
        run_simulation_demo(
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
        )
        return

    if args.train_only:
        train_ppo(
            total_timesteps=args.timesteps,
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
        )
        return

    if args.benchmark_only:
        run_all_benchmarks(
            output_dir="data/results",
            train_ppo_flag=True,
            scenario_names=["nominal"],
        )
        return

    # Pipeline complet
    print("\n" + "=" * 60)
    print("PIPELINE COMPLET — Dispatching minier par RL")
    print("=" * 60)

    # Étape 1 : Simulation
    run_simulation_demo(
        truck_count=args.truck_count,
        shovel_count=args.shovel_count,
        dump_count=args.dump_count,
    )

    # Étape 2 : Entraînement PPO
    print("\n" + "=" * 60)
    print("Entraînement PPO")
    print("=" * 60)
    train_ppo(
        total_timesteps=args.timesteps,
        truck_count=args.truck_count,
        shovel_count=args.shovel_count,
        dump_count=args.dump_count,
    )

    # Étape 3 : Benchmark
    print("\n" + "=" * 60)
    print("Benchmark comparatif")
    print("=" * 60)
    run_all_benchmarks(
        output_dir="data/results",
        train_ppo_flag=True,
        scenario_names=["nominal"],
    )


if __name__ == "__main__":
    main()
