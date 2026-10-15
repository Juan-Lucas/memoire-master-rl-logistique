"""Point d'entrée en ligne de commande.

Tous les modèles sont rangés dans ``data/results/<algo>_<scénario>`` (``rl/model_paths``) :
un entraînement lancé ici écrit exactement là où le benchmark relit. Sans option, le
pipeline complet lance le benchmark des trois régimes, qui entraîne les modèles absents.

Usage :
  python -m memoire_master_rl_logistique.main
  python -m memoire_master_rl_logistique.main --train-only
  python -m memoire_master_rl_logistique.main --train-q-learning
  python -m memoire_master_rl_logistique.main --train-sarsa
  python -m memoire_master_rl_logistique.main --train-dqn
  python -m memoire_master_rl_logistique.main --benchmark-only
  python -m memoire_master_rl_logistique.main --check-env
  python -m memoire_master_rl_logistique.main --visualize-graph
"""

from __future__ import annotations

import argparse

from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.rl.model_paths import RESULTS_DIR, model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS


def run_check_env(
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
) -> None:
    """Valide l'environnement Gymnasium avec check_env."""
    from gymnasium.utils.env_checker import check_env

    from memoire_master_rl_logistique.env.mine_env import MineEnv

    print("=" * 60)
    print("Validation de l'environnement Gymnasium")
    print("=" * 60)

    env = MineEnv(
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
    )
    check_env(env, skip_render_check=True)

    print("\ncheck_env OK")
    print(f"  Observation : {env.observation_space}")
    print(f"  Action      : {env.action_space}")
    print(f"  Camions={truck_count}, Pelles={shovel_count}, Dumps={dump_count}")

def training_dir(algorithm: str, truck_count: int, shovel_count: int, dump_count: int) -> str:
    """Dossier où ranger un modèle entraîné depuis la ligne de commande.

    Configuration nominale → le dossier relu par le benchmark (``<algo>_nominal``) ;
    toute autre configuration → un dossier ``<algo>_custom_…`` distinct, pour ne jamais
    mélanger un modèle hors scénario avec ceux de l'article.
    """
    nominal = SCENARIOS["nominal"]
    if (truck_count, shovel_count, dump_count) == (
        nominal.truck_count, nominal.shovel_count, nominal.dump_count
    ):
        name = "nominal"
    else:
        name = f"custom_{truck_count}t_{shovel_count}s_{dump_count}d"
    return str(model_dir(RESULTS_DIR, algorithm, name, TRAIN_SEEDS[0]))


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Système de dispatching minier par RL",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Entraîner uniquement l'agent PPO",
    )
    parser.add_argument(
        "--train-q-learning", action="store_true",
        help="Entraîner l'agent Q-Learning tabulaire",
    )
    parser.add_argument(
        "--train-sarsa", action="store_true",
        help="Entraîner l'agent SARSA tabulaire",
    )
    parser.add_argument(
        "--train-dqn", action="store_true",
        help="Entraîner l'agent DQN",
    )
    parser.add_argument(
        "--episodes", type=int, default=30_000,
        help="Nombre d'épisodes pour Q-Learning/SARSA (défaut: 30000)",
    )
    parser.add_argument(
        "--benchmark-only", action="store_true",
        help="Exécuter uniquement le benchmark comparatif",
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="Valider l'environnement Gymnasium (check_env)",
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
        "--timesteps", type=int, default=2_000_000,
        help="Nombre de steps d'entraînement PPO/DQN (défaut: 2000000)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Entraînement : remplacer un modèle existant",
    )
    parser.add_argument(
        "--allow-legacy-models", action="store_true",
        help="Benchmark : accepter des modèles sans metadata.json (antérieurs à T8), sans contrôle",
    )
    parser.add_argument(
        "--visualize-graph", action="store_true",
        help="Générer la figure du réseau routier minier",
    )

    args = parser.parse_args()

    if args.visualize_graph:
        from memoire_master_rl_logistique.plots.mine_graph import (
            generate_mine_graph_figure,
        )

        generate_mine_graph_figure(
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
        )
        return

    if args.check_env:
        run_check_env(
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
        )
        return

    fleet = (args.truck_count, args.shovel_count, args.dump_count)

    if args.train_q_learning:
        train_q_learning(
            n_episodes=args.episodes,
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
            output_dir=training_dir("Q-Learning", *fleet),
            overwrite=args.overwrite,
        )
        return

    if args.train_sarsa:
        train_sarsa(
            n_episodes=args.episodes,
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
            output_dir=training_dir("SARSA", *fleet),
            overwrite=args.overwrite,
        )
        return

    if args.train_dqn:
        train_dqn(
            total_timesteps=args.timesteps,
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
            output_dir=training_dir("DQN", *fleet),
            overwrite=args.overwrite,
        )
        return

    if args.train_only:
        train_ppo(
            total_timesteps=args.timesteps,
            truck_count=args.truck_count,
            shovel_count=args.shovel_count,
            dump_count=args.dump_count,
            output_dir=training_dir("PPO", *fleet),
            overwrite=args.overwrite,
        )
        return

    if args.benchmark_only:
        run_all_benchmarks(
            output_dir=str(RESULTS_DIR),
            train_ppo_flag=True,
            scenario_names=None,
            allow_legacy_models=args.allow_legacy_models,
        )
        return

    # Pipeline complet
    print("\n" + "=" * 60)
    print("PIPELINE COMPLET — Dispatching minier par RL")
    print("=" * 60)

    # Benchmark des trois régimes ; les modèles absents sont entraînés dans le bon dossier.
    run_all_benchmarks(
        output_dir=str(RESULTS_DIR),
        train_ppo_flag=True,
        scenario_names=MAIN_SCENARIOS,
        allow_legacy_models=args.allow_legacy_models,
    )


if __name__ == "__main__":
    main()
