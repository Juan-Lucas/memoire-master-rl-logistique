"""Ré-entraînement Q-Learning / SARSA (scénario nominal) avec journalisation
de l'epsilon et de l'erreur TD par épisode (figures de diagnostic RL, §6.3.1).

Hyperparamètres identiques à run_benchmark.py (scénario nominal, seed=42) :
le résultat (q_table / sarsa_table) est donc identique, seul un fichier
supplémentaire `training_diagnostics.csv` (episode, total_reward, epsilon,
mean_abs_td_error) est ajouté.

Usage : python -m memoire_master_rl_logistique.experiments.train_diagnostics_nominal {q_learning,sarsa}
"""

from __future__ import annotations

import sys

from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa


def main() -> None:
    algo = sys.argv[1] if len(sys.argv) > 1 else "q_learning"
    scenario = SCENARIOS["nominal"]

    common = dict(
        n_episodes=30_000,
        alpha=0.2,
        gamma=0.99,
        n_bins=8,
        seed=scenario.seeds[0],
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
    )

    if algo == "q_learning":
        train_q_learning(output_dir="data/results/q_learning_nominal_diag", **common)
    elif algo == "sarsa":
        train_sarsa(output_dir="data/results/sarsa_nominal_diag", **common)
    else:
        raise ValueError(f"Algo inconnu : {algo}")


if __name__ == "__main__":
    main()
