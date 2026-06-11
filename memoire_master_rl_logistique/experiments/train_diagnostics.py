"""Ré-entraînement Q-Learning / SARSA avec journalisation de l'epsilon et de
l'erreur TD par épisode (figures de diagnostic RL, §6.3.2), pour un scénario
expérimental donné (nominal, high_load, high_breakdown).

Usage : python -m memoire_master_rl_logistique.experiments.train_diagnostics {q_learning,sarsa} {nominal,high_load,high_breakdown}
"""

from __future__ import annotations

import sys

from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa


def main() -> None:
    algo = sys.argv[1] if len(sys.argv) > 1 else "q_learning"
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "nominal"
    scenario = SCENARIOS[scenario_name]

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
        train_q_learning(output_dir=f"data/results/q_learning_{scenario_name}_diag", **common)
    elif algo == "sarsa":
        train_sarsa(output_dir=f"data/results/sarsa_{scenario_name}_diag", **common)
    else:
        raise ValueError(f"Algo inconnu : {algo}")


if __name__ == "__main__":
    main()
