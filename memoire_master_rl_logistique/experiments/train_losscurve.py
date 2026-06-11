"""Ré-entraînement DQN / PPO avec journalisation Stable-Baselines3
(sb3_logs/progress.csv) pour les figures Policy/Value Loss et Exploration vs
Exploitation (§6.3.2), pour un scénario expérimental donné (nominal,
high_load, high_breakdown).

Usage : python -m memoire_master_rl_logistique.experiments.train_losscurve {dqn,ppo} {nominal,high_load,high_breakdown}
"""

from __future__ import annotations

import sys

from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo


def main() -> None:
    algo = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "nominal"
    scenario = SCENARIOS[scenario_name]

    common = dict(
        total_timesteps=scenario.total_timesteps,
        seed=scenario.seeds[0],
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
    )

    if algo == "dqn":
        train_dqn(output_dir=f"data/results/dqn_{scenario_name}_losscurve", **common)
    elif algo == "ppo":
        train_ppo(output_dir=f"data/results/ppo_{scenario_name}_losscurve", **common)
    else:
        raise ValueError(f"Algo inconnu : {algo}")


if __name__ == "__main__":
    main()
