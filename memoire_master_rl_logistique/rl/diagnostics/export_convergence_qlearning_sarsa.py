"""Export des courbes de convergence Q-Learning/SARSA pour Power BI.

Source : data/results/{q_learning,sarsa}_{nominal,high_load,high_breakdown}/training_rewards.csv
Sortie : data/results/convergence_qlearning_sarsa_powerbi.csv
Colonnes : episode, reward, reward_smooth, algorithm, scenario
"""

from __future__ import annotations

import pandas as pd

from memoire_master_rl_logistique.experiments.scenarios import SCENARIO_DISPLAYS

SCENARIOS = [(s.key, s.label) for s in SCENARIO_DISPLAYS]

ALGOS = [
    ("q_learning", "Q-Learning"),
    ("sarsa", "SARSA"),
]


def main() -> None:
    base = "data/results"
    frames = []

    for scenario_dir, scenario_label in SCENARIOS:
        for algo_dir, algo_label in ALGOS:
            df = pd.read_csv(f"{base}/{algo_dir}_{scenario_dir}/training_rewards.csv")
            df = df.rename(columns={"total_reward": "reward"})
            df["reward_smooth"] = df["reward"].rolling(100, min_periods=1).mean()
            df["algorithm"] = algo_label
            df["scenario"] = scenario_label
            frames.append(df[["episode", "reward", "reward_smooth", "algorithm", "scenario"]])

    out = pd.concat(frames, ignore_index=True)
    out_path = f"{base}/convergence_qlearning_sarsa_powerbi.csv"
    out.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path} ({len(out)} lignes)")


if __name__ == "__main__":
    main()
