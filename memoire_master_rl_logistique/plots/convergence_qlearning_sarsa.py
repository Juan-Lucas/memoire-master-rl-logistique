"""Courbes de convergence Q-Learning/SARSA (récompense lissée par épisode).

Source : data/results/convergence_qlearning_sarsa_powerbi.csv
Sortie : reports/figures/learning_curves_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from memoire_master_rl_logistique.experiments.scenarios import SCENARIO_DISPLAYS
from memoire_master_rl_logistique.plots.common import save_figure

SCENARIOS = [(s.label, s.suffix, s.label) for s in SCENARIO_DISPLAYS]

ALGO_STYLE = {
    "Q-Learning": "#4CAF50",
    "SARSA": "#9C27B0",
}


def main() -> None:
    df = pd.read_csv("data/results/convergence_qlearning_sarsa_powerbi.csv")
    out_dir = Path("reports/figures")

    for scenario_value, suffix, scenario_label in SCENARIOS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for algo, color in ALGO_STYLE.items():
            sub = df[(df["algorithm"] == algo) & (df["scenario"] == scenario_value)]
            ax.plot(sub["episode"], sub["reward_smooth"], color=color, label=algo, linewidth=1.5)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward — 100-episode moving average")
        ax.set_title(f"Q-Learning and SARSA convergence — {scenario_label} scenario")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"learning_curves_{suffix}.png"
        save_figure(fig, out_path)


if __name__ == "__main__":
    main()
