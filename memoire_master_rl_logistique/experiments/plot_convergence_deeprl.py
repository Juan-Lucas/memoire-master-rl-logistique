"""Courbes de convergence DQN/PPO (productivité lissée par épisode).

Source : data/results/convergence_deeprl_powerbi.csv
Sortie : reports/figures/learning_curves_deeprl_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from memoire_master_rl_logistique.experiments.plot_utils import save_figure
from memoire_master_rl_logistique.experiments.scenarios import SCENARIO_DISPLAYS

SCENARIOS = [(s.label, s.suffix, s.label) for s in SCENARIO_DISPLAYS]

ALGO_STYLE = {
    "DQN": "#2196F3",
    "PPO": "#FF5722",
}


def main() -> None:
    df = pd.read_csv("data/results/convergence_deeprl_powerbi.csv")
    out_dir = Path("reports/figures")

    for scenario_value, suffix, scenario_label in SCENARIOS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for algo, color in ALGO_STYLE.items():
            sub = df[(df["algorithm"] == algo) & (df["scenario"] == scenario_value)]
            ax.plot(sub["episode"], sub["productivity_smooth"], color=color, label=algo, linewidth=1.5)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Productivity (t/h) — smoothed")
        ax.set_title(f"DQN and PPO convergence — {scenario_label} scenario")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"learning_curves_deeprl_{suffix}.png"
        save_figure(fig, out_path)


if __name__ == "__main__":
    main()
