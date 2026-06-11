"""Distribution des actions par politique (graphique en barres groupées).

Source : data/results/action_distribution.csv
Sortie : reports/figures/action_distribution_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALGOS = ["Q-Learning", "SARSA", "DQN", "PPO"]
COLORS = {"Q-Learning": "#4CAF50", "SARSA": "#9C27B0", "DQN": "#2196F3", "PPO": "#FF5722"}

SCENARIOS = [
    ("nominal", "nominal", "scénario nominal"),
    ("high_load", "highload", "scénario high_load"),
    ("high_breakdown", "highbreakdown", "scénario high_breakdown"),
]


def main() -> None:
    df = pd.read_csv("data/results/action_distribution.csv")
    out_dir = Path("reports/figures")

    for scenario_value, suffix, scenario_label in SCENARIOS:
        sub_df = df[df["scenario"] == scenario_value]
        labels = sub_df[sub_df["algorithm"] == "Q-Learning"]["action_label"].tolist()
        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, algo in enumerate(ALGOS):
            sub = sub_df[sub_df["algorithm"] == algo].set_index("action_label").loc[labels]
            ax.bar(x + (i - 1.5) * width, sub["percentage"], width, label=algo, color=COLORS[algo])

        ax.set_xlabel("Action")
        ax.set_ylabel("Fréquence (%)")
        ax.set_title(f"Distribution des actions par politique — {scenario_label} (10 épisodes)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"action_distribution_{suffix}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardée : {out_path}")


if __name__ == "__main__":
    main()